import torch
import numpy as np
from collections import defaultdict
import bitsandbytes as bnb
from peft.tuners.lora.bnb import Linear4bit, Linear8bitLt
from peft.utils.integrations import dequantize_bnb_weight
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import random 
import copy
import re
import os
import json
from transformers.models.llama.modeling_llama import NoAttention, NoMLP

def save_base_scores(attn_inputs, mlp_inputs, save_dir):
    """
    Saves the gradients/magnitudes used to compute masks.
    score_dict = {'model.layers.0.self_attn': tensor(...), ...}
    """
    os.makedirs(save_dir, exist_ok=True)

    attn_path = os.path.join(save_dir, 'base_attn_inputs.pt')
    mlp_path = os.path.join(save_dir, 'base_mlp_inputs.pt')

    # Use torch.save (standard) or safetensors (faster loading)
    torch.save(attn_inputs, attn_path)
    torch.save(mlp_inputs, mlp_path)

def create_sparsity_buffer(sparsity_dict, width_ratio, buffer_dir='/tmp/mask_cache'):
    """
    Saves the best config found by your search.
    sparsity_dict = {'model.layers.0.self_attn': 0.5, ...}
    """
    os.makedirs(buffer_dir, exist_ok=True)
    save_path = os.path.join(buffer_dir, f'sparsity_{width_ratio}.json')

    with open(save_path, 'w') as f:
        # indent=4 makes it human-readable
        json.dump(sparsity_dict, f, indent=4)
        
    return save_path

###Evolution Search Utils
def find_closest_width(
    fisher_mask: Dict[str, torch.Tensor], 
    target_widths: List[float]
) -> float:
    """
    Calculates the actual GLOBAL WIDTH (fraction of retained params) and finds the closest target.
    
    Args:
        fisher_mask: Dict of tensors where 1=Keep, 0=Prune.
        target_widths: List of target WIDTH values (Fraction Kept).
                       e.g., 0.625 means 62.5% Kept (37.5% Pruned).
    """
    # Sort targets for cleaner logic (optional but good practice)
    sorted_targets = sorted(target_widths)
    target_tensor = torch.tensor(sorted_targets, dtype=torch.float32)
    
    total_elements = 0
    retained_elements = 0

    for name, mask in fisher_mask.items():
        total_elements += mask.numel()
        retained_elements += mask.sum().item()
        
    # 1. Calculate Global Density (Weighted Keep Ratio)
    # This matches your 'target_width' concept.
    global_density = retained_elements / total_elements
    
    # 2. Find Closest Target
    # We compare Actual Keep vs Target Keep
    diffs = torch.abs(target_tensor - global_density)
    closest_index = torch.argmin(diffs)
    # closest_target = target_tensor[closest_index].item()

    # 4. CRITICAL FIX: Return from the Python list, not the torch tensor
    # We also round to 4 decimal places to strip any residual noise
    closest_target = round(float(sorted_targets[closest_index]), 4)
    
    # Debug Print
    print(f"[DEBUG] Global Density (Actual Keep): {global_density:.4f}")
    print(f"[DEBUG] Closest Target Width:       {closest_target}")
    
    return closest_target

def check_mask_keys(active_layers,
                mask1, 
                mask2={}
            ):
    relevant_keys = {
        'self_attn.q_proj': 4096, 
        'self_attn.k_proj': 4096, 
        'self_attn.v_proj': 4096, 
        'mlp.up_proj': 11008, 
        'mlp.gate_proj': 11008
        }

    matched_layers = []
    key_layer_match = {}

    all_layer_keys, missing_layer_keys, keys_to_remove = set(mask1.keys()).union(mask2.keys()), set(), set()

    for layer_key in all_layer_keys:
        match = re.search(
            r'layers\.(\d+)\.(self_attn.q_proj|self_attn.k_proj|self_attn.v_proj|mlp.up_proj|mlp.gate_proj)$', 
            layer_key)
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
            
            if active_layers[layer_id] == 1 and layer_type in relevant_keys:
                matched_layers.append(layer_id)

                key_layer_match.setdefault(layer_id, set()).add(layer_type) 
            elif active_layers[layer_id] != 1:
                keys_to_remove.add(layer_key)

    active_layer_indices = np.where(active_layers == 1)[0]

    for layer_id in active_layer_indices:
        matched_types_for_layer = key_layer_match.get(layer_id, set())
        for required_type in relevant_keys.keys(): 
            if required_type not in matched_types_for_layer:
                layer_key_for_missing = f"base_model.model.model.layers.{layer_id}.{required_type}"
                missing_layer_keys.add(layer_key_for_missing)

    del matched_layers
    del key_layer_match
    del all_layer_keys

    return missing_layer_keys, keys_to_remove

def crossover_layer(parent1_active_layers, parent2_active_layers, 
                    min_layer, model_num_hidden_layers):
    """
    Performs crossover between two parent active layer objects.

    Args:
        parent1_active_layers (np.array): Active layer configuration of parent 1.
        parent2_active_layers (np.array): Active layer configuration of parent 2.
        
    """
    # 1. Calculate the layer bounds for crossover
    MIN_LAYER_IDX = model_num_hidden_layers // 2 - 3
    MAX_LAYER_IDX = model_num_hidden_layers
    
    # Ensure indices are valid
    if MIN_LAYER_IDX < 0:
        MIN_LAYER_IDX = 0
    
    # Pre-process: handle multi-dimensional input (as in original function)
    if len(parent1_active_layers.shape) > 1:
        parent1_active_layers = parent1_active_layers[0]
        
    if len(parent2_active_layers.shape) > 1:
        parent2_active_layers = parent2_active_layers[0]

    # 2. Define the segments for crossover
    p1_segment = parent1_active_layers[MIN_LAYER_IDX : MAX_LAYER_IDX]
    p2_segment = parent2_active_layers[MIN_LAYER_IDX : MAX_LAYER_IDX]
    segment_length = len(p1_segment)

    # 3. Choose the crossover point within the segment
    # The crossover point is an index relative to the start of the segment (0 to segment_length)
    if segment_length > 1:
        # Choose a split point between 1 and segment_length - 1 (inclusive)
        crossover_point_segment = random.randint(1, segment_length - 1)
    else:
        # If segment is length 1 or less, no meaningful crossover can occur
        crossover_point_segment = segment_length
        
    # 4. Perform one-point crossover on the segment
    if segment_length > 0:
        offspring_segment = np.concatenate((
            p1_segment[:crossover_point_segment],
            p2_segment[crossover_point_segment:]
        ))
    else:
        offspring_segment = np.array([]) # Empty segment

    # 5. Concatenate untouched layers with the crossed-over segment
    untouched_prefix = parent1_active_layers[:MIN_LAYER_IDX]
    untouched_suffix = parent2_active_layers[MAX_LAYER_IDX:] # This should be an empty array if MAX_LAYER_IDX == len(parent2_active_layers)
    
    offspring_active_layers = np.concatenate((
        untouched_prefix,
        offspring_segment,
        untouched_suffix
    ))
    
    # Sanity check: ensure the resulting array has the correct length
    if len(offspring_active_layers) != model_num_hidden_layers:
         # Adjust the length if the MAX_LAYER_IDX assumption was off
         offspring_active_layers = offspring_active_layers[:model_num_hidden_layers]

    # --- 6. Constraint Checks (as in original function) ---
    
    # Ensure at least min_layer active layers
    num_active = np.sum(offspring_active_layers)
    if num_active < min_layer:
        random_num_layers = random.randint(min_layer, model_num_hidden_layers)
        inactive_indices = np.where(offspring_active_layers == 0)[0]
        num_to_activate = int(random_num_layers - num_active)
        if len(inactive_indices) >= num_to_activate:
            to_activate = np.random.choice(inactive_indices, num_to_activate, replace=False)
            offspring_active_layers[to_activate] = 1
        elif len(inactive_indices) > 0:
            offspring_active_layers[inactive_indices] = 1
    
    # Ensure at most model_num_hidden_layers active layers
    num_active = np.sum(offspring_active_layers)
    if num_active > model_num_hidden_layers:
        active_indices = np.where(offspring_active_layers == 1)[0]
        num_to_deactivate = int(num_active - model_num_hidden_layers)
        if len(active_indices) >= num_to_deactivate:
            to_deactivate = np.random.choice(active_indices, num_to_deactivate, replace=False)
            offspring_active_layers[to_deactivate] = 0
        elif len(active_indices) > 0:
            offspring_active_layers[active_indices] = 0

    return offspring_active_layers

def grouped_crossover(args, offspring_active_layers, parent1_width_mask, parent2_width_mask, choices, layers_var_info):
    """
    Performs crossover on layer masks, applying the logic to groups of layers
    (e.g., all attention layers in a block) to ensure consistent masking.
    
    Args:
        parent1_width_mask (dict): Dictionary of masks from parent 1.
        parent2_width_mask (dict): Dictionary of masks from parent 2.
        choices (list): List of tuples (layer_id, layer_type) to apply crossover to.
        args: Command-line arguments object.

    Returns:
        tuple: A tuple containing the offspring masks (width_mask)
    """
    # # Merge the variance dictionaries for easy lookup
    # layers_var_info = {**attn_layer_var, **mlp_layer_var}
    offspring_width_mask = {}

    all_layer_keys = set(parent1_width_mask.keys()).union(parent2_width_mask.keys())
    
    # Group layers from both parents
    grouped_layers = {}
    for layer_key in all_layer_keys:
        match = re.search(r'layers\.(\d+)\.(self_attn|mlp)', layer_key)
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
            if offspring_active_layers[layer_id]==1:
                if (layer_id, layer_type) not in grouped_layers:
                    grouped_layers[(layer_id, layer_type)] = []
                grouped_layers[(layer_id, layer_type)].append(layer_key)

    # Perform crossover on a group level
    for group_key, layer_keys in grouped_layers.items():
        layer_id, layer_type = group_key
        
        # Check if this group is in the chosen list for crossover
        if (layer_id, layer_type) in [ (c[0], c[1].split('.')[0]) for c in choices]:
            # For this group, check if masks exist in both parents
            parent1_has_masks = all(key in parent1_width_mask for key in layer_keys)
            parent2_has_masks = all(key in parent2_width_mask for key in layer_keys)

            if parent1_has_masks and parent2_has_masks:
                # Crossover logic for the entire group
                first_key = layer_keys[0]
                mask1 = parent1_width_mask[first_key]
                mask2 = parent2_width_mask[first_key]
                
                if mask1 is not None and mask2 is not None:
                    if mask1.shape == mask2.shape:
                        if first_key in layers_var_info:
                            sorted_indices = layers_var_info[first_key]
                            crossover_point_index = random.randint(0, len(sorted_indices))

                            # Create boolean masks for the crossover
                            crossover_mask_p1 = torch.zeros_like(mask1, dtype=torch.bool)
                            crossover_mask_p2 = torch.zeros_like(mask2, dtype=torch.bool)

                            mask1 = mask1.to(crossover_mask_p1)
                            mask2 = mask2.to(crossover_mask_p2)
                            
                            # Set the indices to be taken from parent1 to True
                            crossover_mask_p1[sorted_indices[:crossover_point_index]] = True
                            
                            # Set the remaining indices to be taken from parent2 to True
                            crossover_mask_p2[sorted_indices[crossover_point_index:]] = True

                            # Create the offspring mask by combining parts from both parents
                            offspring_mask = torch.zeros_like(mask1, dtype=torch.bool)
                            offspring_mask[crossover_mask_p1] = mask1[crossover_mask_p1]
                            offspring_mask[crossover_mask_p2] = mask2[crossover_mask_p2]

                        else:
                            # print(f"Warning: No variance data for {first_key}. Skiping.")
                            continue

                    else:
                        # In case of dimension mismatch, select one parent's masks for the whole group
                        chosen_parent_masks = random.choice([parent1_width_mask, parent2_width_mask])
                        for key in layer_keys:
                            offspring_width_mask[key] = copy.deepcopy(chosen_parent_masks[key])
                else:
                    continue

            elif parent1_has_masks:
                # If only parent 1 has masks for this group, inherit all of them
                for key in layer_keys:
                    offspring_width_mask[key] = copy.deepcopy(parent1_width_mask[key])

            elif parent2_has_masks:
                # If only parent 2 has masks for this group, inherit all of them
                for key in layer_keys:
                    offspring_width_mask[key] = copy.deepcopy(parent2_width_mask[key])
        else:
            # If the group is not in choices, randomly inherit from one parent
            parent_to_inherit = random.choice([parent1_width_mask, parent2_width_mask])
            for key in layer_keys:
                if key in parent_to_inherit:
                    offspring_width_mask[key] = copy.deepcopy(parent_to_inherit[key])

    return offspring_width_mask

def crossover(args, model, eval_dataloader, 
                parent1_active_layers, parent1_width_mask,
                parent2_active_layers, parent2_width_mask,
                width_options, min_layer, model_num_hidden_layers, granularity, choices, width
            ):
    """
    Performs crossover between two parent individuals.

    Args:
        parent1_active_layers (np.array): Active layer configuration of parent 1.
        parent1_width_mask (dict): Width masks of parent 1.
        parent2_active_layers (np.array): Active layer configuration of parent 2.
        parent2_width_mask (dict): Width masks of parent 2.
        width_options (list): List of possible width choices.
        model_num_hidden_layers (int): Total number of hidden layers in the model.
        granularity (list): specifies whether you want to modify a block, a sublock or the entire parent.

    Returns:
        tuple: (offspring_active_layers, offspring_width_mask)
    """
    
    relevant_keys = {
        'self_attn.q_proj': 4096, 
        'self_attn.k_proj': 4096, 
        'self_attn.v_proj': 4096, 
        'mlp.up_proj': 11008, 
        'mlp.gate_proj': 11008
        }
    
    if granularity=='#layers':
        offspring_active_layers = crossover_layer(parent1_active_layers, parent2_active_layers, 
                                                    min_layer, model_num_hidden_layers
                                                )
        offspring_width_mask = None
        

    # elif granularity in ['block', 'subblock']:
    #     offspring_active_layers = random.choice([parent1_active_layers, parent2_active_layers])

    #     offspring_width_mask = grouped_crossover(args, offspring_active_layers, parent1_width_mask, parent2_width_mask, choices, layers_var_info)

    #     missing_layer_keys, keys_to_remove = check_mask_keys(offspring_active_layers,
    #                                                         offspring_width_mask,
    #                                                     )

    return offspring_active_layers, offspring_width_mask

def mutate_layer(active_layers, min_layer, model_num_hidden_layers, mutation_rate_active_layers):
    # 1. Define the mutation boundaries
    MIN_LAYER_IDX = model_num_hidden_layers // 2 - 3
    MAX_LAYER_IDX = model_num_hidden_layers
    
    if len(active_layers.shape) > 1:
        active_layers = active_layers[0]
    
    # Ensure MIN_LAYER_IDX is not negative
    if MIN_LAYER_IDX < 0:
        MIN_LAYER_IDX = 0
        
    mutated_active_layers = copy.deepcopy(active_layers)

    # 2. Perform mutation only within the constrained segment (Untouched)
    for i in range(MIN_LAYER_IDX, min(MAX_LAYER_IDX, len(mutated_active_layers))):
        if random.random() < mutation_rate_active_layers:
            mutated_active_layers[i] = 1 - mutated_active_layers[i]

    # Define the indices that are allowed to be modified (the segment)
    segment_indices = np.arange(MIN_LAYER_IDX, min(MAX_LAYER_IDX, len(mutated_active_layers)))

    # 3. Ensure at least min_layer active layers
    num_active = mutated_active_layers.sum()
    if num_active < min_layer:
        random_num_layers = random.randint(min_layer, model_num_hidden_layers)
        num_to_activate = int(random_num_layers - num_active)

        # Find inactive indices ONLY within the segment
        if isinstance(mutated_active_layers, np.ndarray):
            # 1. Find all inactive indices in the whole array
            all_inactive = np.where(mutated_active_layers == 0)[0]
            # 2. Intersect with segment_indices to get inactive indices within the segment
            inactive_indices_in_segment = np.intersect1d(all_inactive, segment_indices)
            np_choice = np.random.choice
        else:
            # Handle PyTorch/other tensor types (Assuming torch is imported)
            all_inactive = torch.where(mutated_active_layers == 0)[0].cpu().numpy()
            inactive_indices_in_segment = np.intersect1d(all_inactive, segment_indices)
            np_choice = np.random.choice

        # Use the segment-restricted inactive indices
        if len(inactive_indices_in_segment) >= num_to_activate:
            to_activate = np_choice(inactive_indices_in_segment, num_to_activate, replace=False)
            mutated_active_layers[to_activate] = 1
        elif len(inactive_indices_in_segment) > 0:
            # If not enough, activate all inactive layers in the segment
            mutated_active_layers[inactive_indices_in_segment] = 1

    # ---

    # 4. Ensure at most model_num_hidden_layers active layers
    num_active = mutated_active_layers.sum()
    if num_active > model_num_hidden_layers:
        num_to_deactivate = int(num_active - model_num_hidden_layers)

        # Find active indices ONLY within the segment
        if isinstance(mutated_active_layers, np.ndarray):
            # 1. Find all active indices in the whole array
            all_active = np.where(mutated_active_layers == 1)[0]
            # 2. Intersect with segment_indices to get active indices within the segment
            active_indices_in_segment = np.intersect1d(all_active, segment_indices)
            np_choice = np.random.choice
        else:
            # Handle PyTorch/other tensor types
            all_active = torch.where(mutated_active_layers == 1)[0].cpu().numpy()
            active_indices_in_segment = np.intersect1d(all_active, segment_indices)
            np_choice = np.random.choice

        # Use the segment-restricted active indices
        if len(active_indices_in_segment) >= num_to_deactivate:
            to_deactivate = np_choice(active_indices_in_segment, num_to_deactivate, replace=False)
            mutated_active_layers[to_deactivate] = 0
        elif len(active_indices_in_segment) > 0:
            # If not enough, deactivate all active layers in the segment
            mutated_active_layers[active_indices_in_segment] = 0
    
    return mutated_active_layers

def grouped_mutation(mutated_active_layers, offspring_width_mask, choices, mutation_rate_width_mask_entry, layers_var_info):
    """
    Mutates masks by applying the same mutation logic to groups of layers 
    (e.g., all attention layers in a block).

    Args:
        offspring_width_mask (dict): A dictionary mapping layer names to their
                                    binary torch masks.
        choices (list): A list of tuples (layer_id, layer_type) to mutate.
        mutation_rate_width_mask_entry (float): The probability of a single bit being flipped.

    Returns:
        dict: The mutated offspring_width_mask.
    """
    # # Merge the variance dictionaries for easy lookup
    # layers_var_info = {**attn_layer_var, **mlp_layer_var}
    mutated_width_mask = copy.deepcopy(offspring_width_mask)
    
    # Group layers by their ID and type
    grouped_layers = {}
    for layer_key in mutated_width_mask.keys():
        match = re.search(r'layers\.(\d+)\.(self_attn|mlp)', layer_key)
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
            if mutated_active_layers[layer_id]==1:
                if (layer_id, layer_type) not in grouped_layers:
                    grouped_layers[(layer_id, layer_type)] = []
                grouped_layers[(layer_id, layer_type)].append(layer_key)

    # Iterate through the choices and apply mutation to the grouped layers
    for layer_id, layer_type in choices:
        group_key = (layer_id, 'self_attn' if 'attn' in layer_type else 'mlp')
        
        if group_key in grouped_layers:
            # Get the first mask in the group to get its shape and size
            first_layer_key = grouped_layers[group_key][0]
            mask_to_mutate = mutated_width_mask[first_layer_key]
            if mask_to_mutate is not None:
                if isinstance(mask_to_mutate, np.ndarray):
                    mask_to_mutate = torch.Tensor(mask_to_mutate).bool()
                
                # Create a single, random mutation mask for the entire group
                
                num_elements = mask_to_mutate.numel()
                # mutation_mask_1d = torch.rand(num_elements) < mutation_rate_width_mask_entry
                num_to_flip = int(num_elements * mutation_rate_width_mask_entry)

                # Check if variance data is available for this layer
                if first_layer_key in layers_var_info:
                    # Get the sorted indices for this layer (lowest variance first)
                    sorted_indices = layers_var_info[first_layer_key]
                    
                    # Select the `num_to_flip` indices with the lowest variance
                    indices_to_flip = sorted_indices[-num_to_flip:]  # Changed to lowest variance
                    
                    # Create a mutation mask with `True` at these specific indices
                    mutation_mask_1d = torch.zeros(num_elements, dtype=torch.bool)
                    mutation_mask_1d[indices_to_flip] = True
                else:
                    # Fallback to random mutation if variance data is not available
                    # print(f"Warning: No variance data for {first_layer_key}. Skipping.")
                    # mutation_mask_1d = torch.rand(num_elements) < mutation_rate_width_mask_entry
                    continue

                current_mask = mask_to_mutate
                flat_mask = current_mask.flatten().bool()
                
                # Use a boolean XOR (^) for efficient bit flipping
                mutated_mask = (flat_mask ^ mutation_mask_1d).reshape(current_mask.shape)
                
                # Apply the same mutation logic to all layers in the group
                for layer_key in grouped_layers[group_key]:
                    mutated_width_mask[layer_key] = mutated_mask

    return mutated_width_mask

def mutate(args, model, eval_dataloader, 
            offspring_active_layers, offspring_width_mask,
            mutation_rate_active_layers, mutation_rate_width_mask_entry,
            min_layer, model_num_hidden_layers, granularity, choices, width, mutation_scale=0.01
        ):
    """
    Performs mutation on an offspring individual.

    Args:
        offspring_active_layers (np.array): Active layer configuration to mutate.
        offspring_width_mask (dict): Width masks to mutate.
        mutation_rate_active_layers (float): Probability to flip an active layer bit.
        mutation_rate_width_mask_entry (float): Probability to flip a single width mask bit.
        min_layer (int): Minimum number of active layers allowed.
        model_num_hidden_layers (int): Total number of hidden layers in the model.

    Returns:
        tuple: (mutated_active_layers, mutated_width_mask)
    """
    relevant_keys = {
        'self_attn.q_proj': 4096, 
        'self_attn.k_proj': 4096, 
        'self_attn.v_proj': 4096, 
        'mlp.up_proj': 11008, 
        'mlp.gate_proj': 11008
        }

    if granularity=='#layers':
        # print(offspring_active_layers)
        mutated_active_layers = mutate_layer(offspring_active_layers, min_layer, model_num_hidden_layers, mutation_rate_active_layers)
        mutated_width_mask = None


    # elif granularity in ['block', 'subblock']:
    #     mutated_active_layers = offspring_active_layers

    #     mutated_width_mask = grouped_mutation(mutated_active_layers, offspring_width_mask, choices, mutation_rate_width_mask_entry, layers_var_info)

    #     missing_layer_keys, keys_to_remove = check_mask_keys(mutated_active_layers,
    #                                                         mutated_width_mask,
    #                                                     )
            
    return mutated_active_layers, mutated_width_mask

def is_duplicate(
    new_individual_active_layers_attn: np.ndarray,
    new_individual_active_layers_mlp: np.ndarray,
    new_individual_width_masks: Dict[str, Union[np.ndarray, torch.Tensor]],
    population_active_layers: List[np.ndarray],
    population_width_masks: List[Dict],
):
    """
    Checks for duplicates based on Active Layers (Attn & MLP).
    Ignores Width Masks as requested.
    """

    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            # Fix for bfloat16 which numpy doesn't support
            if arr.is_floating_point() and arr.dtype == torch.bfloat16:
                arr = arr.float() 
            return arr.detach().cpu().numpy()
        return arr

    # Pre-convert the new individual to numpy once to save time
    target_attn = to_numpy(new_individual_active_layers_attn)
    target_mlp = to_numpy(new_individual_active_layers_mlp)

    for i in range(len(population_active_layers)):
        
        # 1. Check Active Layers
        # population_active_layers[i] is expected to be (attn, mlp)
        pop_attn = to_numpy(population_active_layers[i][0])
        pop_mlp = to_numpy(population_active_layers[i][1])

        if not np.array_equal(target_attn, pop_attn):
            continue
            
        if not np.array_equal(target_mlp, pop_mlp):
            continue

        # If we reach here, Attn and MLP are all identical.
        return True #, i
                
    return False #, -1

def set_width_mask(args, model, width_mask, active_layer_attn_idx, active_layer_mlp_idx):
    ''' Set width masks during evolutionary search with robust key matching '''
    
    # Determine dtype once
    mask_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    for name, module in model.named_modules():
        # 1. Regex to find target layers (q, k, v, up, gate)
        # This implicitly excludes o_proj and down_proj, so we don't need extra if-statements.
        match = re.search(
            r'layers\.(\d+)\.(self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|mlp\.up_proj|mlp\.gate_proj)$',
            name
        )
        
        if match:
            # Extract ID and Type from the Regex
            layer_id = int(match.group(1))
            suffix = match.group(2) # e.g. "self_attn.q_proj"
            
            # 2. Check Active Status
            is_active = False
            if 'self_attn' in suffix and layer_id in active_layer_attn_idx:
                is_active = True
            elif 'mlp' in suffix and layer_id in active_layer_mlp_idx:
                is_active = True
            
            if not is_active:
                continue

            # 3. Robust Key Lookup
            # We generate multiple variations of the name to find which one is in the dictionary.
            
            short_name = match.group(0) # This is "layers.{id}.{suffix}"
            
            candidates = [
                name,                                   # Exact match from named_modules
                short_name,                             # "layers.0.self_attn.q_proj"
                f"model.{short_name}",                  # "model.layers.0..."
                f"base_model.model.{short_name}"        # "base_model.model.layers.0..."
            ]
            
            found_key = None
            for key in candidates:
                if key in width_mask:
                    found_key = key
                    break
            
            # 4. Apply Mask if Key Found
            if found_key and width_mask[found_key] is not None:
                mask_val = width_mask[found_key]
                
                # Convert Numpy -> Tensor if needed
                if isinstance(mask_val, np.ndarray):
                    mask_val = torch.from_numpy(mask_val)
                
                # Move to correct device/dtype
                # We use module.weight.device to ensure it matches the layer
                mask_val = mask_val #.to(device=module.weight.device, dtype=mask_dtype)
                
                # (Optional) Update the dict so we don't convert again next time
                width_mask[found_key] = mask_val

                # Apply
                if hasattr(module, 'set_width_mask'):
                    # print(name, 'setting_mask')
                    module.set_width_mask(width_mask=mask_val)

def reset_width_mask(args, model, width_mask, bias=None):
    ''' Set width masks during evolutionary search'''
    for name, module in model.named_modules():
        if hasattr(module, 'set_width_mask'):
            module.set_width_mask(width_mask=None, output_bias=None)

def get_repeated_values_lwr(popu_structure_list, popu_infl_score_list, limit=200):
    """
    Returns the top unique structures up to a limit.
    """
    N = len(popu_structure_list)
    
    # 1. Combine data into a list of dictionaries for easy sorting
    combined = []
    for i in range(N):
        combined.append({
            'structure': popu_structure_list[i],
            'score': popu_infl_score_list[i],
            'index': i,
            # Create the unique key tuple (Length, Width, Rank)
            'key': (popu_structure_list[i][0], popu_structure_list[i][1], popu_structure_list[i][2])
        })

    # 2. Sort by score descending (Highest score first)
    # This ensures that when we encounter a structure, it is automatically the best version of it.
    combined.sort(key=lambda x: x['score'], reverse=True)

    best_individuals_by_structure = {}

    # 3. Iterate and Fill
    for item in combined:
        # If we have reached the limit, stop immediately
        if len(best_individuals_by_structure) >= limit:
            break
            
        key = item['key']
        
        # Only add if we haven't seen this structure yet
        if key not in best_individuals_by_structure:
            best_individuals_by_structure[key] = {
                'structure': item['structure'],
                'score': item['score'],
                'index': item['index']
            }
            
    return best_individuals_by_structure

def get_pareto_optimal_structures(popu_structure_list, popu_infl_score_list, popu_loss_list,
                                 popu_latency_list, popu_throughput_list, limit=200):
    """
    Returns the Pareto optimal unique structures based on Score (max), 
    Latency (min), and Throughput (max).
    """
    N = len(popu_structure_list)
    
    # 1. First, group by unique structure to get the best representative for each key
    # (Just in case the same structure appeared multiple times in the population)
    unique_candidates = {}
    for i in range(N):
        key = (popu_structure_list[i][0], popu_structure_list[i][1], popu_structure_list[i][2])
        score = popu_infl_score_list[i]
        loss = popu_loss_list[i]
        latency = popu_latency_list[i]
        throughput = popu_throughput_list[i]
        
        # We only keep the best version of a specific structure if it repeats
        if key not in unique_candidates or score > unique_candidates[key]['score']:
            unique_candidates[key] = {
                'structure': popu_structure_list[i],
                'score': score,
                'loss': loss,
                'latency': latency,
                'throughput': throughput,
                'index': i,
                'key': key
            }

    candidates = list(unique_candidates.values())
    pareto_results = {}

    # 2. Pareto Efficiency Check
    # A model is Pareto optimal if no other model dominates it.
    for i, model_a in enumerate(candidates):
        is_dominated = False
        for j, model_b in enumerate(candidates):
            if i == j:
                continue
            
            # Check if model_b dominates model_a
            # Condition: B is at least as good as A in all, and better in one.
            b_better_or_equal = (
                model_b['score'] >= model_a['score'] and
                model_b['loss'] <= model_a['loss'] and
                model_b['latency'] <= model_a['latency'] and
                model_b['throughput'] >= model_a['throughput']
            )
            
            b_strictly_better = (
                model_b['score'] > model_a['score'] or
                model_b['loss'] < model_a['loss'] or
                model_b['latency'] < model_a['latency'] or
                model_b['throughput'] > model_a['throughput']
            )
            
            if b_better_or_equal and b_strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            key = model_a['key']
            pareto_results[key] = {
                'structure': model_a['structure'],
                'score': model_a['score'],
                'loss': model_a['loss'],
                'latency': model_a['latency'],      # Included in return for visibility
                'throughput': model_a['throughput'], # Included in return for visibility
                'index': model_a['index']
            }
            
            # Respect the limit provided in arguments
            if len(pareto_results) >= limit:
                break

    return pareto_results

def get_diversity_preserving_pareto(popu_structure_list, popu_infl_score_list, 
                                    popu_latency_list, popu_throughput_list, limit=200):
    """
    Finds Pareto optimal structures within each unique depth-level 
    to prevent shallow models from dominating deep-thin ones.
    """
    N = len(popu_structure_list)
    
    # 1. Group by Depth (Layer Count)
    # This ensures a 32-layer model isn't discarded just because a 16-layer one is faster.
    depth_groups = {}
    for i in range(N):
        # Index 0 is attn_layers, Index 1 is mlp_layers. We use attn_layers as the 'depth' key.
        depth = popu_structure_list[i][0] 
        if depth not in depth_groups:
            depth_groups[depth] = []
        
        depth_groups[depth].append({
            'structure': popu_structure_list[i],
            'score': popu_infl_score_list[i],
            'latency': popu_latency_list[i],
            'throughput': popu_throughput_list[i],
            'index': i,
            'key': (popu_structure_list[i][0], popu_structure_list[i][1], popu_structure_list[i][2])
        })

    final_pareto_results = {}

    # 2. Run Pareto Check INSIDE each depth group
    for depth, candidates in depth_groups.items():
        for i, model_a in enumerate(candidates):
            is_dominated = False
            for j, model_b in enumerate(candidates):
                if i == j: continue
                
                # B dominates A within the same depth if it's better in score & speed
                # Note: Since they have the same depth, this is now a fair width-to-width fight.
                if (model_b['score'] >= model_a['score'] and 
                    model_b['latency'] <= model_a['latency'] and 
                    model_b['throughput'] >= model_a['throughput']):
                    
                    if (model_b['score'] > model_a['score'] or 
                        model_b['latency'] < model_a['latency'] or 
                        model_b['throughput'] > model_a['throughput']):
                        is_dominated = True
                        break
            
            if not is_dominated:
                final_pareto_results[model_a['key']] = model_a

    # 3. Final Diversity Check: If we have room, add the best-scoring non-pareto unique models
    if len(final_pareto_results) < limit:
        all_unique = {}
        for i in range(N):
            key = (popu_structure_list[i][0], popu_structure_list[i][1], popu_structure_list[i][2])
            if key not in final_pareto_results:
                if key not in all_unique or popu_infl_score_list[i] > all_unique[key]['score']:
                    all_unique[key] = {
                        'structure': popu_structure_list[i],
                        'score': popu_infl_score_list[i],
                        'latency': popu_latency_list[i],
                        'throughput': popu_throughput_list[i],
                        'index': i
                    }
        
        # Sort leftovers by score and fill
        remaining_slots = limit - len(final_pareto_results)
        sorted_leftovers = sorted(all_unique.values(), key=lambda x: x['score'], reverse=True)
        
        for extra in sorted_leftovers[:remaining_slots]:
            key = (extra['structure'][0], extra['structure'][1], extra['structure'][2])
            final_pareto_results[key] = extra

    return final_pareto_results

def get_best_structures_by_score_and_frequency(
    popu_structure_list, 
    popu_infl_score_list, 
    population_width_masks, 
    min_mask_frequency=3 # Note: This parameter is currently unused in your logic, but kept for context.
):
    
    # Key: (w) | Value: {'max_score': float, 'count': int, 'best_mask': tuple}
    mask_metrics = defaultdict(lambda: {'max_score': -float('inf'), 'count': 0, 'best_mask': None})
    
    # 1. Aggregate Scores, Counts, and Find the Highest Scorer's Mask for each 'w'
    for i in range(len(popu_structure_list)):
        structure = popu_structure_list[i]
        w = structure[2]
        current_mask_dict = population_width_masks[i]
        current_score = popu_infl_score_list[i]
        
        # Convert dictionary to hashable tuple
        mask_tuple = tuple(sorted(current_mask_dict.items()))
        
        # Update max score, count, and track the actual best mask
        mask_metrics[w]['count'] += 1
        
        if current_score > mask_metrics[w]['max_score']:
             mask_metrics[w]['max_score'] = current_score
             mask_metrics[w]['best_mask'] = mask_tuple
    
    # Key: w | Value: The actual best-scoring mask tuple found for this 'w'
    best_mask_for_w = {w: metrics['best_mask'] for w, metrics in mask_metrics.items()}
    
    best_individuals_by_structure = {} # Key: (l_attn, l_mlp, w) | Value: {'structure': tuple, 'score': float, 'mask': tuple}
    
    # 2. Select the Best Individual for each unique structure (l, l, w)
    for i in range(len(popu_structure_list)):
        structure = popu_structure_list[i]
        l_attn, l_mlp, w = structure # Unpack for clarity
        grouping_key = structure
        current_mask_dict = population_width_masks[i]
        current_score = popu_infl_score_list[i]
        current_mask_tuple = tuple(sorted(current_mask_dict.items()))
        
        # Determine if the current individual's mask is 'canonical' for its 'w'
        # The canonical check is simplified: check if the actual mask tuple matches the best mask tuple
        # is_canonical = (
        #     w in best_mask_for_w and \
        #     current_mask_tuple == best_mask_for_w[w]
        # )

        is_canonical = (
            w in best_mask_for_w and \
            np.all(
                np.array([val.sum().item() for val in dict(current_mask_tuple).values()]) \
                == \
                np.array([val.sum().item() for val in dict(best_mask_for_w[w]).values()]))
            ) 
        

        # Since the population is sorted by score (descending), the first one we find 
        # for a unique (l, l, w) with the canonical mask is the highest scoring one.
        # Note: The original logic had a potentially complex and error-prone `np.all` check on mask *sums*. 
        # I've simplified it to check if the current mask is *exactly* the best mask.
        if is_canonical and grouping_key not in best_individuals_by_structure:
            best_individuals_by_structure[grouping_key] = {
                'structure': structure,
                'score': current_score,
                'index': i,
                # Store the canonical mask itself, which is the current mask
                'mask': dict(current_mask_tuple) 
            }
            
    # 3. Compile the final lists for return
    selected_structures = []
    new_population_width_masks = []

    for data in best_individuals_by_structure.values():
        # Get the selected structure
        selected_structures.append(data['structure'])
        
        # Use the *canonical* mask for this 'w' (which we stored in 'mask')
        # This mask is shared among all selected structures with the same 'w' 
        # IF your canonical check allows multiple different masks to be canonical.
        # Since the simplified canonical check above requires an *exact* match to the single best mask, 
        # this is already the canonical, shared mask for the chosen (l,l,w) structure.
        new_population_width_masks.append(data['mask'])
    
    # Return the selected structures and the corresponding canonical/shared masks
    return best_individuals_by_structure #, new_population_width_masks


def get_best_structures_by_score_and_frequency_v2(
    popu_structure_list, 
    popu_infl_score_list, 
    population_width_masks, 
    min_mask_frequency=3 
):
    
    N = len(popu_structure_list)

    # 1. FIND THE CANONICAL MASK FOR EACH WIDTH 'w' 
    # Key: w | Value: {'max_score': float, 'best_mask': tuple}
    best_mask_for_w_metrics = defaultdict(lambda: {'max_score': -float('inf'), 'best_mask': None})
    
    for i in range(N):
        w = popu_structure_list[i][2]
        current_mask_dict = population_width_masks[i]
        current_score = popu_infl_score_list[i]
        
        # Convert dictionary to hashable tuple
        mask_tuple = tuple(sorted(current_mask_dict.items()))
        
        if current_score > best_mask_for_w_metrics[w]['max_score']:
             best_mask_for_w_metrics[w]['max_score'] = current_score
             best_mask_for_w_metrics[w]['best_mask'] = mask_tuple
    
    # Final map: w -> The highest scoring mask found for that w (Canonical Mask)
    best_mask_for_w = {w: metrics['best_mask'] for w, metrics in best_mask_for_w_metrics.items()}
    
    
    # 2. SELECT THE ABSOLUTE HIGHEST-SCORING INDIVIDUAL FOR EACH UNIQUE STRUCTURE
    # Key: (l_attn, l_mlp, w) | Value: {'structure': tuple, 'score': float, 'index': int, 'mask': tuple}
    elite_individuals_by_structure = {} 
    
    # Assuming popu_structure_list is sorted by score (descending)
    for i in range(N):
        structure = popu_structure_list[i]
        grouping_key = structure
        current_score = popu_infl_score_list[i]
        
        # The first time we see a unique structure, it is the highest scoring one.
        if grouping_key not in elite_individuals_by_structure:
            elite_individuals_by_structure[grouping_key] = {
                'structure': structure,
                'score': current_score,
                'index': i,
            }
            
    # 3. ASSIGN CANONICAL MASK TO ELITE STRUCTURES (The mask with the highest score for its 'w')
    for grouping_key, data in elite_individuals_by_structure.items():
        w = data['structure'][2]
        canonical_mask = best_mask_for_w.get(w)
        
        # Assign the best mask found for that structure's width 'w'
        data['mask'] = dict(canonical_mask)
    
    # Return only the dictionary of elite structures
    return elite_individuals_by_structure

def get_best_structures_multi_objective(
    popu_structure_list, 
    popu_infl_score_list, 
    population_width_masks, 
    population_loss_list,
    alpha=0.5  # Weighting: 0.5 = Balanced. Higher alpha favors Fisher. Lower favors Loss.
):
    
    N = len(popu_structure_list)
    
    # --- STEP 0: NORMALIZE AND COMPUTE COMPOSITE SCORES ---
    
    # 1. Normalize Fisher Score (Higher is Better)
    min_fisher = min(popu_infl_score_list)
    max_fisher = max(popu_infl_score_list)
    denom_fisher = max_fisher - min_fisher if max_fisher != min_fisher else 1.0
    
    # 2. Normalize Loss (Lower is Better)
    min_loss = min(population_loss_list)
    max_loss = max(population_loss_list)
    denom_loss = max_loss - min_loss if max_loss != min_loss else 1.0
    
    composite_scores = []
    
    for i in range(N):
        # Scale Fisher to [0, 1]
        norm_fisher = (popu_infl_score_list[i] - min_fisher) / denom_fisher
        
        # Scale Loss to [0, 1] and INVERT (so 1.0 is lowest loss/best)
        norm_loss = 1.0 - ((population_loss_list[i] - min_loss) / denom_loss)
        
        # Weighted Combination
        # alpha * Structure + (1-alpha) * Texture
        comp_score = (alpha * norm_fisher) + ((1 - alpha) * norm_loss)
        composite_scores.append(comp_score)

    # --- STEP 1: FIND THE CANONICAL MASK FOR EACH WIDTH 'w' ---
    # Key: w | Value: {'max_score': float, 'best_mask': tuple}
    best_mask_for_w_metrics = defaultdict(lambda: {'max_score': -float('inf'), 'best_mask': None})
    
    for i in range(N):
        w = popu_structure_list[i][2]
        current_mask_dict = population_width_masks[i]
        
        # WE USE THE COMPOSITE SCORE NOW
        # We want the mask that is both Structurally Sound AND Low Perplexity
        current_score = composite_scores[i]
        
        # Convert dictionary to hashable tuple
        mask_tuple = tuple(sorted(current_mask_dict.items()))
        
        if current_score > best_mask_for_w_metrics[w]['max_score']:
             best_mask_for_w_metrics[w]['max_score'] = current_score
             best_mask_for_w_metrics[w]['best_mask'] = mask_tuple
    
    # Final map: w -> The highest scoring mask found for that w (Canonical Mask)
    best_mask_for_w = {w: metrics['best_mask'] for w, metrics in best_mask_for_w_metrics.items()}
    
    
    # --- STEP 2: SELECT THE ABSOLUTE HIGHEST-SCORING INDIVIDUAL FOR EACH UNIQUE STRUCTURE ---
    # Key: (l_attn, l_mlp, w) | Value: {'structure': tuple, 'score': float, 'index': int, 'mask': tuple}
    elite_individuals_by_structure = {} 
    
    for i in range(N):
        structure = popu_structure_list[i]
        grouping_key = structure
        
        # Use Composite Score for selection
        current_comp_score = composite_scores[i]
        
        # Logic: If we haven't seen this structure, OR if this specific instance 
        # has a better composite score than the previous one we saw, take it.
        if grouping_key not in elite_individuals_by_structure or \
           current_comp_score > elite_individuals_by_structure[grouping_key]['composite_score']:
            
            elite_individuals_by_structure[grouping_key] = {
                'structure': structure,
                'score': popu_infl_score_list[i],     # Keep original raw score for logging
                'loss': population_loss_list[i],      # Keep raw loss for logging
                'composite_score': current_comp_score, # The sorting metric
                'index': i,
            }
            
    # --- STEP 3: ASSIGN CANONICAL MASK TO ELITE STRUCTURES ---
    for grouping_key, data in elite_individuals_by_structure.items():
        w = data['structure'][2]
        canonical_mask = best_mask_for_w.get(w)
        
        # Assign the best mask found for that structure's width 'w'
        if canonical_mask:
            data['mask'] = dict(canonical_mask)
        else:
            # Fallback (should theoretically not happen if logic holds)
            idx = data['index']
            data['mask'] = population_width_masks[idx]
    
    # Return only the dictionary of elite structures
    return elite_individuals_by_structure

def get_repeated_values_by_params(popu_params_list, popu_infl_score_list):
    """
    Groups individuals by their total number of parameters and returns the one with
    the highest influence score for each unique parameter count.

    Args:
        popu_params_list (list): A list where each element is the number of
                                parameters for an individual in the population.
        popu_infl_score_list (list): A list of influence scores corresponding to
                                    each individual.

    Returns:
        dict: A dictionary where keys are the number of parameters and values are
            the best individual's score and index for that parameter count.
    """
    best_individuals_by_params = {}

    for i, num_params in enumerate(popu_params_list):
        current_influence_score = popu_infl_score_list[i]

        # Check if we have seen this number of parameters before
        if num_params not in best_individuals_by_params:
            # If not, store the current individual as the "best so far"
            best_individuals_by_params[num_params] = {
                'score': current_influence_score,
                'index': i
            }
        else:
            # If we have, compare the current individual's score to the best we have
            if current_influence_score > best_individuals_by_params[num_params]['score']:
                # If the current one is better, update the stored best individual
                best_individuals_by_params[num_params] = {
                    'score': current_influence_score,
                    'index': i
                }
    
    return best_individuals_by_params

def get_repeated_values_lwr_pareto(popu_structure_list, popu_infl_score_list, population_loss_list):
    """
    Groups individuals by structure and identifies the Pareto front for each group
    for high influence score and low loss.

    Args:
        popu_structure_list (list): A list of (length, width, rank) tuples.
        popu_infl_score_list (list): A list of influence scores.
        population_loss_list (list): A list of loss values.

    Returns:
        dict: A dictionary where keys are (length, width) tuples and values are
            lists of indices of the Pareto optimal individuals for that structure.
    """
    # 1. Group all individuals by their (length, width) structure
    individuals_by_structure = {}
    for i, structure in enumerate(popu_structure_list):
        length_width = (structure[0], structure[1])
        if length_width not in individuals_by_structure:
            individuals_by_structure[length_width] = []
        individuals_by_structure[length_width].append(i)

    # 2. For each group, find the Pareto front
    pareto_individuals_by_structure = {}
    for length_width, indices in individuals_by_structure.items():
        # Get the objectives (score and loss) for all individuals in this group
        objectives = [
            (popu_infl_score_list[i], population_loss_list[i]) for i in indices
        ]

        # Find the non-dominated indices within this group
        pareto_front_indices = []
        for i, obj1 in enumerate(objectives):
            is_dominated = False
            for j, obj2 in enumerate(objectives):
                # Don't compare an individual to itself
                if i == j:
                    continue

                # Check if obj2 dominates obj1
                # obj2 dominates obj1 if it's better on all objectives
                # High score is better, low loss is better.
                if (obj2[0] >= obj1[0] and obj2[1] <= obj1[1]) and \
                (obj2[0] > obj1[0] or obj2[1] < obj1[1]):
                    is_dominated = True
                    break
            
            # If the individual is not dominated by any other in the group, it's on the Pareto front
            if not is_dominated:
                pareto_front_indices.append(indices[i])
        
        pareto_individuals_by_structure[length_width] = pareto_front_indices
        
    return pareto_individuals_by_structure

def get_random_derived_offspring(args, parent1, parent1_width_mask, 
                    parent2, parent2_width_mask,
                ):
                    
    random_choice = random.randint(0, len([parent1, parent2])-1)

    if random_choice==0:
        offspring_active_layers = copy.deepcopy(parent1)
        offspring_width_mask = copy.deepcopy(parent1_width_mask)
    else:
        offspring_active_layers = copy.deepcopy(parent2)
        offspring_width_mask = copy.deepcopy(parent2_width_mask)

    return offspring_active_layers, offspring_width_mask

def generate_parents(args, population_active_layers, 
                    population_width_masks,
                        chosen_combinations, strategy, num_hidden_layers, eval_dataloader, layers_var_info=None
                        ):
    random_choice = random.randint(0,1)

    # if random_choice==0:
    # Selection (Tournament Selection example)
    parent1_idx = random.sample(range(len(population_active_layers)), 2)
    random_idx_choice = random.choice(parent1_idx)

    parent1 = copy.deepcopy(population_active_layers[random_idx_choice])
    parent1_width_mask = copy.deepcopy(population_width_masks[random_idx_choice])

    parent2_idx = random.sample(range(len(population_active_layers)), 2)
    random_idx_choice = random.choice(parent2_idx)

    parent2 = copy.deepcopy(population_active_layers[random_idx_choice])
    parent2_width_mask = copy.deepcopy(population_width_masks[random_idx_choice])

    return parent1, parent1_width_mask, parent2, parent2_width_mask

def get_granularity(args, num_hidden_layers, num_combinations):
    """
    Gets the granularity to which you want to modify the curent parent/parents (cross, mutate, combination). 
    #layers: modify the number of layers.
    block: modify a specific block for specific layers.
    sub-block: modify a specific sub-block for specific layers

    Returns:
        Returns a combination of the three options for the evolution.
    """
    block_types = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj']  

    # if args.layer_pruning or args.use_flap_for_width_search or args.mod_layers_only:
    random_choice = 0
    # elif args.min_num_layer == num_hidden_layers:
    #     random_choice = random.randint(1,5)
    # else:
    #     random_choice = random.randint(0,5)
    if random_choice == 0:
        granularity = '#layers'
        choices = zip()
    elif random_choice == 1:
        granularity = 'subblock'
        layer_choices = [random.randint(0, num_hidden_layers-1) for _ in range(num_combinations)]
        block_choices = [random.choice(block_types) for _ in range(num_combinations)]
        choices = zip(layer_choices, block_choices)
    elif random_choice==2:
        granularity = 'block'
        layer_choices = [random.randint(0, num_hidden_layers-1) for _ in range(num_combinations)]
        choices = [(layer, block_type) for layer in layer_choices for block_type in block_types]
    elif random_choice == 3:
        granularity = 'subblock_full'
        layer_choices = [random.randint(0, num_hidden_layers-1) for _ in range(num_combinations)]
        block_choices = [random.choice(block_types) for _ in range(num_combinations)]
        choices = zip(layer_choices, block_choices)
    elif random_choice==4:
        granularity = 'block_full'
        layer_choices = [random.randint(0, num_hidden_layers)-1 for _ in range(num_combinations)]
        choices = [(layer, block_type) for layer in layer_choices for block_type in block_types]
    elif random_choice==5:
        granularity = 'full_active'
        layer_choices = [layer for layer in range(num_hidden_layers)]
        choices = [(layer, block_type) for layer in layer_choices for block_type in block_types] 
    
    return granularity, choices

def generate_offsprings(args, width_options, model, eval_dataloader,
                        parent1, parent1_width_mask, 
                        parent2, parent2_width_mask,
                        granularity, choices, layers_var_info=None, crossover_rate=0.7, mutation_rate_width_mask_entry=0.2, mutation_rate_active_layers=0.2):

    assert parent1 is not None
    assert parent2 is not None
    if not args.layer_pruning:
        target_width = random.choice(width_options)
    else:
        target_width = width_options[0]

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_layers = model_config.num_hidden_layers

    relevant_keys = {
        'self_attn.q_proj': 4096, 
        'self_attn.k_proj': 4096, 
        'self_attn.v_proj': 4096, 
        'mlp.up_proj': 11008, 
        'mlp.gate_proj': 11008
    }
    
    if not args.layer_pruning:
        random_choice = random.randint(0,2)
    else:
        random_choice = random.randint(0,2)
    if random_choice==0:
        if random.random() < crossover_rate:
            offspring_active_layers, offspring_width_mask = \
                crossover(args, model, eval_dataloader,
                        parent1, parent1_width_mask,
                        parent2, parent2_width_mask,
                        width_options, args.min_num_layer, num_layers,
                        granularity, choices, target_width
                    )
        else:
            offspring_active_layers, offspring_width_mask = get_random_derived_offspring(args, parent1, parent1_width_mask, 
                                                            parent2, parent2_width_mask, 
                                                            )

        # Mutation
        offspring_active_layers, offspring_width_mask = \
            mutate(args, model, eval_dataloader, 
                    offspring_active_layers, offspring_width_mask, 
                    mutation_rate_active_layers, mutation_rate_width_mask_entry, 
                    args.min_num_layer, num_layers, 
                    granularity, choices, target_width)
    
    elif random_choice==1:
        if random.random() < crossover_rate:
            offspring_active_layers, offspring_width_mask = \
                crossover(args, model, eval_dataloader,
                        parent1, parent1_width_mask,
                        parent2, parent2_width_mask,
                        width_options, args.min_num_layer, num_layers, granularity, choices, target_width)
        else:
            offspring_active_layers, offspring_width_mask = get_random_derived_offspring(args, parent1, parent1_width_mask, 
                                                            parent2, parent2_width_mask, 
                                                            )

    elif random_choice==2:
        offspring_active_layers, offspring_width_mask = get_random_derived_offspring(args, parent1, parent1_width_mask, 
                                                            parent2, parent2_width_mask, 
                                                            )

        # Mutation
        offspring_active_layers, offspring_width_mask = mutate(args, model, eval_dataloader, 
                offspring_active_layers, offspring_width_mask, 
                mutation_rate_active_layers, mutation_rate_width_mask_entry, 
                args.min_num_layer, num_layers, 
                granularity, choices, target_width)

    offspring_width_mask = None
    return offspring_active_layers, offspring_width_mask

def count_active_parameters(
    model: torch.nn.Module, 
    active_attn_layers: np.ndarray, 
    active_mlp_layers: np.ndarray, 
    width_masks: Dict[str, torch.Tensor] = None, 
    round_to: int = 1e6
) -> Tuple[int, int]:
    """
    Counts parameters for a Hard Pruned model (Inactive Layers = Ghost Modules).
    Ignores bias injection parameters.
    """
    
    # 1. Setup Model Access
    try:
        if hasattr(model, "base_model"):
            base_model = model.base_model.model.model
        elif hasattr(model, "model"):
            base_model = model.model
        else:
            base_model = model
            
        layers = base_model.layers
        config = base_model.config
    except AttributeError:
        layers = model.model.layers
        config = model.model.config
    
    is_pythia = 'pythia' in config._name_or_path.lower()

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    # num_kv_heads = config.num_key_value_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = num_heads
    head_dim = hidden_size // num_heads
    is_gqa = num_kv_heads < num_heads

    # Identify active layer indices
    # Assumes input is binary array/list [1, 0, 1...]
    active_layer_indices = set(np.where(active_attn_layers == 1)[0]) | set(np.where(active_mlp_layers == 1)[0])

    active_base_params = 0
    total_lora_params = 0

    # if is_pythia:
    #     # Pythia specific naming
    #     embed_tokens = base_model.embed_in.weight.numel()
    #     norm = base_model.final_layer_norm.weight.numel()
        
    #     lm_head = 0
    #     # In Pythia, the LM head is often model.embed_out
    #     if hasattr(model, "embed_out"):
    #         lm_head = model.embed_out.weight.numel()
    # else:
    #     # 2. Count Global Parameters (Embeddings, Final Norm, LM Head)
    #     # These are usually never pruned
    #     embed_tokens = base_model.embed_tokens.weight.numel()
    #     norm = base_model.norm.weight.numel()
        
    #     lm_head = 0
    #     if hasattr(model, "lm_head"):
    #         lm_head = model.lm_head.weight.numel()
    embed_tokens = (base_model.embed_in if is_pythia else base_model.embed_tokens).weight.numel()
    norm = (base_model.final_layer_norm if is_pythia else base_model.norm).weight.numel()
    tie_embeddings = getattr(config, "tie_word_embeddings", False)
    # Only add lm_head if it's NOT tied to the input embeddings
    lm_head = 0
    if not tie_embeddings:
        if is_pythia and hasattr(model, "embed_out"):
            lm_head = model.embed_out.weight.numel()
        elif hasattr(model, "lm_head"):
            lm_head = model.lm_head.weight.numel()
    
    active_base_params += (embed_tokens + norm + lm_head)

    # 3. Iterate Layers
    for layer_idx, layer_module in enumerate(layers):
        
        # --- DEPTH PRUNING LOGIC ---
        # Even in inactive "Ghost" layers, LayerNorms remain physically present
        norm_params = layer_module.input_layernorm.weight.numel() + \
                      layer_module.post_attention_layernorm.weight.numel()
        active_base_params += norm_params

        # If layer is inactive (Ghost), Linear layers are 0 params. Skip.
        if layer_idx not in active_layer_indices:
            continue

        # --- WIDTH PRUNING LOGIC (Active Layers Only) ---
        
        # Helper to extract active dimension sum from mask dict
        def get_active_width(name_key, full_dim):
            if width_masks and name_key in width_masks:
                mask = width_masks[name_key]
                
                # Sanity check: ensure it's a tensor/array before summing
                if isinstance(mask, (torch.Tensor, np.ndarray)):
                    return int(mask.sum())
                # If it's a list, convert and sum
                elif isinstance(mask, list):
                    return int(sum(mask))
                    
            return full_dim

        if is_pythia:
            qkv_key = f"layers.{layer_idx}.attention.query_key_value"
            o_key =  f"layers.{layer_idx}.attention.query_key_value"
            active_base_params += get_active_width(qkv_key, full_dim=num_heads*head_dim*3)
            active_base_params += get_active_width(o_key, full_dim=num_heads*head_dim)*3
        else:
            # 3.1 ATTENTION PARAMS
            # Q_Proj: Output Pruned
            # q_key = f'base_model.model.model.layers.{layer_idx}.self_attn.q_proj'
            q_key = f'layers.{layer_idx}.self_attn.q_proj'
            # active_q_heads = get_active_width(q_key, full_dim=num_heads * head_dim)
            # active_base_params += (active_q_heads * hidden_size)
            active_base_params += get_active_width(q_key, full_dim=num_heads*head_dim*hidden_size)

            # K/V Proj
            if is_gqa:
                # GQA models (Llama 3) do not prune K/V -> Count Full Size
                kv_size = (num_kv_heads * head_dim) * hidden_size
                active_base_params += (kv_size * 2) 
            else:
                # Standard models (Llama 2) -> Prune Output
                # k_key = f'base_model.model.model.layers.{layer_idx}.self_attn.k_proj'
                # v_key = f'base_model.model.model.layers.{layer_idx}.self_attn.v_proj'
                k_key = f'layers.{layer_idx}.self_attn.k_proj'
                v_key = f'layers.{layer_idx}.self_attn.v_proj'
                # active_k = get_active_width(k_key, num_heads * head_dim)
                # active_v = get_active_width(v_key, num_heads * head_dim)
                # active_base_params += (active_k * hidden_size)
                # active_base_params += (active_v * hidden_size)
                active_base_params += get_active_width(k_key, full_dim=num_heads*head_dim*hidden_size)
                active_base_params += get_active_width(v_key, full_dim=num_heads*head_dim*hidden_size)

            # O_Proj: Input Pruned (Shares mask with V or Q)
            sibling_type = 'self_attn.q_proj' if is_gqa else 'self_attn.v_proj'
            # o_sibling_key = f'base_model.model.model.layers.{layer_idx}.{sibling_type}'
            o_sibling_key = f'layers.{layer_idx}.{sibling_type}'
            
            # active_o_input = get_active_width(o_sibling_key, num_heads * head_dim)
            # active_base_params += (hidden_size * active_o_input)
            active_base_params += get_active_width(q_key, full_dim=num_heads*head_dim*hidden_size) 

        if is_pythia:
            up_key = f"layers.{layer_idx}.mlp.dense_h_to_4h"
            down_key = f"layers.{layer_idx}.mlp.dense_4h_to_h"
            active_base_params += get_active_width(up_key, full_dim=config.intermediate_size)
            active_base_params += get_active_width(down_key, full_dim=config.intermediate_size)
        else:
            # 3.2 MLP PARAMS
            # Up & Gate: Output Pruned
            # up_key = f'base_model.model.model.layers.{layer_idx}.mlp.up_proj'
            # gate_key = f'base_model.model.model.layers.{layer_idx}.mlp.gate_proj'
            up_key = f'layers.{layer_idx}.mlp.up_proj'
            gate_key = f'layers.{layer_idx}.mlp.gate_proj'
            
            # active_up = get_active_width(up_key, config.intermediate_size)
            # active_gate = get_active_width(gate_key, config.intermediate_size)
            
            # active_base_params += (active_up * hidden_size)
            # active_base_params += (active_gate * hidden_size)
            active_base_params += get_active_width(up_key, full_dim=config.intermediate_size*hidden_size)
            active_base_params += get_active_width(gate_key, full_dim=config.intermediate_size*hidden_size)

            # Down: Input Pruned (Shares mask with Up)
            # down_sibling_key = f'base_model.model.model.layers.{layer_idx}.mlp.up_proj'
            down_sibling_key = f'layers.{layer_idx}.mlp.up_proj'
            # active_down_in = get_active_width(down_sibling_key, config.intermediate_size)
            # active_base_params += (hidden_size * active_down_in)
            active_base_params += get_active_width(gate_key, full_dim=config.intermediate_size*hidden_size)


    # 4. LoRA Params
    # Assumption: If the layer is active, the LoRA adapter is attached and active.
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this param belongs to an inactive layer
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_id = int(match.group(1))
                if layer_id in active_layer_indices:
                    total_lora_params += param.numel()
            else:
                # Non-layer params (like head LoRA)
                total_lora_params += param.numel()

    # 5. Rounding
    if round_to is not None and round_to > 0:
        total_lora_params = int(round(total_lora_params / round_to) * round_to)
        active_base_params = int(round(active_base_params / round_to) * round_to)

    return int(total_lora_params), int(active_base_params)