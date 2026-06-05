from tqdm import tqdm
from peft.utils.integrations import dequantize_bnb_weight
import torch 
import numpy as np
from torch import nn
from typing import * #TYPE_CHECKING, Any, Callable, Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from transformers.tracenas_utils.evol_search_utils import *
import torch.nn.functional as F
from torch.nn.functional import kl_div
import types
from transformers.models.llama.modeling_llama import LlamaMLP,LlamaAttention
import transformers
import math
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from transformers.models.llama.modeling_llama import NoAttention, NoMLP

standarlization = lambda x: (x - torch.mean(x, axis=0, keepdim=True)) / torch.std(x, axis=0, keepdim=True)

cos = nn.CosineSimilarity(dim=0, eps=1e-6)

def dequantize(layer):
    if hasattr(layer, 'quant_state'):
        if hasattr(layer, 'get_base_layer'):
            weight = layer.get_base_layer().weight
        else:
            weight = layer.weight
        dequant_weight = dequantize_bnb_weight(weight, state=weight.quant_state)
        
        return dequant_weight
    else:
        return layer.weight

def get_layer_sizes(model: Any, active_layers_attn_indices: List[int], active_layers_mlp_indices: List[int]) -> Dict[int, int]:
    """
    Calculates the total number of prunable neurons (dimensions) for each layer type.
    We are interested in the size of the *output* dimensions of the projection layers,
    which correspond to the number of scores in the importance lists.
    """
    if hasattr(model.config, 'hidden_size_list') and hasattr(model.config, 'intermediate_size_list'):
        # attn_neuron_count =
        attn_neuron_count = defaultdict(int)
        mlp_neuron_count = defaultdict(int)

        for i in range(len(model.config.hidden_size_list)):
            if model.config.hidden_size_list[i] > 0:
                attn_neuron_count[i] = model.config.hidden_size_list[i]
                
            if model.config.intermediate_size_list[i] > 0:
                mlp_neuron_count[i] = model.config.intermediate_size_list[i]
            
    else:
        attn_neuron_count = defaultdict(int)
        mlp_neuron_count = defaultdict(int)

        for module_name, module in model.named_modules():
           # Updated Attention Output Projection Regex
            # Matches: 
            # Llama/Gemma: layers.0.self_attn.o_proj
            # Pythia:      layers.0.attention.dense
            match_attn = re.search(r'layers\.(\d+)\.(self_attn\.o_proj|attention\.dense)$', module_name)

            # Updated MLP Down-Projection Regex
            # Matches:
            # Llama/Gemma: layers.0.mlp.down_proj
            # Pythia:      layers.0.mlp.dense_4h_to_h
            match_mlp = re.search(r'layers\.(\d+)\.mlp\.(down_proj|dense_4h_to_h)$', module_name)

            if match_attn:
                layer_id = int(match_attn.group(1))
                if layer_id in active_layers_attn_indices:
                    # Add the output size (which is the dimension we are pruning)
                    attn_neuron_count[layer_id] += module.out_features
            elif match_mlp:
                layer_id = int(match_mlp.group(1))
                if layer_id in active_layers_mlp_indices:
                    # Add the output size
                    mlp_neuron_count[layer_id] += module.in_features

    return attn_neuron_count, mlp_neuron_count 

def get_valid_head_count(
    proposed_pruned_params: float, 
    total_layer_params: int, 
    num_heads: int, 
    num_kv_heads: int, 
    is_gqa: bool
) -> int:
    """
    Calculates the maximum number of active heads allowed that satisfies
    GQA constraints and is <= the proposed pruning target.
    """
    # 1. Calculate params per head
    params_per_head = total_layer_params / num_heads
    
    # 2. Calculate ideal active parameters
    target_active_params = total_layer_params - proposed_pruned_params
    
    # 3. Convert to raw active heads (floored)
    raw_active_heads = int(target_active_params // params_per_head)
    
    # 4. Apply GQA Constraint
    # If GQA, active Q heads must be a multiple of active KV heads. 
    # Since we assume KV isn't pruned here (based on your code), we assume full KV heads.
    if is_gqa:
        # Group size is how many Q heads share one KV head
        group_size = num_heads // num_kv_heads 
        # Snap down to nearest valid group
        # This ensures active_Q % active_K == 0
        
        # However, your original code logic implies:
        # We want active_Q to be a multiple of num_kv_heads
        valid_active_heads = (raw_active_heads // num_kv_heads) * num_kv_heads
        
        # Safety: Ensure at least 1 group remains if not fully pruning
        if valid_active_heads == 0 and raw_active_heads > 0:
             valid_active_heads = num_kv_heads
    else:
        # Standard MHA: any integer number of heads is valid (usually)
        valid_active_heads = raw_active_heads

    return valid_active_heads, params_per_head

def get_valid_neuron_count(
    proposed_pruned_params: float,
    total_layer_params: int,
    param_block_size: int
) -> Tuple[int, int]:
    """
    Calculates the number of active 'blocks' (e.g. groups of 256 neurons) to keep,
    ensuring the layer size remains hardware-friendly.

    Args:
        proposed_pruned_params (float): How many parameters the algorithm WANTS to remove.
        total_layer_params (int): The total parameters in the unpruned layer.
        param_block_size (int): The number of parameters corresponding to one unit of alignment.
                                (e.g., if alignment is 256 neurons, this is 256 * hidden_size).

    Returns:
        valid_blocks (int): The integer number of blocks to keep (e.g., 12 chunks of 256).
        param_block_size (int): Passed back for symmetry with the head_count function.
    """
    
    # 1. Calculate the ideal amount of parameters we want to KEEP
    target_active_params = total_layer_params - proposed_pruned_params
    
    # 2. Calculate how many full blocks fit into that target
    # We use integer division (//) to floor it to the nearest valid block
    raw_active_blocks = int(target_active_params // param_block_size)
    
    # 3. Safety Floor: Always keep at least 1 block
    # This prevents the layer from being fully deleted (lobotomized), which would break the network flow
    valid_blocks = max(1, raw_active_blocks)

    return valid_blocks, param_block_size

def _generate_random_candidate(
    config, layer_scores, layer_sizes, global_pruning_target, 
    k_priority, MAX_LAYER_PRUNING, EPS
):
    """
    Original initialization logic: Prioritizes low-saliency layers randomly.
    """
    attn_k, mlp_k = k_priority
    total_params = sum(layer_sizes['attn'].values()) + sum(layer_sizes['mlp'].values())
    total_prune_target = total_params * (1 - global_pruning_target)
    remaining_budget = total_prune_target
    
    pruning_ratios = {'attn': {k: 0.0 for k in layer_sizes['attn']}, 'mlp': {k: 0.0 for k in layer_sizes['mlp']}}

    # Sort layers by score (Ascending: Lowest score = Prune first)
    sorted_attn = sorted([k for k in layer_scores if k in layer_sizes['attn']], key=lambda k: layer_scores[k])
    sorted_mlp = sorted([k for k in layer_scores if k in layer_sizes['mlp']], key=lambda k: layer_scores[k])

    # 1. Prune Priority Attention
    is_gqa = config.num_key_value_heads < config.num_attention_heads
    for name in sorted_attn[:attn_k]:
        if remaining_budget <= 0: break
        ratio = np.random.uniform(low=MAX_LAYER_PRUNING, high=1.0 - MAX_LAYER_PRUNING)
        current_size = layer_sizes['attn'][name]
        proposed = min(current_size * (1 - ratio), remaining_budget)
        
        valid_heads, p_per_head = get_valid_head_count(proposed, current_size, config.num_attention_heads, config.num_key_value_heads, is_gqa)
        actual_pruned = current_size - (valid_heads * p_per_head)
        
        pruning_ratios['attn'][name] = actual_pruned / current_size
        remaining_budget -= actual_pruned

    # 2. Prune Priority MLP
    for name in sorted_mlp[:mlp_k]:
        if remaining_budget <= 0: break
        ratio = np.random.uniform(low=MAX_LAYER_PRUNING/2, high=1.0 - (MAX_LAYER_PRUNING/2))
        current_size = layer_sizes['mlp'][name]
        proposed = min(current_size * (1 - ratio), remaining_budget)
        
        pruning_ratios['mlp'][name] = proposed / current_size
        remaining_budget -= proposed

    return pruning_ratios, remaining_budget

def _get_layer_index(layer_name: str) -> int:
    """Extracts layer index from string 'model.layers.15.self_attn...'"""
    # Finds the first number in the string
    match = re.search(r'\.(\d+)\.', layer_name)
    if match:
        return int(match.group(1))
    return 0 # Fallback

def _generate_evolutionary_candidate_perfect(
    parents: list,
    config, 
    layer_sizes, 
    global_pruning_target, # Assumed to be "Target Params to KEEP" (e.g. 5B or 0.5)
    crossover_prob=0.4,
    mutation_prob=0.4,
    mutation_strength=0.3,
    MAX_LAYER_PRUNING=0.75, 
    MIN_HEADS=6,
    MLP_ALIGNMENT=32
):
    
    # --- 1. SETUP & CONSTANTS ---
    # Constraints
    min_keep_attn = 1.0 - MAX_LAYER_PRUNING
    min_keep_mlp = 1.0 - MAX_LAYER_PRUNING
    
    # GQA / MQA Grid Setup
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    
    # Group Size (GQA alignment)
    group_size = num_q_heads // num_kv_heads
    total_groups = num_kv_heads 

    # [NEW] Calculate GQA Alignment Step
    # For Llama 3 (8 KV, 4 GroupSize): Step is 2.
    if num_q_heads > num_kv_heads:
        common_divisor = math.gcd(total_groups, group_size)
        attn_alignment_step = total_groups // common_divisor
    else:
        attn_alignment_step = 1
    
    # [NEW] Grid Step Ratio: (Alignment Step) / Num_KV_Heads
    # This ensures valid ratios like 0.25, 0.5 (multiples of the alignment block)
    attn_step_ratio = attn_alignment_step / max(1, total_groups) 
    
    # --- 2. BREEDING (Crossover & Mutation) ---
    # Select Parent 1 
    p1_idx = 0 if random.random() < 0.4 else random.randint(0, len(parents)-1)
    parent1 = parents[p1_idx]
    
    do_crossover = (len(parents) >= 2 and random.random() < crossover_prob)
    parent2 = parents[random.randint(0, len(parents)-1)] if do_crossover else None

    child_ratios = {'attn': {}, 'mlp': {}}
    
    # We store "steps" for the budget solver
    param_costs = [] 

    for module, min_keep in [('attn', min_keep_attn), ('mlp', min_keep_mlp)]:
        for layer_name, p1_sparsity in parent1[module].items():
            
            # A. CROSSOVER (Arithmetic Blend)
            if do_crossover and parent2:
                bias = random.uniform(0.2, 0.8) 
                p2_sparsity = parent2[module][layer_name]
                val_sparsity = (bias * p1_sparsity) + ((1.0 - bias) * p2_sparsity)
            else:
                val_sparsity = p1_sparsity

            # CONVERT TO KEEP RATIO 
            val_keep = 1.0 - val_sparsity
            
            # B. DEFINE GRID & BOUNDS
            total_params = layer_sizes[module][layer_name]

            if module == 'attn':
                current_step_ratio = attn_step_ratio # Use aligned step
                
                # Align Min Heads Constraint
                min_heads_ratio = MIN_HEADS / num_q_heads
                raw_min_keep = max(min_keep, min_heads_ratio)
                
                # [NEW] Snap Min Keep to nearest aligned step UP
                layer_min_keep = math.ceil(raw_min_keep / current_step_ratio) * current_step_ratio
                
                params_per_step = total_params * current_step_ratio
                
            else: # MLP
                current_step_ratio = MLP_ALIGNMENT / total_params
                layer_min_keep = min_keep
                params_per_step = MLP_ALIGNMENT

            # C. MUTATION (Operate on Keep Ratio)
            if random.random() < mutation_prob:
                scale = random.gauss(1.0, mutation_strength)
                val_keep *= scale
                
            # D. SNAP TO GRID
            # Snap the KEEP ratio to the nearest valid ALIGNED step
            snapped_ratio = round(val_keep / current_step_ratio) * current_step_ratio
            val_keep = max(layer_min_keep, min(snapped_ratio, 1.0))
            
            # E. RECORD FOR BUDGETER
            current_steps = int(round(val_keep / current_step_ratio))
            min_steps = int(np.ceil(layer_min_keep / current_step_ratio))
            max_steps = int(round(1.0 / current_step_ratio))
            
            param_costs.append({
                'type': module,
                'name': layer_name,
                'cost': params_per_step,
                'steps': current_steps,
                'min': min_steps,
                'max': max_steps,
                'step_ratio': current_step_ratio
            })

    # # --- 3. BUDGET ENFORCEMENT ---
    # # Sum of active params
    # current_total_params = sum(item['steps'] * item['cost'] for item in param_costs)
    # param_diff = current_total_params - global_pruning_target
    
    # random.shuffle(param_costs)
    
    # # Simple budget loop (add/remove steps until fit)
    # for _ in range(200):
    #     if abs(param_diff) < (global_pruning_target * 0.001): break
        
    #     for item in param_costs:
    #         if abs(param_diff) < (global_pruning_target * 0.001): break
            
    #         if param_diff > 0: # Prune (remove steps)
    #             if item['steps'] > item['min']:
    #                 item['steps'] -= 1
    #                 param_diff -= item['cost']
    #         else: # Restore (add steps)
    #             if item['steps'] < item['max']:
    #                 item['steps'] += 1
    #                 param_diff += item['cost']

    # --- 4. OUTPUT FORMATTING (Invert back to Sparsity) ---
    for item in param_costs:
        # 1. Get Final Keep Ratio
        keep_ratio = item['steps'] * item['step_ratio']
        keep_ratio = min(max(keep_ratio, 0.0), 1.0)
        
        # 2. Convert to Pruning Ratio (Sparsity)
        final_sparsity = 1.0 - keep_ratio
        
        child_ratios[item['type']][item['name']] = final_sparsity

    return child_ratios

def _generate_evolutionary_candidate(
    parents: list,
    config, 
    layer_sizes, 
    global_pruning_target, # Absolute param count (e.g., 2.7e9)
    crossover_prob=0.7,
    mutation_prob=0.2,
    mutation_strength=0.3,
    MAX_LAYER_PRUNING=0.72, 
    MIN_HEADS=8,
    MLP_ALIGNMENT=32
):
    """
    Breeds a new candidate by blending parent widths and respecting depth-pruning.
    Uses 'Revival Logic' to ensure that current active layers are not 
    accidentally masked just because they were inactive in a parent.
    """
    # --- 1. SETUP & CONSTANTS ---
    num_layers = config.num_hidden_layers
    min_keep_val = 1.0 - MAX_LAYER_PRUNING
    
    num_q_heads = config.num_attention_heads
    # Pythia/GPT-NeoX lacks num_key_value_heads; default to num_q_heads (MHA)
    num_kv_heads = getattr(config, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = num_q_heads

    total_groups = num_kv_heads 

    step_ratio = 1.0 / num_q_heads

    # Gemma 2 check (requires specific alignment for Sliding Window Attention/GQA)
    is_gemma = "gemma2" in getattr(config, "model_type", "").lower()

    # MLP block size: Gemma/Llama use 32; Pythia can use 64 for efficiency
    MLP_ALIGNMENT = 32 if not ("pythia" in str(config.architectures).lower()) else 64
    
    # GQA Alignment Calculation
    if num_q_heads > num_kv_heads:
        group_size = num_q_heads // num_kv_heads
        common_divisor = math.gcd(total_groups, group_size)
        attn_alignment_step = total_groups // common_divisor
    else:
        attn_alignment_step = 1
    
    attn_step_ratio = attn_alignment_step / max(1, total_groups) 

    # --- 2. PARENT SELECTION ---
    p1_idx = 0 if random.random() < 0.4 else random.randint(0, len(parents)-1)
    parent1 = parents[p1_idx]
    
    do_crossover = (len(parents) >= 2 and random.random() < crossover_prob)
    parent2 = parents[random.randint(0, len(parents)-1)] if do_crossover else None

    child_ratios = {'attn': {}, 'mlp': {}}
    param_costs = [] 

    # --- 3. BREEDING LOOP (Joint Width/Depth Handling) ---
    for module in ['attn', 'mlp']:
        for layer_idx in range(num_layers):
            layer_name = layer_idx 

            # A. HARD DEPTH CHECK: Is the layer active in the CURRENT iteration?
            # We use layer_sizes (the current candidate's config) as the ground truth.
            total_params = layer_sizes[module].get(layer_name, 0)
            
            if total_params == 0:
                # Current Depth Pruner says this layer is DEAD. Force 1.0 Sparsity.
                child_ratios[module][layer_name] = 1.0
                continue
            
            # B. PARENT DATA EXTRACTION & REVIVAL
            # If the parent had this layer dead (1.0), but it is now alive (>0),
            # we 'revive' it to 0.0 (full width) so the breeder can explore its width.
            p1_sparsity = parent1[module].get(layer_name, 1.0)
            if p1_sparsity >= 1.0 and total_params > 0:
                p1_sparsity = 0.0 
            
            if do_crossover and parent2:
                p2_sparsity = parent2[module].get(layer_name, 1.0)
                if p2_sparsity >= 1.0 and total_params > 0:
                    p2_sparsity = 0.0
                
                # Arithmetic Blend
                bias = random.uniform(0.2, 0.8) 
                val_sparsity = (bias * p1_sparsity) + ((1.0 - bias) * p2_sparsity)
            else:
                val_sparsity = p1_sparsity

            # C. WIDTH CONSTRAINTS & MUTATION
            val_keep = 1.0 - val_sparsity

            if module == 'attn':
                current_step_ratio = attn_step_ratio

                is_swa_layer = is_gemma and (layer_idx % 2 == 0)
                if is_swa_layer:
                    # We enforce a stricter floor for SWA layers (e.g., keep at least 50% heads)
                    # This ensures local context window stability during CPT
                    dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.5))
                else:
                    dynamic_min_heads = MIN_HEADS

                min_heads_ratio = dynamic_min_heads / num_q_heads
                layer_min_keep = math.ceil(max(min_keep_val, min_heads_ratio) / current_step_ratio) * current_step_ratio
                params_per_step = total_params * current_step_ratio
            else:
                current_step_ratio = MLP_ALIGNMENT / total_params
                layer_min_keep = min_keep_val
                params_per_step = MLP_ALIGNMENT

            # Mutation
            if random.random() < mutation_prob:
                random_factor = random.choice([0,1])
                if random_factor == 0:
                    # Scale Mutation
                    scale = random.gauss(1.0, mutation_strength)
                    val_keep *= scale
                else:
                    # # Additive Mutation
                    # scale = random.gauss(0, mutation_strength)
                    # val_keep += scale
                    # Additive: Nudge by a specific number of heads/blocks
                    # Strength of 1.5 means we jitter by ~1-2 discrete steps
                    nudge_steps = round(random.gauss(0, 1.5)) 
                    val_keep += (nudge_steps * step_ratio)
            
            # Bound the ratio before snapping to prevent invalid configurations
            val_keep = max(min_keep_val, min(val_keep, 1.0))

            # Snap to Hardware Grid (GEMM alignment)
            snapped_ratio = round(val_keep / current_step_ratio) * current_step_ratio
            val_keep = max(layer_min_keep, min(snapped_ratio, 1.0))
            
            # D. RECORD FOR BUDGET ENFORCEMENT
            current_steps = int(round(val_keep / current_step_ratio))
            param_costs.append({
                'type': module,
                'name': layer_name,
                'cost': params_per_step,
                'steps': current_steps,
                'min': int(np.ceil(layer_min_keep / current_step_ratio)),
                'max': int(round(1.0 / current_step_ratio)),
                'step_ratio': current_step_ratio
            })

    # --- 5. FINAL OUTPUT FORMATTING ---
    # A. Fill with Budgeter Results
    for item in param_costs:
        keep_ratio = item['steps'] * item['step_ratio']
        child_ratios[item['type']][item['name']] = 1.0 - float(keep_ratio)

    # B. Force any missing layers (Inactives) to 1.0 Sparsity
    for module in ['attn', 'mlp']:
        for i in range(num_layers):
            if i not in child_ratios[module]:
                child_ratios[module][i] = 1.0

    return child_ratios

def get_evolutionary_pruning_ratios(
    config,
    layer_scores: Dict[str, float], 
    layer_sizes: Dict[str, int],
    global_pruning_target: float,
    best_sparsity_values: List[Dict] = None, 
    k_priority: tuple = (5, 5),
    MAX_LAYER_PRUNING: float = 0.6
):
    if not best_sparsity_values:
        print(">>> [Evolution] History empty. Generating via Iterative Priority Initialization.")
        return _generate_random_candidate(
            config, layer_scores, layer_sizes, global_pruning_target, 
            k_priority, MAX_LAYER_PRUNING, EPS=1e-4
        )
    else:
        print(f">>> [Evolution] Evolving from {len(best_sparsity_values)} elites.")
        # NEW CALL with Crossover probabilities
        return _generate_evolutionary_candidate(
            best_sparsity_values,
            config,
            layer_sizes,
            global_pruning_target,
            crossover_prob=0.7, # 50% chance to mix parents
            mutation_prob=0.2   # 50% chance to mutate (can be additive)
        )

def get_pruning_masks(args, model, target_width, pruning_ratios, active_layers_attn_indices, active_layers_mlp_indices, attn_metrics_list, mlp_metrics_list):
    attn_final_masks, mlp_final_masks = {}, {}

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    for module_name, module in model.named_modules():
        if ('o_proj' in module_name or 'down_proj' in module_name) and 'base_layer' in module_name:

            match = re.search(
                r'layers\.(\d+)\.(self_attn.o_proj|mlp.down_proj).base_layer$',
                module_name
            )
            layer_id = int(match.group(1))

            if 'o_proj' in module_name:
                if layer_id in active_layers_attn_indices:
                    #Random sampling
                    threshold = pruning_ratios['attn'][layer_id] #* 3
                    num_total_dims = attn_metrics_list[layer_id]
                    num_to_keep = int(num_total_dims * (1-threshold)) 
                    num_to_keep = max(1, num_to_keep) # Ensure at least one dimension is kept
                    
                    # print(f"Keeping {num_to_keep} out of {num_total_dims} dimensions.")
                    keep_indices = random.sample(range(0,num_total_dims), num_to_keep)
                    
                    mask_1d = torch.zeros(num_total_dims, dtype=torch.bool)
                    mask_1d[keep_indices] = True
                    
                    attn_final_masks[layer_id] = mask_1d

            elif 'down_proj' in module_name:
                if layer_id in active_layers_mlp_indices:
                    #Random sampling
                    threshold = pruning_ratios['mlp'][layer_id] #* 2
                    num_total_dims = mlp_metrics_list[layer_id]
                    num_to_keep = int(num_total_dims * (1-threshold)) #int(num_total_dims * (1.0 - threshold)) #int(num_total_dims * threshold * 2) #
                    num_to_keep = max(1, num_to_keep) # Ensure at least one dimension is kept
                    
                    keep_indices = random.sample(range(0,num_total_dims), num_to_keep)
                    
                    mask_1d = torch.zeros(num_total_dims, dtype=torch.bool)
                    mask_1d[keep_indices] = True
                    
                    mlp_final_masks[layer_id] = mask_1d

    is_gqa_or_mqa = model.config.num_key_value_heads != model.config.num_attention_heads

    final_masks = {}
    for module_name, module in model.named_modules():
        match = re.search(
            r'layers\.(\d+)\.(self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|mlp\.up_proj|mlp\.gate_proj)$',
            module_name
        )
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)

            if 'self_attn' in layer_type:
                if layer_id in active_layers_attn_indices:
                    # if 'Qwen' not in model.model.__class__.__name__:
                    if not is_gqa_or_mqa:
                        final_masks[module_name] = attn_final_masks[layer_id] #[:module.out_features]
                    else:
                        if 'k_proj' in module_name or 'v_proj' in module_name:              
                            # print('here!')          
                            # threshold = pruning_ratios['attn'][layer_id] #* 3
                            num_total_dims = module.out_features
                            # max_pruning = num_total_dims * (1 - (2*0.12))
                            # threshold = threshold/8 #min(max_pruning, threshold)
                            
                            # num_to_keep = int(num_total_dims * (1-threshold)) 
                            # num_to_keep = max(1, num_to_keep) # Ensure at least one dimension is kept
                            if target_width==1:
                                num_to_keep = int(num_total_dims)
                            else:
                                min_keep_percent = 1 #0.85 
                                num_to_keep = int(num_total_dims * min_keep_percent)
                                num_to_keep = max(1, num_to_keep)
                            
                            # print(f"Keeping {num_to_keep} out of {num_total_dims} dimensions.")
                            keep_indices = random.sample(range(0,num_total_dims), num_to_keep)
                            
                            mask_1d = torch.zeros(num_total_dims, dtype=torch.bool)
                            mask_1d[keep_indices] = True

                            final_masks[module_name] = mask_1d
                            # print(num_to_keep)
                            # exit()
                        else:
                            final_masks[module_name] = attn_final_masks[layer_id]
            elif 'mlp' in layer_type:
                if layer_id in active_layers_mlp_indices:
                    final_masks[module_name] = mlp_final_masks[layer_id]
            else:
                continue
    # print(np.array([val.sum().item()/val.shape[0] for val in final_masks.values()]))
    # exit()
    return final_masks, attn_final_masks, mlp_final_masks

def get_attn_mlp_masks(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx):
    config = model.module.config if hasattr(model, "module") else model.config
    attn_masks, mlp_masks = {}, {}
    for module_name, module_mask in fisher_mask.items():
        if isinstance(module_mask, np.ndarray):
            module_mask = torch.from_numpy(module_mask)
            
        match = re.search(
            r'layers\.(\d+)\.(self_attn\.q_proj|mlp\.up_proj)$',
            module_name
        )
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
                            
            if 'self_attn' in layer_type:
                if layer_id in active_layer_attn_idx:
                    attn_masks[layer_id] = copy.deepcopy(module_mask)
                # final_mask[module_name] = copy.deepcopy(module_mask)
            
            elif 'mlp' in layer_type:
                if layer_id in active_layer_mlp_idx:
                    mlp_masks[layer_id] = copy.deepcopy(module_mask)
                # final_mask[module_name] = copy.deepcopy(module_mask)
    # # print(attn_masks.keys(), mlp_masks.keys())
    # is_gqa_or_mqa = config.num_key_value_heads != config.num_attention_heads
    # final_masks = {}
    # for module_name, module in model.named_modules():
    #     match = re.search(
    #         r'layers\.(\d+)\.(self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|mlp\.up_proj|mlp\.gate_proj)$',
    #         module_name
    #     )
    #     if match:
    #         layer_id = int(match.group(1))
    #         layer_type = match.group(2)

    #         if 'self_attn' in layer_type:
    #             if layer_id in active_layer_attn_idx:
    #                 # if 'Qwen' not in model.model.__class__.__name__:
    #                 if not is_gqa_or_mqa:
    #                     final_masks[module_name] = attn_masks[layer_id]
    #                 else:
    #                     # if attn_masks[layer_id].shape!=module.out_features:
    #                     #     final_masks[module_name] = attn_masks[layer_id][:module.out_features]
    #                     if 'k_proj' in module_name or 'v_proj' in module_name:
    #                         final_masks[module_name] = copy.deepcopy(fisher_mask[module_name])
    #                     else:
    #                         final_masks[module_name] = attn_masks[layer_id]
                            
    #         elif 'mlp' in layer_type:
    #             if layer_id in active_layer_mlp_idx:
    #                 final_masks[module_name] = mlp_masks[layer_id]
    #         else:
    #             continue

    return fisher_mask, attn_masks, mlp_masks

def get_attn_mlp_sparsity(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx):
    config = model.module.config if hasattr(model, "module") else model.config
    num_layers = config.num_hidden_layers
    
    # Initialize with 0.0 so the dict always has the correct indices
    attn_sparsity = {i: 0.0 for i in range(num_layers)}
    mlp_sparsity = {i: 0.0 for i in range(num_layers)}
    
    for module_name, module_mask in fisher_mask.items():
        if isinstance(module_mask, np.ndarray):
            module_mask = torch.from_numpy(module_mask)
            
        match = re.search(
            r'layers\.(\d+)\.(self_attn\.q_proj|attention\.query_key_value|mlp\.up_proj|mlp\.gate_proj|mlp\.dense_h_to_4h)$',
            module_name
        )
        
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
            
            # Check Attention
            if 'self_attn' in layer_type or 'attention' in layer_type:
                if layer_id in active_layer_attn_idx:
                    attn_sparsity[layer_id] = module_mask.sum().item() / module_mask.numel()
                else:
                    attn_sparsity[layer_id] = 0.0 # Force zero for inactive
            
            # Check MLP
            elif 'mlp' in layer_type:
                if layer_id in active_layer_mlp_idx:
                    mlp_sparsity[layer_id] = module_mask.sum().item() / module_mask.numel()
                else:
                    mlp_sparsity[layer_id] = 0.0 # Force zero for inactive

    return {'attn': attn_sparsity, 'mlp': mlp_sparsity}

def get_layer_importance(model, input_tensors_attn, input_tensors_mlp):
    """
    Calculates global importance of each layer using a WANDA-like proxy.
    Score = Sum( |W| * ||Input|| )
    """
    layer_scores = {'attn': {}, 'mlp': {}}

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_layers = model_config.num_hidden_layers

    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
    
    elif hasattr(model.model.model, 'language_model'):
        lm_model = model.model.model.language_model
    
        if hasattr(lm_model, 'layers'):
            model_layers = lm_model.layers

    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP)
    }
    
    print(">>> Calculating WANDA Layer Importance (Corrected)...")
    
    for i in range(num_layers):
        # --- Process ATTENTION ---
        if i in skip_attn_indices:
            # Do parameter extraction and accumulation here...
            continue

        # --- Process MLP ---
        if i in skip_mlp_indices:
            # Do parameter extraction and accumulation here...
            continue

        # layer = model.model.model.layers[i]
        layer = model_layers[i]
        
        # --- 1. Attention Importance ---
        # Input: [Hidden_Dim]. Weights: [Hidden_Dim, Hidden_Dim]
        # We use o_proj because it aggregates the heads' output back to the residual stream.
        
        if hasattr(layer, "self_attn"):
            w_attn = layer.self_attn.o_proj.weight
        
        # Check Pythia / GPT-NeoX style
        elif hasattr(layer, "attention"):
            w_attn = layer.attention.dense.weight
        
        # Prepare Input Norm
        inp_attn = input_tensors_attn[i].to(w_attn.device)
        
        # FIX: Ensure we use L2 norm if these are raw activations
        # If input_tensors are already norms, skip this.
        if inp_attn.dim() > 1: 
            # Collapse (Batch, Seq, Hidden) -> (Hidden) using L2 norm
            inp_norm_attn = torch.norm(inp_attn.float(), p=2, dim=0)
        else:
            inp_norm_attn = inp_attn.float()
            
        # WANDA Calculation: |W| * ||X||
        # |W| is [Out, In], inp_norm is [In]. 
        # We assume standard WANDA: element-wise product of magnitude and norm.
        # Broadcasting: [Out, In] * [1, In]
        w_abs = torch.abs(w_attn)
        
        # FIX: Use SUM to capture total importance mass (consistent with MLP)
        wanda_score_attn = torch.mean(w_abs * inp_norm_attn)
        layer_scores['attn'][i] = wanda_score_attn.item()

        # --- 2. MLP Importance ---
        # FIX: Use Gate and Up Projections. 
        # down_proj expects intermediate inputs (size 11008), but we only have block inputs (size 4096).
        # We sum the importance of both 'entry' weights to get the total block sensitivity.
        
        if hasattr(layer.mlp, "down_proj"):
            w_down = layer.mlp.down_proj.weight
        
        # Pythia / GPT-NeoX
        elif hasattr(layer.mlp, "dense_4h_to_h"):
            w_down = layer.mlp.dense_4h_to_h.weight
        
        inp_mlp = input_tensors_mlp[i].to(w_down.device)
        if inp_mlp.dim() > 1:
            inp_norm_mlp = torch.norm(inp_mlp.float(), p=2, dim=0)
        else:
            inp_norm_mlp = inp_mlp.float()
            
        # Calculate scores for both entry layers
        score_down = torch.mean(torch.abs(w_down) * inp_norm_mlp)
        
        # Combine them (Total MLP Input Sensitivity)
        layer_scores['mlp'][i] = (score_down).item()

    return layer_scores

def pruning_ratio_calculator(
    config,
    layer_scores: dict, 
    layer_sizes: dict,
    global_pruning_target: float, # Treated as "Target Keep Ratio" (e.g., 0.375)
    min_k_priority: tuple = (2, 2),
    MAX_LAYER_PRUNING: float = 0.72,
    EPS: float = 1e-4,
    MIN_HEADS: int = 6,
    layer_protection_strength: float = 0.15
) -> dict:
    num_layers = config.num_hidden_layers
    all_layer_indices = set(range(num_layers))

    # --- 1. Setup & Constants ---
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)
    
    # Group Size (GQA alignment)
    group_size = num_q_heads // num_kv_heads
    total_groups = num_kv_heads 
    mlp_block_size = 32 #256

    # [NEW] Calculate Valid Steps (Synced with Breeder)
    # This determines the "Atomic Unit" of pruning for the calculator.
    if num_q_heads > num_kv_heads:
        common_divisor = math.gcd(total_groups, group_size)
        attn_alignment_step = total_groups // common_divisor
        # For Llama 3: 8 // 2 = 4 Steps. (32/4 = 8 heads per step)
        valid_attn_steps = total_groups // attn_alignment_step
    else:
        # MHA: Full granularity allowed
        valid_attn_steps = total_groups
    
    # Pruning Priority (Lower = Prune this type earlier/more aggressively)
    # 0.8 means MLPs effective score is 80% of actual, making them "look worse" so they go first.
    MLP_PRUNE_PRIORITY = 1.0 #0.8 

    # Calculate Budget
    total_attn_params = sum(layer_sizes['attn'].values())
    total_mlp_params = sum(layer_sizes['mlp'].values())
    total_params = total_attn_params + total_mlp_params
    
    # Treat input as "Target Keep Ratio" (e.g. 0.375 means Keep 37.5%)
    target_keep_params = total_params * global_pruning_target 

    # --- Helper: Layer Index Extractor ---
    # max_layer_idx = max([n for n in layer_sizes['attn'].keys()] + [0])
    # Safe extraction depending on if keys are strings or ints
    keys = list(layer_sizes['attn'].keys())
    active_indices = set(keys)

    inactive_indices = all_layer_indices - active_indices

    if isinstance(keys[0], str):
        # Extracts "20" from "model.layers.20.self_attn"
        max_layer_idx = max([int(re.search(r'\.(\d+)\.', n).group(1)) for n in keys] + [0])
    else:
        max_layer_idx = max(keys + [0])

    # --- 2. Vectorization & Score Biasing ---
    def vectorize_module(module_type, max_prune):
        # Get raw names
        raw_names = list(layer_sizes[module_type].keys())
        
        # --- PROTECTION LOGIC ---
        weighted_scores = {}
        for n in raw_names:
            # Handle string vs int keys for position factor
            if isinstance(n, str):
                match = re.search(r'\.(\d+)\.', n)
                idx = int(match.group(1)) if match else 0
            else:
                idx = int(n)
 
            position_factor = idx / (max_layer_idx + 1e-6)
            
            # Boost score based on depth (Last layers get boost)
            weighted_scores[n] = layer_scores[module_type][n] * (1.0 + layer_protection_strength * position_factor)

        # Sort by WEIGHTED score (Lowest score = Pruned First)
        names = sorted(raw_names, key=lambda x: weighted_scores[x])
        
        # Create Arrays
        sizes = np.array([layer_sizes[module_type][n] for n in names], dtype=np.int64)
        
        if module_type == 'attn':
            # max_units_val = total_groups
            # step_sizes = sizes // max_units_val
            
            # # --- FIX: Respect BOTH Min Heads AND Max Pruning ---
            # # 1. Minimum groups based on MAX_LAYER_PRUNING (e.g. keep 20%)
            # min_from_ratio = np.ceil(total_groups * (1.0 - max_prune))
            
            # # 2. Minimum groups based on MIN_HEADS hard constraint
            # min_from_heads = np.ceil(MIN_HEADS / group_size)
            
            # # Take the stricter (higher) floor
            # final_min_groups = int(max(1, max(min_from_ratio, min_from_heads)))
            
            # min_keep_units = np.full_like(sizes, final_min_groups)
            # max_units = np.full_like(sizes, max_units_val)
           
            #__-----
            max_units_val = valid_attn_steps
            step_sizes = sizes // max_units_val
            
            # --- FIX: Respect BOTH Min Heads AND Max Pruning ---
            # 1. Minimum groups based on MAX_LAYER_PRUNING (e.g. keep 20%)
            min_from_ratio = np.ceil(max_units_val * (1.0 - max_prune))
            
            # 2. Minimum steps based on MIN_HEADS (Converted to Steps)
            # Heads per step = Total Heads / Total Steps
            heads_per_step = num_q_heads / max_units_val
            min_from_heads = np.ceil(MIN_HEADS / heads_per_step)
            
            final_min_groups = int(max(1, max(min_from_ratio, min_from_heads)))
            
            min_keep_units = np.full_like(sizes, final_min_groups)
            max_units = np.full_like(sizes, max_units_val)
            
        else: # MLP
            max_units_val = sizes // mlp_block_size
            step_sizes = np.full_like(sizes, mlp_block_size)
            min_keep_units = np.ceil((sizes * (1.0 - max_prune)) / mlp_block_size).astype(np.int64)
            max_units = max_units_val

        return names, sizes, step_sizes, min_keep_units, max_units

    # Prepare Arrays
    attn_names, attn_sizes, attn_steps, attn_min_units, attn_max_units = vectorize_module('attn', MAX_LAYER_PRUNING)
    mlp_names, mlp_sizes, mlp_steps, mlp_min_units, mlp_max_units = vectorize_module('mlp', MAX_LAYER_PRUNING)
    
    # --- 3. Allocation (Random Start) ---
    def generate_random_allocation(sizes, min_units, max_units, max_prune):
        min_ratios = 1.0 - max_prune
        rand_ratios = np.random.uniform(low=min_ratios, high=1.0, size=len(sizes))
        raw_target_units = (max_units * rand_ratios)
        units = np.round(raw_target_units).astype(np.int64)
        units = np.clip(units, min_units, max_units)
        return units

    attn_units = generate_random_allocation(attn_sizes, attn_min_units, attn_max_units, MAX_LAYER_PRUNING)
    # MLP gets slightly looser start to allow the budget loop to tighten it
    mlp_units = generate_random_allocation(mlp_sizes, mlp_min_units, mlp_max_units, MAX_LAYER_PRUNING) 
    
    current_active = np.sum(attn_units * attn_steps) + np.sum(mlp_units * mlp_steps)
    param_diff = current_active - target_keep_params
    TOLERANCE_PARAMS = total_params * 0.001 
    
    # --- 4. UNIFIED BUDGET LOOP (The Priority Fix) ---
    
    # A. Create a unified list of candidates
    pruning_candidates = []
    
    # Add Attn
    for i, name in enumerate(attn_names):
        # Retrieve score (re-calc weighted score logic or grab raw)
        # We use raw score logic here, but biased by type
        raw_score = layer_scores['attn'][name]
        pruning_candidates.append({
            'type': 'attn', 'idx': i, 'score': raw_score, 'cost': attn_steps[i]
        })
        
    # Add MLP (With Bias)
    for i, name in enumerate(mlp_names):
        raw_score = layer_scores['mlp'][name]
        # Multiply by Priority Factor (<1.0 makes it look "less important", so it gets pruned first)
        biased_score = raw_score * MLP_PRUNE_PRIORITY 
        pruning_candidates.append({
            'type': 'mlp', 'idx': i, 'score': biased_score, 'cost': mlp_steps[i]
        })
        
    # B. Sort: Lowest Score -> Prune First
    pruning_candidates.sort(key=lambda x: x['score'])
    
    # C. Loop
    for _ in range(15): # Increased iterations slightly for convergence
        if abs(param_diff) < TOLERANCE_PARAMS: break
            
        # --- PRUNE MODE (Too many params) ---
        if param_diff > 0: 
            for cand in pruning_candidates:
                if param_diff <= TOLERANCE_PARAMS: break
                
                idx = cand['idx']
                
                if cand['type'] == 'attn':
                    if attn_units[idx] > attn_min_units[idx]:
                        available = attn_units[idx] - attn_min_units[idx]
                        needed = int(np.ceil(param_diff / cand['cost']))
                        remove = min(needed, available)
                        
                        attn_units[idx] -= remove
                        param_diff -= (remove * cand['cost'])
                        
                else: # MLP
                    if mlp_units[idx] > mlp_min_units[idx]:
                        available = mlp_units[idx] - mlp_min_units[idx]
                        needed = int(np.ceil(param_diff / cand['cost']))
                        remove = min(needed, available)
                        
                        mlp_units[idx] -= remove
                        param_diff -= (remove * cand['cost'])

        # --- RESTORE MODE (Too few params) ---
        else: 
            abs_diff = abs(param_diff)
            # Restore Reverse (Highest Score First)
            restore_candidates = sorted(pruning_candidates, key=lambda x: x['score'], reverse=True)
            
            for cand in restore_candidates:
                if abs_diff <= TOLERANCE_PARAMS: break
                
                idx = cand['idx']
                
                if cand['type'] == 'attn':
                    if attn_units[idx] < attn_max_units[idx]:
                        available = attn_max_units[idx] - attn_units[idx]
                        needed = int(np.ceil(abs_diff / cand['cost']))
                        add_count = min(needed, available)
                        
                        attn_units[idx] += add_count
                        abs_diff -= (add_count * cand['cost'])
                        
                else: # MLP
                    if mlp_units[idx] < mlp_max_units[idx]:
                        available = mlp_max_units[idx] - mlp_units[idx]
                        needed = int(np.ceil(abs_diff / cand['cost']))
                        add_count = min(needed, available)
                        
                        mlp_units[idx] += add_count
                        abs_diff -= (add_count * cand['cost'])
            
            param_diff = -abs_diff

    # --- 5. Final Output Formatting ---
    
    # Calculate KEEP ratios
    attn_keep_ratios = attn_units / attn_max_units
    mlp_keep_ratios = mlp_units / mlp_max_units
    
    result = {'attn': {}, 'mlp': {}}
    
    # Return SPARSITY (1.0 - Keep)
    for name, keep_ratio in zip(attn_names, attn_keep_ratios):
        result['attn'][name] = float(1.0 - keep_ratio) 
        
    for name, keep_ratio in zip(mlp_names, mlp_keep_ratios):
        result['mlp'][name] = float(1.0 - keep_ratio)

    # Defaults
    for k in layer_sizes['attn']:
        if k not in result['attn']: result['attn'][k] = 0.0
    for k in layer_sizes['mlp']:
        if k not in result['mlp']: result['mlp'][k] = 0.0
    
    # Verify
    final_active_params = np.sum(attn_units * attn_steps) + np.sum(mlp_units * mlp_steps)

    # Inactive Indices (Forced to 1.0 Sparsity)
    for idx in inactive_indices:
        result['attn'][idx] = 1.0
        result['mlp'][idx] = 1.0
    
    print(f"Target Keep:   {int(target_keep_params):,} params")
    print(f"Actual Keep:   {int(final_active_params):,} params")
    
    return result

def pruning_ratio_calculator_overengineered(
    config,
    layer_scores: dict, 
    layer_sizes: dict,
    global_pruning_target: float,  # Target Keep Ratio (e.g., 0.375 for 2.7B)
    min_k_priority: tuple = (2, 2),
    MAX_LAYER_PRUNING: float = 0.72,
    EPS: float = 1e-4,
    MIN_HEADS: int = 6,            # The absolute floor for middle/end layers
    layer_protection_strength: float = 0.15
) -> dict:
    num_layers = config.num_hidden_layers
    all_layer_indices = set(range(num_layers))

    # --- 1. Setup & Constants ---
    num_q_heads = config.num_attention_heads
    # Pythia/GPT-NeoX lacks num_key_value_heads; default to num_q_heads (MHA)
    num_kv_heads = getattr(config, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = num_q_heads

    is_llama_7b = (
        config.hidden_size == 4096 and 
        config.num_hidden_layers == 32 and 
        config.intermediate_size==11008
    )
    # Llama 3.1 8B
    is_llama3_8b = (
        config.num_hidden_layers == 32 and 
        config.hidden_size == 4096 and 
        config.intermediate_size == 14336
    )

    # Llama 3.1 70B (Pruning target for your 50B variant)
    is_llama3_70b = (
        config.num_hidden_layers == 80 and 
        config.hidden_size == 8192 and 
        config.intermediate_size == 28672
    )

    # Qwen 2.5 14B
    # Note: Qwen models often have non-standard intermediate ratios
    is_qwen2_5_14b = (
        config.num_hidden_layers == 48 and 
        config.hidden_size == 5120 and 
        config.intermediate_size == 13824
    )
    
    # Gemma 2 check (requires specific alignment for Sliding Window Attention/GQA)
    is_gemma = "gemma2" in getattr(config, "model_type", "").lower()
    
    group_size = num_q_heads // num_kv_heads
    total_groups = num_kv_heads 
    
    # MLP block size: Gemma/Llama use 32; Pythia can use 64 for efficiency
    mlp_block_size = 32 if not ("pythia" in str(config.architectures).lower()) else 64

    # Calculate valid attention steps for GQA alignment
    if num_q_heads > num_kv_heads:
        common_divisor = math.gcd(total_groups, group_size)
        attn_alignment_step = total_groups // common_divisor
        valid_attn_steps = total_groups // attn_alignment_step
    else:
        valid_attn_steps = total_groups
    
    MLP_PRUNE_PRIORITY = 1.0 

    # Calculate Budget
    total_attn_params = sum(layer_sizes['attn'].values())
    total_mlp_params = sum(layer_sizes['mlp'].values())
    total_params = total_attn_params + total_mlp_params
    target_keep_params = total_params * global_pruning_target 

    # Helper: Extract indices to detect depth
    keys = list(layer_sizes['attn'].keys())
    active_indices = set(keys)
    inactive_indices = all_layer_indices - active_indices

    if isinstance(keys[0], str):
        max_layer_idx = max([int(re.search(r'\.(\d+)\.', n).group(1)) for n in keys] + [0])
    else:
        max_layer_idx = max(keys + [0])

    # --- 2. Vectorization with Depth-Aware Protection ---
    def vectorize_module(module_type, max_prune):
        raw_names = list(layer_sizes[module_type].keys())
        weighted_scores = {}
        
        for n in raw_names:
            if isinstance(n, str):
                match = re.search(r'\.(\d+)\.', n)
                idx = int(match.group(1)) if match else 0
            else:
                idx = int(n)
            
            # Position factor (0.0 at layer 0, 1.0 at final layer)
            position_factor = idx / (max_layer_idx + 1e-6)
            
            # Depth-based protection: Boost early layer importance to avoid "Thin Stem"
            # Early layers get an artificial score boost so they are pruned LAST
            # depth_protection = 0.3 * (1.0 - position_factor) 
            # weighted_scores[n] = layer_scores[module_type][n] * (1.0 + layer_protection_strength * position_factor + depth_protection)
            
            # High protection at the Stem (0.0) AND the "Reasoning Belt" (0.4 - 0.6)
            # Using a Gaussian-style peak around the middle layers
            middle_protection = 0.25 * np.exp(-((position_factor - 0.5) ** 2) / 0.05)
            stem_protection = 0.35 * np.exp(-(position_factor ** 2) / 0.02)
            
            # Combine protections so the middle doesn't drop too low
            total_protection = max(middle_protection, stem_protection)
            
            weighted_scores[n] = layer_scores[module_type][n] * (1.0 + total_protection)


        names = sorted(raw_names, key=lambda x: weighted_scores[x])
        sizes = np.array([layer_sizes[module_type][n] for n in names], dtype=np.int64)
        
        if module_type == 'attn':
            max_units_val = valid_attn_steps
            step_sizes = sizes // max_units_val
            
            # --- NEW: Dynamic MIN_HEADS per Layer ---
            # We calculate a floor for each specific layer index
            min_keep_units = []
            for n in names:
                idx = int(re.search(r'\.(\d+)\.', n).group(1)) if isinstance(n, str) else int(n)
                depth_ratio = idx / num_layers

                is_swa_layer = is_gemma and (idx % 2 == 0)
    
                if is_swa_layer:
                    # Never drop below 50% heads for SWA layers to keep local context window stable
                    dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.5))
                elif is_llama_7b:
                    # Dynamic floor logic:
                    if depth_ratio < 0.20: # First 6-7 layers
                        dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.5)) # Keep at least 16 heads
                    # elif depth_ratio < 0.40: # Middle-early
                    elif 0.35 <= depth_ratio <= 0.70: # THE REASONING BELT
                        dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.4)) # Keep 10-12 heads
                    else:
                        dynamic_min_heads = MIN_HEADS # Standard floor for deep layers

                # elif is_llama3_8b or is_llama3_70b or is_qwen2_5_14b:
                else:
                    # Dynamic floor logic:
                    if depth_ratio < 0.20: # First 6-7 layers
                        ratio=random.choice([0.65,0.7,0.75])
                        dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * ratio)) # Keep at least 16 heads
                    elif 0.35 <= depth_ratio <= 0.70: # THE REASONING BELT
                        ratio=random.choice([0.5,0.55,0.6,0.65])
                        dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.4)) # Keep 10-12 heads
                    # elif depth_ratio > 0.85: # THE EXIT (Final 4-5 layers)
                    #     # CRITICAL: Prevent the ' A' vs 'A' flip by keeping the exit wide
                    #     dynamic_min_heads = max(MIN_HEADS, int(num_q_heads * 0.75)) # ~24 heads
                    else:
                        dynamic_min_heads = MIN_HEADS # Standard floor for deep layers

                
                heads_per_step = num_q_heads / max_units_val
                min_from_heads = np.ceil(dynamic_min_heads / heads_per_step)
                min_from_ratio = np.ceil(max_units_val * (1.0 - max_prune))
                
                min_keep_units.append(int(max(1, max(min_from_ratio, min_from_heads))))
            
            min_keep_units = np.array(min_keep_units)
            max_units = np.full_like(sizes, max_units_val)
            
        else: # MLP
            max_units_val = sizes // mlp_block_size
            step_sizes = np.full_like(sizes, mlp_block_size)
            # MLPs also get a depth-aware floor to keep the early FFNs healthy
            min_keep_units = []
            final_max_units = []

            for i, n in enumerate(names):
                idx = int(re.search(r'\.(\d+)\.', n).group(1)) if isinstance(n, str) else int(n)
                depth_ratio = idx / num_layers


                if is_llama_7b: 
                    # dynamic_max_prune = max_prune if depth_ratio > 0.25 else 0.60
                    # Force higher MLP capacity in the middle
                    if 0.35 <= depth_ratio <= 0.75:
                        # MLP reasoning floor: approx 45% of 11008 = 4928
                        dynamic_max_prune = 0.55 
                    elif depth_ratio < 0.20:
                        dynamic_max_prune = 0.60 # Stem protection
                    else:
                        dynamic_max_prune = max_prune # Allow tail/transition pruning
                else:
                    dynamic_max_prune = max_prune if depth_ratio > 0.25 else 0.60
                # elif is_llama3_8b or is_llama3_70b or is_qwen2_5_14b:
                #     if depth_ratio < 0.20: # THE STEM (Knowledge Keys)
                #         # Shallow layers are the 'index'. If you prune these too much, 
                #         # the model can't 'find' the knowledge in later layers.
                #         dynamic_max_prune = 0.35 # Keep ~9300 (Original 14336)

                #     elif 0.35 <= depth_ratio <= 0.75: # THE REASONING BELT
                #         # This is the 'Knowledge Relational' core.
                #         # To pass MMLU/ARC-c, you need more than 5000 parameters here.
                #         dynamic_max_prune = 0.45 # Keep ~7800

                #     elif depth_ratio > 0.85: # THE EXIT (Token Refinement)
                #         # CRITICAL: This fixes the ' A' vs 'A' token flip.
                #         # The final layers need high capacity to map internal logic to correct vocabulary IDs.
                #         dynamic_max_prune = 0.30 # Keep ~10000+ (Very wide)

                #     else: # TRANSITION LAYERS
                #         dynamic_max_prune = max_prune # Allow more aggressive pruning here (e.g., 0.60)

                # dynamic_max_prune = max_prune if depth_ratio > 0.25 else 0.60
                final_max_units.append(max_units_val[i])
                    
                min_keep_units.append(np.ceil((layer_sizes['mlp'][n] * (1.0 - dynamic_max_prune)) / mlp_block_size))
            
            min_keep_units = np.array(min_keep_units).astype(np.int64)
            # max_units = max_units_val
            max_units = np.array(final_max_units).astype(np.int64)

        return names, sizes, step_sizes, min_keep_units, max_units

    # Prepare Arrays
    attn_names, attn_sizes, attn_steps, attn_min_units, attn_max_units = vectorize_module('attn', MAX_LAYER_PRUNING)
    mlp_names, mlp_sizes, mlp_steps, mlp_min_units, mlp_max_units = vectorize_module('mlp', MAX_LAYER_PRUNING)
    
    # --- 3. Allocation & Budget Loop ---
    # We start with the MINIMUM allowed per layer to give us headroom to add back
    attn_units = attn_min_units.copy()
    mlp_units = mlp_min_units.copy()
    
    current_active = np.sum(attn_units * attn_steps) + np.sum(mlp_units * mlp_steps)
    param_diff = target_keep_params - current_active
    TOLERANCE_PARAMS = total_params * 0.001 
    
    # Unified list for restoring params (Highest Score = Add back first)
    restore_candidates = []
    for i, name in enumerate(attn_names):
        restore_candidates.append({'type': 'attn', 'idx': i, 'score': layer_scores['attn'][name], 'cost': attn_steps[i]})
    for i, name in enumerate(mlp_names):
        restore_candidates.append({'type': 'mlp', 'idx': i, 'score': layer_scores['mlp'][name] * MLP_PRUNE_PRIORITY, 'cost': mlp_steps[i]})
    
    restore_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Simple allocation loop
    for cand in restore_candidates:
        if param_diff <= TOLERANCE_PARAMS: break
        idx = cand['idx']
        if cand['type'] == 'attn':
            available = attn_max_units[idx] - attn_units[idx]
            add = min(available, int(param_diff // cand['cost']))
            attn_units[idx] += add
            param_diff -= (add * cand['cost'])
        else:
            available = mlp_max_units[idx] - mlp_units[idx]
            add = min(available, int(param_diff // cand['cost']))
            mlp_units[idx] += add
            param_diff -= (add * cand['cost'])

    # --- 4. Final Formatting ---
    attn_keep_ratios = attn_units / attn_max_units
    mlp_keep_ratios = mlp_units / mlp_max_units
    result = {'attn': {}, 'mlp': {}}
    
    for name, keep_ratio in zip(attn_names, attn_keep_ratios):
        result['attn'][name] = float(1.0 - keep_ratio)
    for name, keep_ratio in zip(mlp_names, mlp_keep_ratios):
        result['mlp'][name] = float(1.0 - keep_ratio)
    
    # Handle missing layers (force to 1.0 sparsity if they were dropped during evolution)
    for idx in inactive_indices:
        result['attn'][idx] = 1.0
        result['mlp'][idx] = 1.0
    
    final_active = np.sum(attn_units * attn_steps) + np.sum(mlp_units * mlp_steps)
    print(f"Target Keep:   {int(target_keep_params):,} params")
    print(f"Actual Keep:   {int(final_active):,} params")
    
    return result

def compute_pruning_masks(
    args, 
    model, 
    target_width, 
    layer_scores=None, 
    attn_layer_sizes=None, 
    mlp_layer_sizes=None, 
    input_tensors_attn=None, 
    input_tensors_mlp=None, 
    best_sparsity=[],
    k=50,
    base_grad_vecs_attn=None,
    base_grad_vecs_mlp=None
):
    print(f'Getting sparsities for target width {target_width}')

    # --- 1. Setup ---
    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    layer_sizes = {'attn': attn_layer_sizes, 'mlp': mlp_layer_sizes}
    
    is_pythia = "GPTNeoX" in model.config.architectures[0]

    # Identify indices to ignore based on the CONFIG (0 values)
    # We combine the class-based check with the config-list check
    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
        prefix = "layers"
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
        prefix = "layers"
    elif hasattr(model.model.model, 'language_model'):
        model_layers = model.model.model.language_model.layers
        prefix = "layers"

    # Identify skip indices from both module type AND config list zeros
    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention) or 
        (hasattr(model_config, 'hidden_size_list') and model_config.hidden_size_list[idx] == 0)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP) or 
        (hasattr(model_config, 'intermediate_size_list') and model_config.intermediate_size_list[idx] == 0)
    } 

    # Params
    num_q_heads_base = model_config.num_attention_heads
    num_kv_heads_base = getattr(model_config, 'num_key_value_heads', num_q_heads_base)
    head_dim = getattr(model_config, 'head_dim', model_config.hidden_size // num_q_heads_base)
    
    is_gqa_or_mqa = num_q_heads_base > num_kv_heads_base
    group_size = num_q_heads_base // num_kv_heads_base

    # --- 2. Calculate Sparsity ---
    if not args.eval_after_evolsearch:
        if not best_sparsity:
            sparsity = pruning_ratio_calculator(model_config, layer_scores, layer_sizes, target_width)
        else:
            sparsity = get_evolutionary_pruning_ratios(
                model_config, layer_scores, layer_sizes, target_width, best_sparsity
            )
    else:
        sparsity = best_sparsity

    # --- 3. Pruning Loop ---
    final_masks = {}
    attn_sparsity = sparsity['attn']
    mlp_sparsity = sparsity['mlp']

    # print(attn_sparsity)
    # print(mlp_sparsity)

    for i, layer in enumerate(model_layers):
        if i not in attn_sparsity and i not in mlp_sparsity:
            if str(i) in attn_sparsity and str(i) in mlp_sparsity:
                i = str(i)

        # Determine current hidden size for this specific layer
        if hasattr(model_config, 'hidden_size_list'):
            curr_hidden_size = model_config.hidden_size_list[i]
            curr_num_q_heads = model_config.num_attention_heads_list[i]
            # Maintain GQA ratio if applicable
            kv_ratio = num_q_heads_base // num_kv_heads_base
            curr_num_kv_heads = curr_num_q_heads // kv_ratio if is_gqa_or_mqa else curr_num_q_heads
        else:
            curr_hidden_size = model_config.hidden_size
            curr_num_q_heads = num_q_heads_base
            curr_num_kv_heads = num_kv_heads_base

        # =======================================================
        # ATTENTION BLOCK 
        # =======================================================
        if i not in skip_attn_indices and i in layer_sizes['attn']:
            # raw_o_proj = layer.self_attn.o_proj
            if hasattr(layer, "self_attn"):
                raw_o_proj =  layer.self_attn.o_proj
            
            # Check Pythia / GPT-NeoX style
            elif hasattr(layer, "attention"):
                raw_o_proj = layer.attention.dense

            o_proj_module = getattr(raw_o_proj, "base_layer", raw_o_proj)
            
            weight = dequantize(o_proj_module)
            C_out, C_in = weight.shape
            scale_factor = input_tensors_attn[i].to(weight.device)
            # print(weight.shape, scale_factor.shape)
            if base_grad_vecs_attn is None:
                importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)
            else:
                grad_sensitivity=base_grad_vecs_attn[i].to(weight.device)
                importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0) * grad_sensitivity

                # print('here!')
                # print(grad_sensitivity)
                # print(torch.abs(weight) * scale_factor.unsqueeze(0))
                # print(importance_scores) 
                
            structured_scores = importance_scores.sum(dim=0) # [curr_hidden_size] 

            if is_gqa_or_mqa:
                kv_full_dim = curr_num_kv_heads * head_dim
                mask_kv_1d = torch.ones(kv_full_dim, dtype=torch.bool, device=weight.device)

                q_scores = structured_scores.view(curr_num_q_heads, head_dim).sum(dim=1)
                
                raw_keep_count = curr_num_q_heads * (1.0 - attn_sparsity[i])
                step_size = curr_num_kv_heads
                target_q_heads = int(round(raw_keep_count / step_size) * step_size)
                target_q_heads = max(step_size, min(target_q_heads, curr_num_q_heads))
                
                _, top_q_indices = torch.topk(q_scores, k=target_q_heads)
                
                mask_q_1d = torch.zeros(curr_hidden_size, dtype=torch.bool, device=weight.device)
                for q_idx in top_q_indices:
                    mask_q_1d[q_idx*head_dim : (q_idx+1)*head_dim] = True

            #########################################################
            # GQA-AWARE HEAD SELECTION LOGIC (Grouped Importance)
            #########################################################
            # if is_gqa_or_mqa:
            #     # 1. Calculate Importance per individual Q-head
            #     q_scores_per_head = structured_scores.view(curr_num_q_heads, head_dim).sum(dim=1)
                
            #     # --- CHANGE START ---
            #     # 2. AGGREGATE: Sum Q-scores into their respective KV groups
            #     # Reshape to [num_kv, group_size] and sum across the group dimension
            #     group_scores = q_scores_per_head.view(curr_num_kv_heads, group_size).sum(dim=1)
                
            #     # 3. SELECT GROUPS: Decide how many WHOLE groups to keep
            #     # We calculate how many KV heads worth of Q-heads to keep
            #     num_groups_to_keep = max(1, int(round(curr_num_q_heads * (1.0 - attn_sparsity[i]) / group_size)))
                
            #     # 4. RANK GROUPS: topk now picks the best GROUPS, not best individual heads
            #     _, top_group_indices = torch.topk(group_scores, k=num_groups_to_keep)
                
            #     # 5. MASKING: Activate ALL Q-heads within the selected groups
            #     mask_q_1d = torch.zeros(curr_hidden_size, dtype=torch.bool, device=weight.device)
            #     mask_kv_1d = torch.ones(curr_num_kv_heads * head_dim, dtype=torch.bool, device=weight.device)

            #     for group_idx in top_group_indices:
            #         # Activate the entire block of Q-heads belonging to this KV group
            #         start = group_idx * group_size * head_dim
            #         end = (group_idx + 1) * group_size * head_dim
            #         mask_q_1d[start:end] = True
            #     # --- CHANGE END ---
            # #########################################################
            else:
                head_scores = structured_scores.view(curr_num_q_heads, head_dim).sum(dim=1)
                num_keep = max(1, int(round(curr_num_q_heads * (1.0 - attn_sparsity[i]))))
                
                _, top_indices = torch.topk(head_scores, k=num_keep)
                
                mask_q_1d = torch.zeros(curr_hidden_size, dtype=torch.bool, device=weight.device)
                mask_kv_1d = torch.zeros(curr_hidden_size, dtype=torch.bool, device=weight.device)

                for h_idx in top_indices:
                    start = h_idx * head_dim
                    end = (h_idx + 1) * head_dim
                    mask_q_1d[start:end] = True
                    mask_kv_1d[start:end] = True
            
            # if '70B' in args.model_name_or_path:
            #     final_masks[f"{prefix}.{i}.self_attn.q_proj"] = mask_q_1d
            #     final_masks[f"{prefix}.{i}.self_attn.k_proj"] = mask_kv_1d
            #     final_masks[f"{prefix}.{i}.self_attn.v_proj"] = mask_kv_1d
            # else:
            if 'gemma' in model.config.__class__.__name__.lower():
                mask_2d_q = mask_q_1d.unsqueeze(0).expand(C_in, -1).detach()
                mask_2d_kv = mask_kv_1d.unsqueeze(-1).expand(-1,C_out).detach()
            else:
                mask_2d_q = mask_q_1d.unsqueeze(0).expand(C_out, -1).detach()
                mask_2d_kv = mask_kv_1d.unsqueeze(1).expand(-1, C_out).detach()

            # Pythia (Fused Projections)
            if is_pythia:
                # Since Pythia fuses QKV, you likely need a single mask.
                # If your mask_2d_q and mask_2d_kv were calculated for the full dim:
                # final_masks[f"{prefix}.{i}.attention.query_key_value"] = torch.cat([mask_q, mask_k, mask_v], dim=0)
                final_masks[f"{prefix}.{i}.attention.query_key_value"] = mask_2d_q # Or your combined mask
                # print(weight.shape)
                # print(mask_2d_q.shape)
            else:
                final_masks[f"{prefix}.{i}.self_attn.q_proj"] = mask_2d_q
                final_masks[f"{prefix}.{i}.self_attn.k_proj"] = mask_2d_kv
                final_masks[f"{prefix}.{i}.self_attn.v_proj"] = mask_2d_kv

        # =======================================================
        # MLP BLOCK
        # =======================================================
        if i not in skip_mlp_indices and i in layer_sizes['mlp']:
            # raw_down_proj = layer.mlp.down_proj
            if hasattr(layer.mlp, "down_proj"):
                raw_down_proj = layer.mlp.down_proj
            
            # Pythia / GPT-NeoX
            elif hasattr(layer.mlp, "dense_4h_to_h"):
                raw_down_proj = layer.mlp.dense_4h_to_h

            down_proj_module = getattr(raw_down_proj, "base_layer", raw_down_proj)

            weight = dequantize(down_proj_module)
            C_out, C_in = weight.shape
            scale_factor = input_tensors_mlp[i].to(weight.device)
            
            if base_grad_vecs_mlp is None:
                importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)
            else:
                grad_sensitivity=base_grad_vecs_mlp[i].to(weight.device)
                importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0) * grad_sensitivity

                # print('here!')
                # print(grad_sensitivity)
                # print(torch.abs(weight) * scale_factor.unsqueeze(0))
                # print(importance_scores) 

                # exit()
            structured_scores = importance_scores.sum(dim=0)
            
            num_to_keep = max(1, int(C_in * (1 - mlp_sparsity[i])))

            _, sorted_indices = torch.topk(structured_scores, k=num_to_keep)
            
            mask_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
            mask_1d[sorted_indices] = True

            # if '70B' in args.model_name_or_path:
            #     final_masks[f"{prefix}.{i}.mlp.gate_proj"] = mask_1d
            #     final_masks[f"{prefix}.{i}.mlp.up_proj"] = mask_1d
            # else:
            mask_2d = mask_1d.unsqueeze(0).expand(C_out, C_in)
            mask_2d_transposed = mask_2d.t().detach()
            
            if is_pythia:
                final_masks[f"{prefix}.{i}.mlp.dense_h_to_4h"] = mask_2d_transposed
            else:
                final_masks[f"{prefix}.{i}.mlp.gate_proj"] = mask_2d_transposed
                final_masks[f"{prefix}.{i}.mlp.up_proj"] = mask_2d_transposed

    return final_masks, sparsity
  
def apply_mask_to_grad(grad, mask):
    """Applies the mask to the 11008 dimension of the gradient using broadcasting."""
    D = mask.numel()
    
    if grad.shape[0] == D:
        # Case 1: Shape is [11008, B] (11008 is the rows/output dimension)
        # Reshape mask from [D] to [D, 1]
        return grad * mask.unsqueeze(1)
    
    elif grad.shape[1] == D:
        # Case 2: Shape is [A, 11008] (11008 is the columns/output dimension)
        # Reshape mask from [D] to [1, D]
        return grad * mask.unsqueeze(0)
        
    else:
        # Default: If the mask dimension (11008) is not found, return the unmasked gradient 
        # (or handle as an error if all gradients should be maskable)
        return grad

def add_input_hooks(model, input_tensors_attn, input_tensors_mlp):
    """
    Adds forward hooks to all attention and MLP blocks to capture their input tensors.
    """
    hooks = []
    
    # Use the named_modules approach
    for module_name, module in model.named_modules():
        match = re.search(
            # r'layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)$',
            r'layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj|attention\.dense|mlp\.dense_4h_to_h)$',
            module_name
        )
        if match:
            layer_id = int(match.group(1))
            layer_type = match.group(2)
            
            # 1. Attention Block Hook Function (Defined inside the loop)
            if 'self_attn' in layer_type or 'attention' in layer_type:
                # 💡 FIX 1: Define the hook function here, passing layer_id as a default argument (layer_idx=layer_id)
                def attn_hook_fn(module, input, output, layer_idx=layer_id):
                    if len(input) > 0:
                        hidden_state = input[0].detach() 
                        
                        # Original shape: (bs, seq_len, dim)
                        # Desired shape: (dim, bs * seq_len)
                        reshaped_input = hidden_state.permute(2, 0, 1).reshape(hidden_state.shape[2], -1)
                        
                        input_tensors_attn[layer_idx] = reshaped_input
                    
                if layer_id in input_tensors_attn:
                    hook = module.register_forward_hook(attn_hook_fn)
                    hooks.append(hook)

            # 2. MLP Block Hook Function (Defined inside the loop)
            elif 'mlp' in layer_type:
                # 💡 FIX 1: Define the hook function here, passing layer_id as a default argument (layer_idx=layer_id)
                def mlp_hook_fn(module, input, output, layer_idx=layer_id):
                    if len(input) > 0:
                        hidden_state = input[0].detach() 
                        
                        # Original shape: (bs, seq_len, dim)
                        # Desired shape: (dim, bs * seq_len)
                        reshaped_input = hidden_state.permute(2, 0, 1).reshape(hidden_state.shape[2], -1)
                        
                        input_tensors_mlp[layer_idx] = reshaped_input
                
                if layer_id in input_tensors_mlp:
                    hook = module.register_forward_hook(mlp_hook_fn)
                    hooks.append(hook)
                    
    return hooks

def calculate_base_influence(args, model, dataloader, compute_loss_context_manager, compute_loss, accelerator):

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_layers = model_config.num_hidden_layers
    device = model.device

    active_layers_attn = torch.ones(num_layers, dtype=torch.bool, device=device)
    active_layers_mlp = torch.ones(num_layers, dtype=torch.bool, device=device)
    model.set_active_layers(active_layers_attn, active_layers_mlp, width=1)
    
    # 1. Setup Storage (FP32 Accumulators)
    # We map Layer Index -> FP32 Tensor
    accum_grad_attn = {} 
    accum_grad_mlp = {}

    accum_grad_attn_o = {} 
    accum_grad_mlp_down = {}
    
    # # MWP Storage
    accum_mwp_norm_attn = {layer_idx: 0.0 for layer_idx in range(num_layers)}
    accum_mwp_norm_mlp = {layer_idx: 0.0 for layer_idx in range(num_layers)}

    # NEW: Storage for Variance Calculation (Sum and Sum-of-Squares)
    # We map Layer Index -> Tensor of shape (Hidden_Dim,)
    accum_sum_attn = {} 
    accum_sq_sum_attn = {}
    accum_tokens_count = 0 # Track total tokens observed
    
    # Keep MLP storage if you prune MLPs too
    accum_sum_mlp = {} 
    accum_sq_sum_mlp = {}
    
    input_tensors_attn = {layer_idx: None for layer_idx in range(num_layers)}
    input_tensors_mlp = {layer_idx: None for layer_idx in range(num_layers)}
    
    hooks = add_input_hooks(model, input_tensors_attn, input_tensors_mlp)

    # We will zero grad EVERY batch to extract the raw signal
    model.zero_grad()
    
    total_batches = 0
    total_loss = 0.0

    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
    
    elif hasattr(model.model.model, 'language_model'):
        lm_model = model.model.model.language_model
    
        if hasattr(lm_model, 'layers'):
            model_layers = lm_model.layers

    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP)
    }

    with compute_loss_context_manager():
        for i, test_inputs in enumerate(dataloader):
            total_batches += 1
            
            # 1. Zero Grad (Critical for manual accumulation)
            model.zero_grad()
            
            # 2. Forward & Backward
            try:
                batch_base_loss = compute_loss(model, test_inputs)
            except ValueError:
                batch_base_loss, _ = compute_loss(model, test_inputs, return_runtime=True)
                
            total_loss += batch_base_loss.item()
            batch_base_loss.backward()
            
            # 3. MANUAL FP32 ACCUMULATION
            with torch.no_grad():
                for layer_idx in range(num_layers):
                    # --- Process ATTENTION ---
                    if layer_idx in skip_attn_indices:
                        # Do parameter extraction and accumulation here...
                        continue

                    # --- Process MLP ---
                    if layer_idx in skip_mlp_indices:
                        # Do parameter extraction and accumulation here...
                        continue

                    target_layer = model_layers[layer_idx]

                    attn_filter = 'attention' if hasattr(target_layer, 'attention') else 'self_attn'
                    attn_out_filter = 'o_proj' if 'llama' in args.model_name_or_path.lower() else 'dense'
                    mlp_out_filter = 'gate_proj' if 'llama' in args.model_name_or_path.lower() else 'dense_h_to_4h'

                    # --- ATTENTION ---
                    # Extract raw BF16 grads for this batch
                    attn_grads = parameters_to_vector([
                        p.grad for n, p in target_layer.named_parameters() 
                        if p.grad is not None and attn_filter in n
                    ])

                    # o_proj_grads = parameters_to_vector([
                    #     p.grad for n, p in target_layer.named_parameters() 
                    #     if p.grad is not None and attn_out_filter in n
                    # ])
                    
                    if attn_grads.numel() > 0:
                        # Initialize Buffer on first batch
                        if layer_idx not in accum_grad_attn:
                            # storage_device = 'cpu' if '70B' in args.model_name_or_path else device
                            storage_device = device
                            accum_grad_attn[layer_idx] = torch.zeros_like(attn_grads, device=storage_device) #, dtype=torch.float32, device='cpu')
                            if args.lora_r == 4096:
                                accum_grad_attn[layer_idx] = accum_grad_attn[layer_idx].to('cpu')
                        
                        if accum_grad_attn[layer_idx].device.type == 'cpu':
                            accum_grad_attn[layer_idx] += attn_grads.to('cpu')
                        elif args.lora_r == 4096:
                            accum_grad_attn[layer_idx] += attn_grads.to('cpu')
                        else:
                            accum_grad_attn[layer_idx] += attn_grads

                    # if o_proj_grads.numel() > 0:
                    #     if layer_idx not in accum_grad_attn_o:
                    #         accum_grad_attn_o[layer_idx] = torch.zeros_like(o_proj_grads, device=device)
                    #     accum_grad_attn_o[layer_idx] += o_proj_grads
                    # for n, p in target_layer.named_parameters():
                    #     # We specifically target lora_B because it maps Rank -> Hidden Dim (4096)
                    #     if p.grad is not None and attn_out_filter in n  and 'lora_B' in n:
                    #         if layer_idx not in accum_grad_attn_o:
                    #             # Initialize a 1D tensor of size 4096
                    #             accum_grad_attn_o[layer_idx] = torch.zeros(p.grad.shape[0], device=device)
                            
                    #         # Calculate row-wise L2 norm and accumulate
                    #         # This gives us the "Trace Sensitivity" for each of the 4096 channels
                    #         accum_grad_attn_o[layer_idx] += torch.norm(p.grad, p=2, dim=1)

                    # --- MLP ---
                    mlp_grads = parameters_to_vector([
                        p.grad for n, p in target_layer.named_parameters() 
                        if p.grad is not None and 'mlp' in n
                    ])

                    # down_proj_grads = parameters_to_vector([
                    #     p.grad for n, p in target_layer.named_parameters() 
                    #     if p.grad is not None and mlp_out_filter in n
                    # ])

                    if mlp_grads.numel() > 0:
                        if layer_idx not in accum_grad_mlp:
                            # storage_device = 'cpu' if '70B' in args.model_name_or_path else device
                            storage_device = device
                            accum_grad_mlp[layer_idx] = torch.zeros_like(mlp_grads, device=storage_device) #, dtype=torch.float32, device='cpu')

                            if args.lora_r == 4096:
                                accum_grad_mlp[layer_idx] = accum_grad_mlp[layer_idx].to('cpu')
                        
                        if accum_grad_mlp[layer_idx].device.type == 'cpu':
                            accum_grad_mlp[layer_idx] += mlp_grads.to('cpu') #.to(torch.float32) #.cpu()
                        elif args.lora_r == 4096:
                            accum_grad_mlp[layer_idx] += mlp_grads.to('cpu')
                        else:
                            accum_grad_mlp[layer_idx] += mlp_grads

                    # if down_proj_grads.numel() > 0:
                    #     if layer_idx not in accum_grad_mlp_down:
                    #         accum_grad_mlp_down[layer_idx] = torch.zeros_like(mlp_grads, device=device)
                    #     accum_grad_mlp_down[layer_idx] += down_proj_grads

                    # for n, p in target_layer.named_parameters():
                    #     if p.grad is not None and mlp_out_filter in n and 'lora_B' in n:
                    #         if layer_idx not in accum_grad_mlp_down:
                    #             accum_grad_mlp_down[layer_idx] = torch.zeros(p.grad.shape[0], device=device)
                            
                    #         accum_grad_mlp_down[layer_idx] += torch.norm(p.grad, p=2, dim=1)

            # 4. MWP Accumulation (Same as before) -> accumulates mean (loudness)
            for layer_idx in range(num_layers):
                if input_tensors_attn[layer_idx] is not None:
                    cur_input = input_tensors_attn[layer_idx] #.detach()
                    # batch_norm = torch.norm(cur_input, p=2, dim=1).sum(dim=0).float().cpu() # Move to CPU
                    batch_norm = torch.norm(cur_input, p=2, dim=1) #.float().cpu() # Move to CPU
                    if '70B' not in args.model_name_or_path:
                        accum_mwp_norm_attn[layer_idx] += batch_norm
                    else:
                        accum_mwp_norm_attn[layer_idx] += batch_norm #.to('cpu')
                    # input_tensors_attn[layer_idx] = None
                
                if input_tensors_mlp[layer_idx] is not None:
                    cur_input = input_tensors_mlp[layer_idx] #.detach()
                    # batch_norm = torch.norm(cur_input, p=2, dim=1).sum(dim=0).float().cpu()
                    batch_norm = torch.norm(cur_input, p=2, dim=1) #.float().cpu()
                    if '70B' not in args.model_name_or_path:
                        accum_mwp_norm_mlp[layer_idx] += batch_norm
                    else:
                        accum_mwp_norm_mlp[layer_idx] += batch_norm #.to('cpu')
                    # input_tensors_mlp[layer_idx] = None

            if 'llama-3.1' in args.model_name_or_path.lower():
                # --- 4. VARIANCE ACCUMULATION (Replace your MWP block) --- (differentiating between tokens)
                for layer_idx in range(num_layers):
                    
                    # --- ATTENTION ---
                    if input_tensors_attn[layer_idx] is not None:
                        # Shape comes from hook: (Hidden_Dim, Batch * SeqLen)
                        cur_input = input_tensors_attn[layer_idx].to(device) #.float() # Ensure float32 for stability
                        
                        # 1. Initialize stats if new
                        if layer_idx not in accum_sum_attn:
                            feature_dim = cur_input.shape[0]
                            accum_sum_attn[layer_idx] = torch.zeros(feature_dim, device=device)
                            accum_sq_sum_attn[layer_idx] = torch.zeros(feature_dim, device=device)
                        
                        # 2. Accumulate Sums along Token Dimension (dim=1)
                        # We collapse (Hidden, Tokens) -> (Hidden,)
                        accum_sum_attn[layer_idx] += cur_input.sum(dim=1)
                        accum_sq_sum_attn[layer_idx] += (cur_input ** 2).sum(dim=1)
                        
                        # Track count only once (assuming all layers see same token count)
                        if layer_idx == 0:
                            accum_tokens_count += cur_input.shape[1]

                        # input_tensors_attn[layer_idx] = None # Clear VRAM

                    # --- MLP (Repeat logic) ---
                    if input_tensors_mlp[layer_idx] is not None:
                        cur_input = input_tensors_mlp[layer_idx].to(device) #.float()
                        if layer_idx not in accum_sum_mlp:
                            feature_dim = cur_input.shape[0]
                            accum_sum_mlp[layer_idx] = torch.zeros(feature_dim, device=device)
                            accum_sq_sum_mlp[layer_idx] = torch.zeros(feature_dim, device=device)
                        
                        accum_sum_mlp[layer_idx] += cur_input.sum(dim=1)
                        accum_sq_sum_mlp[layer_idx] += (cur_input ** 2).sum(dim=1)
                        # input_tensors_mlp[layer_idx] = None

    for layer_idx in range(num_layers):
        input_tensors_attn[layer_idx] = None
        input_tensors_mlp[layer_idx] = None

    if 'llama-3.1' in args.model_name_or_path.lower():
        # --- FINAL VARIANCE CALCULATION ---
        # Var(X) = E[X^2] - (E[X])^2
        final_std_attn = {}
        
        for layer_idx, sum_x in accum_sum_attn.items():
            mean_x = sum_x / accum_tokens_count
            mean_sq_x = accum_sq_sum_attn[layer_idx] / accum_tokens_count
            
            # Clamp to avoid negative zero errorsdevice
            variance = torch.clamp(mean_sq_x - (mean_x ** 2), min=1e-8)
            final_std_attn[layer_idx] = torch.sqrt(variance) #.cpu() # Return to CPU

        # (Repeat for MLP)
        final_std_mlp = {}
        for layer_idx, sum_x in accum_sum_mlp.items():
            mean_x = sum_x / accum_tokens_count
            mean_sq_x = accum_sq_sum_mlp[layer_idx] / accum_tokens_count
            variance = torch.clamp(mean_sq_x - (mean_x ** 2), min=1e-8)
            final_std_mlp[layer_idx] = torch.sqrt(variance) #.cpu()

    # --- Cleanup ---
    for hook in hooks:
        hook.remove()
    model.zero_grad()

    del input_tensors_attn
    del input_tensors_mlp

    if 'llama-3.1' not in args.model_name_or_path.lower(): 
        return accum_grad_attn, accum_grad_mlp, None, None, total_loss / total_batches, accum_mwp_norm_attn, accum_mwp_norm_mlp, accum_grad_attn_o, accum_grad_mlp_down
    else:
        return accum_grad_attn, accum_grad_mlp, None, None, total_loss / total_batches, final_std_attn, final_std_mlp, accum_grad_attn_o, accum_grad_mlp_down

def calculate_influence_w_l2normdist(args, model, dataloader, active_layers_attn, active_layers_mlp, w, 
                                     base_grad_vecs_attn, base_grad_vecs_mlp, base_grad_attn_fisher, base_grad_mlp_fisher, 
                                     batch_base_loss, compute_loss_context_manager, compute_loss, accelerator, 
                                     fisher_mask=None, attn_masks=None, mlp_masks=None, 
                                     base_attn_tensors=None, base_mlp_tensors=None, base_attn_op_tensors=None, base_mlp_op_tensors=None):

    # --- Setup ---
    l_attn = int(active_layers_attn.sum().item())
    l_mlp = int(active_layers_mlp.sum().item())

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    
    max_layer = model_config.num_hidden_layers

    model.eval() 
    model.zero_grad()
    
    # Identify active indices
    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]
    
    # Access layers wrapper
    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
    
    elif hasattr(model.model.model, 'language_model'):
        lm_model = model.model.model.language_model
    
        if hasattr(lm_model, 'layers'):
            model_layers = lm_model.layers

    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP)
    }

    # -------------------------------------------------------------------------
    # OPTIMIZATION 1: Pre-fetch parameter references
    # We identify which parameters belong to which layer/type ONCE before the loop.
    # -------------------------------------------------------------------------
    params_map_attn = {} # {layer_idx: [list of param objects]}
    params_map_mlp = {}  # {layer_idx: [list of param objects]}

    # Collect Attn Params
    for layer_idx in active_layer_attn_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_attn_indices:
            continue
        attn_module = getattr(model_layers[layer_idx], "attention", 
              getattr(model_layers[layer_idx], "self_attn", None))
        # Filter for self_attn parameters that require grad
        params = [p for n, p in attn_module.named_parameters() if p.requires_grad]
        if params:
            params_map_attn[layer_idx] = params

    # Collect MLP Params
    for layer_idx in active_layer_mlp_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_mlp_indices:
            continue
        # Filter for MLP parameters that require grad
        params = [p for n, p in model_layers[layer_idx].mlp.named_parameters() if p.requires_grad]
        if params:
            params_map_mlp[layer_idx] = params

    # -------------------------------------------------------------------------
    # OPTIMIZATION 2: GPU-resident Shadow Accumulators
    # Map parameter object ID -> Accumulation Tensor (on GPU)
    # -------------------------------------------------------------------------
    param_accumulators = {} 

    # --- Masks & Pruning Logic (Preserved) ---
    if not args.layer_pruning:
        sparsity = get_attn_mlp_sparsity(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
    else:
        sparsity = None

    if args.number_of_params_thresh is not None:
        model_size = count_active_parameters(model, active_layers_attn, active_layers_mlp, fisher_mask)[1]

        number_of_params_thresh = list(eval(args.number_of_params_thresh))
        if not number_of_params_thresh[0] < model_size < number_of_params_thresh[1]:
            print(f'Skipping model ${active_layers_attn.sum().item(), w} with model size {model_size} outside threshold {number_of_params_thresh}')
            return None, None, None, None, None, None, None

        if not args.layer_pruning:
            set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
        print(active_layers_attn.sum().item(), active_layers_mlp.sum().item(), w, model_size)
    else:
        set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)
    
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    total_runtime_gpu = 0.0

    # --- ONLINE LOOP ---
    with compute_loss_context_manager():
        for i, test_inputs in enumerate(dataloader):
            total_batches += 1

            # Assuming batch size is the first dimension of the first tensor in inputs
            batch_size = next(iter(test_inputs.values())).size(0) if isinstance(test_inputs, dict) else test_inputs[0].size(0)
            total_samples += batch_size

            # 1. SYNCHRONIZE & START GPU TIMER
            # This ensures we measure the time the GPU actually spends on the math
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize() # Clear existing queue
            start_event.record()
            
            # Forward & Backward
            rand_loss, _ = compute_loss(model, test_inputs, return_runtime=True)
            rand_loss.backward()

            # 3. END GPU TIMER
            end_event.record()
            torch.cuda.synchronize() # Wait for kernels to finish

            total_loss += rand_loss.item()


            # elapsed_time returns milliseconds
            batch_time_ms = start_event.elapsed_time(end_event)
            total_runtime_gpu += (batch_time_ms / 1000.0) # Convert to seconds
            # -----------------------------------------------------------------
            # OPTIMIZATION 3: Accumulate gradients in-place without flattening
            # -----------------------------------------------------------------
            with torch.no_grad():
                # Helper to accumulate specific params
                def accum_layer_grads(param_map):
                    for layer_idx, params in param_map.items():
                        for p in params:
                            if p.grad is not None:
                                if p not in param_accumulators:
                                    # Initialize accumulator on same device as p
                                    param_accumulators[p] = p.grad.detach().clone()
                                else:
                                    param_accumulators[p] += p.grad
                
                accum_layer_grads(params_map_attn)
                accum_layer_grads(params_map_mlp)

            model.zero_grad() 

    # --- Performance Metrics Calculation ---
    avg_loss = total_loss / total_batches
    avg_latency = total_runtime_gpu / total_batches  # Avg time per batch
    throughput = total_samples / total_runtime_gpu   # Samples per second

    # --- FINAL CALCULATION ---
    # Now we flatten and calculate scores *once* at the end
    
    def calculate_final_scores(param_map, base_grads, sparsity_dict, module_name):
        inf_accum = 0.0
        ssim_accum = 0.0
        tracin_dict = {}

        for layer_idx, params in param_map.items():
            
            # Reconstruct the summed gradient vector for this layer
            # We use the accumulated tensors in param_accumulators
            accumulated_grads = []
            for p in params:
                if p in param_accumulators:
                    accumulated_grads.append(param_accumulators[p])
                else:
                    # Handle case where no grad was ever observed (rare/impossible if active)
                    accumulated_grads.append(torch.zeros_like(p))
            
            # Flatten NOW (only once per layer)
            # Move to CPU here if base_grads are on CPU to save GPU memory, 
            # or move base_grads to GPU if you prefer speed. 
            # Assuming base_grads are CPU for safety:
            grad_sum_vec = parameters_to_vector(accumulated_grads).to(base_grads[layer_idx].device)
            
            # Average
            mean_grad = grad_sum_vec / total_batches
            grad_base = base_grads[layer_idx] # Already on correct device?
            
            # Correlation
            # Ensure standarlization is available in scope or imported
            corr = torch.dot(standarlization(mean_grad), standarlization(grad_base.to(mean_grad))) / mean_grad.shape[0]
            tracin_dict[layer_idx] = corr.cpu()
            
            # if sparsity_dict:
            #     ratio = sparsity_dict[module_name][layer_idx]
            #     inf_accum += ratio * corr.cpu() #.float()
            # else:
            inf_accum += corr.cpu() #.float()

        return inf_accum, ssim_accum, tracin_dict

    # Calculate for Attn and MLP
    inf_attn, ssim_attn, tracin_attn = calculate_final_scores(
        params_map_attn, base_grad_vecs_attn, sparsity if 'sparsity' in locals() else None, 'attn'
    )
    
    inf_mlp, ssim_mlp, tracin_mlp = calculate_final_scores(
        params_map_mlp, base_grad_vecs_mlp, sparsity if 'sparsity' in locals() else None, 'mlp'
    )

    term_attn = inf_attn 
    term_mlp = inf_mlp 
    final_score = term_attn + term_mlp

    # Cleanup
    if not args.layer_pruning:
        reset_width_mask(args, model, fisher_mask)
    model.zero_grad()
    
    # Clear large accumulators
    del param_accumulators
    torch.cuda.empty_cache()

    print(f'Final Score: {final_score.item():.4f} (Attn: {term_attn:.4f} + MLP: {term_mlp:.4f}), Loss: {avg_loss:.4f}')
    # exit()
    return final_score, fisher_mask, avg_loss, tracin_attn, tracin_mlp, avg_latency, throughput

def calculate_influence_w_cosine(args, model, dataloader, active_layers_attn, active_layers_mlp, w, 
                                     base_grad_vecs_attn, base_grad_vecs_mlp, base_grad_attn_fisher, base_grad_mlp_fisher, 
                                     batch_base_loss, compute_loss_context_manager, compute_loss, accelerator, 
                                     fisher_mask=None, attn_masks=None, mlp_masks=None, 
                                     base_attn_tensors=None, base_mlp_tensors=None, base_attn_op_tensors=None, base_mlp_op_tensors=None):

    # --- Setup ---
    l_attn = int(active_layers_attn.sum().item())
    l_mlp = int(active_layers_mlp.sum().item())

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    
    max_layer = model_config.num_hidden_layers

    model.eval() 
    model.zero_grad()
    
    # Identify active indices
    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]
    
    # Access layers wrapper
    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
    
    elif hasattr(model.model.model, 'language_model'):
        lm_model = model.model.model.language_model
    
        if hasattr(lm_model, 'layers'):
            model_layers = lm_model.layers

    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP)
    }

    # -------------------------------------------------------------------------
    # OPTIMIZATION 1: Pre-fetch parameter references
    # We identify which parameters belong to which layer/type ONCE before the loop.
    # -------------------------------------------------------------------------
    params_map_attn = {} # {layer_idx: [list of param objects]}
    params_map_mlp = {}  # {layer_idx: [list of param objects]}

    # Collect Attn Params
    for layer_idx in active_layer_attn_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_attn_indices:
            continue
        attn_module = getattr(model_layers[layer_idx], "attention", 
              getattr(model_layers[layer_idx], "self_attn", None))
        # Filter for self_attn parameters that require grad
        params = [p for n, p in attn_module.named_parameters() if p.requires_grad]
        if params:
            params_map_attn[layer_idx] = params

    # Collect MLP Params
    for layer_idx in active_layer_mlp_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_mlp_indices:
            continue
        # Filter for MLP parameters that require grad
        params = [p for n, p in model_layers[layer_idx].mlp.named_parameters() if p.requires_grad]
        if params:
            params_map_mlp[layer_idx] = params

    # -------------------------------------------------------------------------
    # OPTIMIZATION 2: GPU-resident Shadow Accumulators
    # Map parameter object ID -> Accumulation Tensor (on GPU)
    # -------------------------------------------------------------------------
    param_accumulators = {} 

    # --- Masks & Pruning Logic (Preserved) ---
    if not args.layer_pruning:
        sparsity = get_attn_mlp_sparsity(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
    else:
        sparsity = None

    if args.number_of_params_thresh is not None:
        model_size = count_active_parameters(model, active_layers_attn, active_layers_mlp, fisher_mask)[1]

        number_of_params_thresh = list(eval(args.number_of_params_thresh))
        if not number_of_params_thresh[0] < model_size < number_of_params_thresh[1]:
            print(f'Skipping model ${active_layers_attn.sum().item(), w} with model size {model_size} outside threshold {number_of_params_thresh}')
            return None, None, None, None, None, None, None

        if not args.layer_pruning:
            set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
        print(active_layers_attn.sum().item(), active_layers_mlp.sum().item(), w, model_size)
    else:
        set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)
    
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    total_runtime_gpu = 0.0

    # --- ONLINE LOOP ---
    with compute_loss_context_manager():
        for i, test_inputs in enumerate(dataloader):
            total_batches += 1

            # Assuming batch size is the first dimension of the first tensor in inputs
            batch_size = next(iter(test_inputs.values())).size(0) if isinstance(test_inputs, dict) else test_inputs[0].size(0)
            total_samples += batch_size

            # 1. SYNCHRONIZE & START GPU TIMER
            # This ensures we measure the time the GPU actually spends on the math
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize() # Clear existing queue
            start_event.record()
            
            # Forward & Backward
            rand_loss, _ = compute_loss(model, test_inputs, return_runtime=True)
            rand_loss.backward()

            # 3. END GPU TIMER
            end_event.record()
            torch.cuda.synchronize() # Wait for kernels to finish

            total_loss += rand_loss.item()


            # elapsed_time returns milliseconds
            batch_time_ms = start_event.elapsed_time(end_event)
            total_runtime_gpu += (batch_time_ms / 1000.0) # Convert to seconds
            # -----------------------------------------------------------------
            # OPTIMIZATION 3: Accumulate gradients in-place without flattening
            # -----------------------------------------------------------------
            with torch.no_grad():
                # Helper to accumulate specific params
                def accum_layer_grads(param_map):
                    for layer_idx, params in param_map.items():
                        for p in params:
                            if p.grad is not None:
                                if p not in param_accumulators:
                                    # Initialize accumulator on same device as p
                                    param_accumulators[p] = p.grad.detach().clone()
                                else:
                                    param_accumulators[p] += p.grad
                
                accum_layer_grads(params_map_attn)
                accum_layer_grads(params_map_mlp)

            model.zero_grad() 

    # --- Performance Metrics Calculation ---
    avg_loss = total_loss / total_batches
    avg_latency = total_runtime_gpu / total_batches  # Avg time per batch
    throughput = total_samples / total_runtime_gpu   # Samples per second

    # --- FINAL CALCULATION ---
    # Now we flatten and calculate scores *once* at the end
    
    def calculate_final_scores(param_map, base_grads, sparsity_dict, module_name):
        inf_accum = 0.0
        ssim_accum = 0.0
        tracin_dict = {}

        for layer_idx, params in param_map.items():
            
            # Reconstruct the summed gradient vector for this layer
            # We use the accumulated tensors in param_accumulators
            accumulated_grads = []
            for p in params:
                if p in param_accumulators:
                    accumulated_grads.append(param_accumulators[p])
                else:
                    # Handle case where no grad was ever observed (rare/impossible if active)
                    accumulated_grads.append(torch.zeros_like(p))
            
            # Flatten NOW (only once per layer)
            # Move to CPU here if base_grads are on CPU to save GPU memory, 
            # or move base_grads to GPU if you prefer speed. 
            # Assuming base_grads are CPU for safety:
            grad_sum_vec = parameters_to_vector(accumulated_grads).to(base_grads[layer_idx].device)
            
            # Average
            mean_grad = grad_sum_vec / total_batches
            grad_base = base_grads[layer_idx] # Already on correct device?
            
            # Correlation
            # Ensure standarlization is available in scope or imported
            # --- COSINE SIMILARITY CALCULATION ---
            # Cosine Sim = (A · B) / (||A|| * ||B||)
            # F.cosine_similarity expects a dimension, so we unsqueeze
            cosine_sim = F.cosine_similarity(mean_grad.unsqueeze(0), grad_base.unsqueeze(0))
            
            # Extract scalar value
            score = cosine_sim.item()
            tracin_dict[layer_idx] = score
            
            inf_accum += score

        return inf_accum, ssim_accum, tracin_dict

    # Calculate for Attn and MLP
    inf_attn, ssim_attn, tracin_attn = calculate_final_scores(
        params_map_attn, base_grad_vecs_attn, sparsity if 'sparsity' in locals() else None, 'attn'
    )
    
    inf_mlp, ssim_mlp, tracin_mlp = calculate_final_scores(
        params_map_mlp, base_grad_vecs_mlp, sparsity if 'sparsity' in locals() else None, 'mlp'
    )

    term_attn = inf_attn 
    term_mlp = inf_mlp 
    final_score = term_attn + term_mlp

    # Cleanup
    if not args.layer_pruning:
        reset_width_mask(args, model, fisher_mask)
    model.zero_grad()
    
    # Clear large accumulators
    del param_accumulators
    torch.cuda.empty_cache()

    print(f'Final Score: {final_score:.4f} (Attn: {term_attn:.4f} + MLP: {term_mlp:.4f}), Loss: {avg_loss:.4f}')
    # exit()
    return final_score, fisher_mask, avg_loss, tracin_attn, tracin_mlp, avg_latency, throughput

def calculate_influence_w_dot_prod(args, model, dataloader, active_layers_attn, active_layers_mlp, w, 
                                     base_grad_vecs_attn, base_grad_vecs_mlp, base_grad_attn_fisher, base_grad_mlp_fisher, 
                                     batch_base_loss, compute_loss_context_manager, compute_loss, accelerator, 
                                     fisher_mask=None, attn_masks=None, mlp_masks=None, 
                                     base_attn_tensors=None, base_mlp_tensors=None, base_attn_op_tensors=None, base_mlp_op_tensors=None):

    # --- Setup ---
    l_attn = int(active_layers_attn.sum().item())
    l_mlp = int(active_layers_mlp.sum().item())

    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    
    max_layer = model_config.num_hidden_layers

    model.eval() 
    model.zero_grad()
    
    # Identify active indices
    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]
    
    # Access layers wrapper
    if hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    
    elif hasattr(model.model.model, 'layers'):
        model_layers = model.model.model.layers
    
    elif hasattr(model.model.model, 'language_model'):
        lm_model = model.model.model.language_model
    
        if hasattr(lm_model, 'layers'):
            model_layers = lm_model.layers

    skip_attn_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(model_layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP)
    }

    # -------------------------------------------------------------------------
    # OPTIMIZATION 1: Pre-fetch parameter references
    # We identify which parameters belong to which layer/type ONCE before the loop.
    # -------------------------------------------------------------------------
    params_map_attn = {} # {layer_idx: [list of param objects]}
    params_map_mlp = {}  # {layer_idx: [list of param objects]}

    # Collect Attn Params
    for layer_idx in active_layer_attn_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_attn_indices:
            continue
        attn_module = getattr(model_layers[layer_idx], "attention", 
              getattr(model_layers[layer_idx], "self_attn", None))
        # Filter for self_attn parameters that require grad
        params = [p for n, p in attn_module.named_parameters() if p.requires_grad]
        if params:
            params_map_attn[layer_idx] = params

    # Collect MLP Params
    for layer_idx in active_layer_mlp_idx:
        layer_idx = layer_idx.item()
        if layer_idx in skip_mlp_indices:
            continue
        # Filter for MLP parameters that require grad
        params = [p for n, p in model_layers[layer_idx].mlp.named_parameters() if p.requires_grad]
        if params:
            params_map_mlp[layer_idx] = params

    # -------------------------------------------------------------------------
    # OPTIMIZATION 2: GPU-resident Shadow Accumulators
    # Map parameter object ID -> Accumulation Tensor (on GPU)
    # -------------------------------------------------------------------------
    param_accumulators = {} 

    # --- Masks & Pruning Logic (Preserved) ---
    if not args.layer_pruning:
        sparsity = get_attn_mlp_sparsity(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
    else:
        sparsity = None

    if args.number_of_params_thresh is not None:
        model_size = count_active_parameters(model, active_layers_attn, active_layers_mlp, fisher_mask)[1]

        number_of_params_thresh = list(eval(args.number_of_params_thresh))
        if not number_of_params_thresh[0] < model_size < number_of_params_thresh[1]:
            print(f'Skipping model ${active_layers_attn.sum().item(), w} with model size {model_size} outside threshold {number_of_params_thresh}')
            return None, None, None, None, None, None, None

        if not args.layer_pruning:
            set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
        print(active_layers_attn.sum().item(), active_layers_mlp.sum().item(), w, model_size)
    else:
        set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)
    
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    total_runtime_gpu = 0.0

    # --- ONLINE LOOP ---
    with compute_loss_context_manager():
        for i, test_inputs in enumerate(dataloader):
            total_batches += 1

            # Assuming batch size is the first dimension of the first tensor in inputs
            batch_size = next(iter(test_inputs.values())).size(0) if isinstance(test_inputs, dict) else test_inputs[0].size(0)
            total_samples += batch_size

            # 1. SYNCHRONIZE & START GPU TIMER
            # This ensures we measure the time the GPU actually spends on the math
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize() # Clear existing queue
            start_event.record()
            
            # Forward & Backward
            rand_loss, _ = compute_loss(model, test_inputs, return_runtime=True)
            rand_loss.backward()

            # 3. END GPU TIMER
            end_event.record()
            torch.cuda.synchronize() # Wait for kernels to finish

            total_loss += rand_loss.item()


            # elapsed_time returns milliseconds
            batch_time_ms = start_event.elapsed_time(end_event)
            total_runtime_gpu += (batch_time_ms / 1000.0) # Convert to seconds
            # -----------------------------------------------------------------
            # OPTIMIZATION 3: Accumulate gradients in-place without flattening
            # -----------------------------------------------------------------
            with torch.no_grad():
                # Helper to accumulate specific params
                def accum_layer_grads(param_map):
                    for layer_idx, params in param_map.items():
                        for p in params:
                            if p.grad is not None:
                                if p not in param_accumulators:
                                    # Initialize accumulator on same device as p
                                    param_accumulators[p] = p.grad.detach().clone()
                                else:
                                    param_accumulators[p] += p.grad
                
                accum_layer_grads(params_map_attn)
                accum_layer_grads(params_map_mlp)

            model.zero_grad() 

    # --- Performance Metrics Calculation ---
    avg_loss = total_loss / total_batches
    avg_latency = total_runtime_gpu / total_batches  # Avg time per batch
    throughput = total_samples / total_runtime_gpu   # Samples per second

    # --- FINAL CALCULATION ---
    # Now we flatten and calculate scores *once* at the end
    
    def calculate_final_scores(param_map, base_grads, sparsity_dict, module_name):
        inf_accum = 0.0
        ssim_accum = 0.0
        tracin_dict = {}

        for layer_idx, params in param_map.items():
            
            # Reconstruct the summed gradient vector for this layer
            # We use the accumulated tensors in param_accumulators
            accumulated_grads = []
            for p in params:
                if p in param_accumulators:
                    accumulated_grads.append(param_accumulators[p])
                else:
                    # Handle case where no grad was ever observed (rare/impossible if active)
                    accumulated_grads.append(torch.zeros_like(p))
            
            # Flatten NOW (only once per layer)
            # Move to CPU here if base_grads are on CPU to save GPU memory, 
            # or move base_grads to GPU if you prefer speed. 
            # Assuming base_grads are CPU for safety:
            grad_sum_vec = parameters_to_vector(accumulated_grads).to(base_grads[layer_idx].device)
            
            # Average
            mean_grad = grad_sum_vec / total_batches
            grad_base = base_grads[layer_idx] # Already on correct device?
            
            # Correlation
            # Ensure standarlization is available in scope or imported
            # --- COSINE SIMILARITY CALCULATION ---
            # Cosine Sim = (A · B) / (||A|| * ||B||)
            # F.cosine_similarity expects a dimension, so we unsqueeze
            tracin = torch.dot(mean_grad, grad_base)
            
            # Extract scalar value
            score = tracin.item()
            tracin_dict[layer_idx] = score
            
            inf_accum += score

        return inf_accum, ssim_accum, tracin_dict

    # Calculate for Attn and MLP
    inf_attn, ssim_attn, tracin_attn = calculate_final_scores(
        params_map_attn, base_grad_vecs_attn, sparsity if 'sparsity' in locals() else None, 'attn'
    )
    
    inf_mlp, ssim_mlp, tracin_mlp = calculate_final_scores(
        params_map_mlp, base_grad_vecs_mlp, sparsity if 'sparsity' in locals() else None, 'mlp'
    )

    term_attn = inf_attn 
    term_mlp = inf_mlp 
    final_score = term_attn + term_mlp

    # Cleanup
    if not args.layer_pruning:
        reset_width_mask(args, model, fisher_mask)
    model.zero_grad()
    
    # Clear large accumulators
    del param_accumulators
    torch.cuda.empty_cache()

    print(f'Final Score: {final_score:.4f} (Attn: {term_attn:.4f} + MLP: {term_mlp:.4f}), Loss: {avg_loss:.4f}')
    # exit()
    return final_score, fisher_mask, avg_loss, tracin_attn, tracin_mlp, avg_latency, throughput

###ZICO###
def compute_zico(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    
    def calculate_zico(layer_grads_attn, layer_grads_mlp, pruning_ratios=None):
    
        nsr_mean_sum_abs = 0
        nsr_mean_avg_abs = 0
        for layer_idx, grad_info in layer_grads_attn.items():
            grad_dict = torch.stack(grad_info)

            nsr_std = torch.std(grad_dict, dim=0).float()
            nonzero_idx = torch.nonzero(nsr_std).squeeze() #[0]
            nsr_mean_abs = torch.mean(torch.abs(grad_dict), dim=0).float()
            tmpsum = torch.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])

            if tmpsum == 0:
                pass
            else:
                nsr_mean_sum_abs += torch.log(tmpsum)
                # nsr_mean_sum_abs = torch.log(tmpsum) - 10*(1-pruning_ratios['attn'][layer_idx])
                # nsr_mean_avg_abs += torch.log(torch.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))

        for layer_idx, grad_info in layer_grads_mlp.items():
            grad_dict = torch.stack(grad_info)

            nsr_std = torch.std(grad_dict, dim=0).float()
            nonzero_idx = torch.nonzero(nsr_std).squeeze() #[0]
            nsr_mean_abs = torch.mean(torch.abs(grad_dict), dim=0).float()
            tmpsum = torch.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])

            if tmpsum == 0:
                pass
            else:
                nsr_mean_sum_abs += torch.log(tmpsum)
                # nsr_mean_sum_abs = torch.log(tmpsum) - 10*(1-pruning_ratios['mlp'][layer_idx])
                # nsr_mean_avg_abs += torch.log(torch.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))  
        
        return nsr_mean_sum_abs.item()

    eval_iterator = iter(dataloader)

    influence_score = 0

    l_attn, l_mlp, max_layer = int(active_layers_attn.sum().item()), int(active_layers_mlp.sum().item()), model.config.num_hidden_layers


    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]

    attn_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_attn_idx}
    mlp_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_mlp_idx}
    
    layer_tracin_attn = {layer_idx.item(): 0 for layer_idx in active_layer_attn_idx}
    layer_tracin_mlp = {layer_idx.item(): 0 for layer_idx in active_layer_mlp_idx}

    is_full_model = (l_attn == max_layer and l_mlp == max_layer and w == 1)

    sparsity = get_attn_mlp_sparsity(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)


    if args.number_of_params_thresh is not None:
        model_size = count_active_parameters(model, active_layers_attn, active_layers_mlp, fisher_mask)[1]

        number_of_params_thresh = list(eval(args.number_of_params_thresh))
        if not number_of_params_thresh[0] < model_size < number_of_params_thresh[1]:
            return None, None, None
        
        set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)

    layer_grads_attn = {layer_idx.item(): list() for layer_idx in active_layer_attn_idx}
    layer_grads_mlp = {layer_idx.item(): list() for layer_idx in active_layer_mlp_idx}
    
    total_time_s = 0
    total_batches = 0
    with compute_loss_context_manager():                    
        for i, test_inputs in enumerate(eval_iterator):
            # if not (l_attn==max_layer and l_mlp==max_layer):
            model.zero_grad()
            
            rand_loss, b_total_time_s = compute_loss(model, test_inputs, return_runtime=True)
            total_time_s += b_total_time_s
            total_batches += 1
            
            accelerator.backward(rand_loss, retain_graph=True)
            W_metric = []
            for layer_idx in active_layer_attn_idx:
                layer_idx=layer_idx.item()
                # if not (l_attn==max_layer and l_mlp==max_layer):
                if not hasattr(model.model, 'layers'):
                    layer_grads = parameters_to_vector([p.grad for n, p in model.model.model.layers[layer_idx].named_parameters() if p.grad is not None and 'self_attn' in n])
                else:
                    layer_grads = parameters_to_vector([p.grad for n, p in model.model.layers[layer_idx].named_parameters() if p.grad is not None and 'self_attn' in n])
                
                if layer_grads.numel() > 0:
                    layer_grads_attn[layer_idx].append(layer_grads)

            for layer_idx in active_layer_mlp_idx:
                layer_idx=layer_idx.item()
                # if not (l_attn==max_layer and l_mlp==max_layer):
                if not hasattr(model.model, 'layers'):
                    layer_grads = parameters_to_vector([p.grad for n, p in model.model.model.layers[layer_idx].named_parameters() if p.grad is not None and 'mlp' in n])
                else:
                    layer_grads = parameters_to_vector([p.grad for n, p in model.model.layers[layer_idx].named_parameters() if p.grad is not None and 'mlp' in n])
                
                if layer_grads.numel() > 0:
                    layer_grads_mlp[layer_idx].append(layer_grads)
            
        zico_score = calculate_zico(layer_grads_attn, layer_grads_mlp)
    reset_width_mask(args, model, fisher_mask)
    return zico_score, fisher_mask, rand_loss

###MeCo###
def compute_meco(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):    
    eval_iterator = iter(dataloader)

    influence_score = 0

    l_attn, l_mlp, max_layer = int(active_layers_attn.sum().item()), int(active_layers_mlp.sum().item()), model.config.num_hidden_layers


    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]

    attn_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_attn_idx}
    mlp_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_mlp_idx}
    
    layer_tracin_attn = {layer_idx.item(): 0 for layer_idx in active_layer_attn_idx}
    layer_tracin_mlp = {layer_idx.item(): 0 for layer_idx in active_layer_mlp_idx}

    fisher_mask, attn_masks, mlp_masks = get_attn_mlp_masks(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)


    # set_width_mask(args, model, fisher_mask)
    set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)

    result_list = []
    def forward_hook(module, data_input, data_output):
        
        # og_dtype = data_output.dtype
        # data_output = data_output.to(torch.float32)

        data_output = data_output
        if isinstance(data_output, tuple):
            fea = data_output[0] # LlamaAttention returns tuple
        else:
            fea = data_output    # LlamaMLP returns Tensor
        fea = fea.reshape(fea.shape[0], -1)
        n = fea.shape[0]
        corr = torch.corrcoef(fea).to(torch.float32)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)

        # data_output = data_output.to(og_dtype)

    hook_handles = [] # Good practice: store handles to remove hooks later
    for name, module in model.named_modules():
        # Check if the module is an instance of ANY type in the tuple
        # if isinstance(module, TARGET_LAYERS_TUPLE):
        if isinstance(module, (LlamaAttention, LlamaMLP)):
            # print(f"✅ Hooking: {name} ({module.__class__.__name__})")
            # Register the hook and store the handle
            handle = module.register_forward_hook(forward_hook)
            hook_handles.append(handle)
    
    total_time_s = 0
    total_batches = 0
    with compute_loss_context_manager():                    
        for i, test_inputs in enumerate(eval_iterator):
            rand_loss, b_total_time_s = compute_loss(model, test_inputs, return_runtime=True)
        
        results = torch.stack(result_list)
        results = results[torch.logical_not(torch.isnan(results))]
        v = torch.sum(results)
        result_list.clear()

        meco_score = v.item()

    for handle in hook_handles:
        handle.remove()

    reset_width_mask(args, model, fisher_mask)

    return meco_score, fisher_mask, rand_loss

###NASWOT###
def compute_naswot(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            K = K.float()
            K2 = K2.float()
            model.K = model.K + K + K2
        except Exception as err:
            print('---- error on model : ')
            print(model)
            raise err


    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    def logdet(K):
        s, ld = np.linalg.slogdet(K)
        return ld

    eval_iterator = iter(dataloader)

    influence_score = 0

    l_attn, l_mlp, max_layer = int(active_layers_attn.sum().item()), int(active_layers_mlp.sum().item()), model.config.num_hidden_layers


    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]

    attn_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_attn_idx}
    mlp_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_mlp_idx}
    
    layer_tracin_attn = {layer_idx.item(): 0 for layer_idx in active_layer_attn_idx}
    layer_tracin_mlp = {layer_idx.item(): 0 for layer_idx in active_layer_mlp_idx}

    is_full_model = (l_attn == max_layer and l_mlp == max_layer and w == 1)

    # set_width_mask(args, model, fisher_mask)
    if not is_full_model:
        set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)

    for name, module in model.named_modules():
        # if 'ReLU' in str(type(module)):
        if isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            # module.register_backward_hook(counting_backward_hook)

    layer_grads_attn = {layer_idx.item(): list() for layer_idx in active_layer_attn_idx}
    layer_grads_mlp = {layer_idx.item(): list() for layer_idx in active_layer_mlp_idx}
    
    total_time_s = 0
    total_batches = 0
    with compute_loss_context_manager():                    
        for i, test_inputs in enumerate(eval_iterator):
            if i==0:
                model.K = torch.zeros((test_inputs['input_ids'].shape[0], test_inputs['input_ids'].shape[0])).to(test_inputs['input_ids'].device)
                
            # if not (l_attn==max_layer and l_mlp==max_layer):
            model.zero_grad()
            
            rand_loss, b_total_time_s = compute_loss(model, test_inputs, return_runtime=True)
            total_time_s += b_total_time_s
            total_batches += 1
        
        naswot_score = logdet(model.K.data.cpu().numpy())
    reset_width_mask(args, model, fisher_mask)

    return naswot_score, fisher_mask, rand_loss


def compute_gradnorm(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    eval_iterator = iter(dataloader)

    influence_score = 0

    l_attn, l_mlp, max_layer = int(active_layers_attn.sum().item()), int(active_layers_mlp.sum().item()), model.config.num_hidden_layers


    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]

    attn_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_attn_idx}
    mlp_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_mlp_idx}
    
    layer_tracin_attn = {layer_idx.item(): 0 for layer_idx in active_layer_attn_idx}
    layer_tracin_mlp = {layer_idx.item(): 0 for layer_idx in active_layer_mlp_idx}

    # set_width_mask(args, model, fisher_mask)
    set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    model.set_active_layers(active_layers_attn, active_layers_mlp)

    layer_grads_attn = {layer_idx.item(): list() for layer_idx in active_layer_attn_idx}
    layer_grads_mlp = {layer_idx.item(): list() for layer_idx in active_layer_mlp_idx}
    
    total_time_s = 0
    total_batches = 0
    gradnorm_score = 0
    with compute_loss_context_manager():                    
        for i, test_inputs in enumerate(eval_iterator):
            model.zero_grad()
            
            rand_loss, b_total_time_s = compute_loss(model, test_inputs, return_runtime=True)
            total_time_s += b_total_time_s
            total_batches += 1

            accelerator.backward(rand_loss, retain_graph=True)

            if not hasattr(model.model, 'layers'):
                layer_grads = parameters_to_vector([p.grad for n, p in model.model.model.named_parameters() if p.grad is not None])
            else:
                layer_grads = parameters_to_vector([p.grad for n, p in model.model.named_parameters() if p.grad is not None])
        
            gradnorm_score += float(torch.norm(layer_grads).sum().item())
    
    reset_width_mask(args, model, fisher_mask)

    return gradnorm_score, fisher_mask, rand_loss

def compute_lpzero(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    
    # 1. Helper: Find the correct mask key in the dictionary
    # Because PEFT/LoRA nesting can vary (base_model.model.model vs base_model.model),
    # we search for the key that ends with our target suffix.
    def get_mask_by_suffix(target_suffix, layer_idx, fisher_mask):
        if fisher_mask is None: return None
        
        # Construct the critical part of the name, e.g., "layers.0.mlp.up_proj"
        # We search specifically for the 'base_layer' if that's how your keys are named
        search_pattern = f"layers.{layer_idx}.{target_suffix}"
        
        for key, mask in fisher_mask.items():
            if search_pattern in key:
                # Optional: Ensure we aren't matching 'up_proj' when looking for 'gate_proj'
                # but "mlp.up_proj" is usually unique enough.
                return mask
        return None

    # 2. Helper: Load, Dequantize, and Mask
    def get_masked_weight(module, mask, dim_to_mask):
        # Get the underlying weight
        # If it's a LoRA layer, the weight is usually in module.base_layer.weight or module.weight
        if hasattr(module, 'base_layer'):
            weight = module.base_layer.weight
        else:
            weight = module.weight

        # Dequantize if 4-bit
        if hasattr(module, 'quant_state') or (hasattr(module, 'base_layer') and hasattr(module.base_layer, 'quant_state')):
            # Handle potential nesting of quant state
            q_module = module.base_layer if hasattr(module, 'base_layer') else module
            weight = bnb.functional.dequantize_4bit(weight, q_module.quant_state)
        
        weight = weight.float()

        # Apply Mask
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(weight.device)
            if dim_to_mask == 0: # Mask Output (Rows)
                weight = weight * mask.view(-1, 1)
            else: # Mask Input (Cols)
                weight = weight * mask.view(1, -1)
        
        return weight

    lpzero_score = 0.0
    
    # Access the actual layer list (handling potential PEFT wrapping)
    if hasattr(model.model, 'model'):
        layers = model.model.model.layers
    elif hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    for i, layer in enumerate(layers):
        
        # --- MLP BLOCK ---
        # W1: up_proj (Expansion)
        # W2: down_proj (Contraction)
        # The mask for the "intermediate neurons" is usually stored under the up_proj key.
        
        if active_layers_mlp[i] == 1:
            # Try to find mask for up_proj. Keys look like: ...layers.0.mlp.up_proj.base_layer
            neuron_mask = get_mask_by_suffix("mlp.up_proj", i, fisher_mask)
            
            # W1 (up_proj): Prune Output Rows (dim 0)
            w1 = get_masked_weight(layer.mlp.up_proj, neuron_mask, dim_to_mask=0)
            
            # W2 (down_proj): Prune Input Cols (dim 1) using THE SAME mask
            w2 = get_masked_weight(layer.mlp.down_proj, neuron_mask, dim_to_mask=1)
            
            # LPZero Calc
            term1 = (torch.norm(w1, p=1) ** 2)
            
            w2_flat = w2.reshape(-1)
            # Only softmax the non-zero elements?
            # Standard LPZero applies softmax to the whole vector. 
            # The masked weights become 0. exp(0) = 1.
            # This is fine, it penalizes "dead" weights slightly less than huge negative weights,
            # but maintains the proxy logic.
            probs = torch.softmax(w2_flat, dim=0)
            term2 = torch.sum(torch.sqrt(probs))
            
            lpzero_score += (term1 + term2).item()

        # --- ATTENTION BLOCK ---
        # W1: v_proj (Value)
        # W2: o_proj (Output)
        # Mask is usually on v_proj (or q/k) representing Heads.
        
        if active_layers_attn[i] == 1:
            # Find mask for v_proj
            head_mask = get_mask_by_suffix("self_attn.v_proj", i, fisher_mask)
            
            # NOTE: If your mask is size [Num_Heads] (e.g. 32), we must expand it.
            # If your mask is already size [Hidden_Dim] (e.g. 4096), use as is.
            if head_mask is not None and len(head_mask) < w1.shape[1]:
                 head_dim = w1.shape[0] // len(head_mask)
                 head_mask = torch.repeat_interleave(head_mask, head_dim)

            # W1 (v_proj): Prune Output Rows (dim 0)
            w1 = get_masked_weight(layer.self_attn.v_proj, head_mask, dim_to_mask=1)
            
            # W2 (o_proj): Prune Input Cols (dim 1)
            w2 = get_masked_weight(layer.self_attn.o_proj, head_mask, dim_to_mask=0)
            
            term1 = (torch.norm(w1, p=1) ** 2)
            w2_flat = w2.reshape(-1)
            probs = torch.softmax(w2_flat, dim=0)
            term2 = torch.sum(torch.sqrt(probs))
            
            lpzero_score += (term1 + term2).item()

    return lpzero_score, fisher_mask, 0.0

def compute_pruner_zero(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    eval_iterator = iter(dataloader)
    
    # 1. Setup Active Layers
    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]
    
    # Apply structural masks (Heads/Neurons)
    set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)
    model.set_active_layers(active_layers_attn, active_layers_mlp)

    # 2. Get a single batch for the gradient calculation
    model.zero_grad()
    try:
        test_inputs = next(eval_iterator)
    except StopIteration:
        return 0.0, fisher_mask, None

    # 3. Backward pass to populate LoRA gradients
    with compute_loss_context_manager():
        rand_loss, _ = compute_loss(model, test_inputs, return_runtime=True)
        accelerator.backward(rand_loss)

    # 4. Collection phase
    layer_data = []
    all_abs_grads = []

    for name, module in model.named_modules():
        # Target PEFT LoRA layers (e.g., q_proj, k_proj, etc.)
        if hasattr(module, 'lora_A') and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            
            # Identify layer index from name
            parts = name.split('.')
            layer_idx = next((int(p) for p in parts if p.isdigit()), None)
            
            # Check if layer is active
            is_attn = any(x in name for x in ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'])
            is_mlp = any(x in name for x in ['mlp', 'gate_proj', 'up_proj', 'down_proj'])
            
            if is_attn and layer_idx not in active_layer_attn_idx: continue
            if is_mlp and layer_idx not in active_layer_mlp_idx: continue

            # Get the frozen base weight W
            base_layer = module.base_layer
            if hasattr(base_layer, "weight"):
                if hasattr(base_layer, "quant_state"): # Handle 4-bit
                    W = bnb.functional.dequantize_4bit(base_layer.weight.data, base_layer.quant_state).float()
                else:
                    W = base_layer.weight.data.float()
            
            # Get the LoRA gradients G
            # G is approximated by the norm of the adapter gradients
            grad_a = module.lora_A.default.weight.grad
            grad_b = module.lora_B.default.weight.grad
            
            if grad_a is not None and grad_b is not None:
                # Calculate a scalar gradient sensitivity for the layer
                # We use the mean magnitude of gradients as the proxy for |G|
                g_sens = (grad_a.detach().abs().mean() + grad_b.detach().abs().mean()) / 2
                
                layer_data.append({
                    'w_sq_sum': W.pow(2).sum().item(),
                    'g_sens': g_sens
                })
                all_abs_grads.append(g_sens)

    if not layer_data:
        print("Error: No active LoRA gradients found. Ensure adapters are trainable.")
        return 0.0, fisher_mask, rand_loss

    # 5. Global Min-Max Scaling for sigma(|G|)
    all_abs_grads = torch.stack(all_abs_grads)
    g_min = all_abs_grads.min()
    g_max = all_abs_grads.max()
    
    total_pruner_zero_score = 0.0
    for data in layer_data:
        # sigma(|G|)
        norm_grad = (data['g_sens'] - g_min) / (g_max - g_min + 1e-12)
        
        # S = |W|^2 * sigma(|G|)
        total_pruner_zero_score += data['w_sq_sum'] * norm_grad.item()

    # Cleanup
    model.zero_grad()
    reset_width_mask(args, model, fisher_mask)

    return total_pruner_zero_score, fisher_mask, rand_loss

def compute_synaptic_(args, model, dataloader, active_layers_attn, active_layers_mlp, w, compute_loss_context_manager, compute_loss, accelerator, fisher_mask=None):
    TARGET_TYPES = (nn.Linear, transformers.Conv1D, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)

    def synaptic_saliency(model):
        metric_array = []
        count_checked = 0
        count_no_grad = 0
        
        for name, module in model.named_modules():
            # Check against extended types tuple
            if isinstance(module, TARGET_TYPES):
                count_checked += 1
                
                # Verify weight and grad exist
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.grad is not None and not torch.all(module.weight.grad==0):
                        # Calculation: |W * dW|
                        metric = torch.abs(module.weight * module.weight.grad)
                        metric_array.append(metric)
                    else:
                        count_no_grad += 1
        
        if len(metric_array) == 0:
            print(f"[Warning] Saliency found 0 valid layers. Checked: {count_checked}. No Grad: {count_no_grad}.")
            return 0.0

        # Efficient Sum
        summed = sum([torch.nansum(m) for m in metric_array])
        return summed.detach().item()
    
    def compute_neuron_mixed_saliency(model):
        total_score = 0
        
        # Pre-compile regex for speed
        # Captures standard Llama 2/3 module names
        target_pattern = re.compile(r'layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$')

        for name, module in model.named_modules():
            if target_pattern.search(name):
                
                # --- 1. Get Base Weight Rows (Neurons) ---
                # We need the weights in valid float format
                w_base = module.weight 
                if hasattr(module, 'quant_state'):
                    # Dequantize to get accurate magnitude of base neurons
                    w_base = bnb.functional.dequantize_4bit(w_base, module.quant_state)
                
                # Calculate Norm per Row (dim=1) -> Shape: [Out_Features]
                # This tells us: "How 'heavy' is this specific neuron in the base model?"
                w_row_norms = torch.norm(w_base.float(), p=2, dim=1)

                # --- 2. Get LoRA B Gradient Rows ---
                # LoRA B is the "Output" adapter, so its rows align 1:1 with Base Model rows.
                # (LoRA A is input-side, so dimensions wouldn't match for row-wise)
                if hasattr(module, 'lora_B') and 'default' in module.lora_B:
                    lora_b_grad = module.lora_B['default'].weight.grad
                    
                    if lora_b_grad is not None:
                        # Calculate Gradient Norm per Row (dim=1) -> Shape: [Out_Features]
                        # This tells us: "How actively is this specific neuron being updated?"
                        g_row_norms = torch.norm(lora_b_grad.float(), p=2, dim=1)
                        
                        # --- 3. Combine Neuron-wise ---
                        # Element-wise multiplication: Base_Neuron_Mag * LoRA_Grad_Mag
                        # Result is a vector of scores for every neuron in this layer
                        neuron_scores = w_row_norms * g_row_norms
                        
                        # Sum all neuron scores to get the total contribution
                        total_score += neuron_scores.sum().item()

        return total_score
    
    def compute_layer_mixed_diveristy(model):
        total_score = 0
        for name, module in model.named_modules():
            match = re.search(
                r'layers\.(\d+)\.(self_attn.q_proj|self_attn.k_proj|self_attn.v_proj|self_attn.o_proj|mlp.up_proj|mlp.gate_proj|mlp.down_proj)$',
                name
            )
            if match:                
                # 1. Base Weight Norm (Frobenius)
                # You don't even need to dequantize fully if you just want a rough estimate,
                # but dequantizing is safer for accuracy.
                w_base = module.weight.float() #.detach()
                if hasattr(module, 'quant_state'):
                    w_base = bnb.functional.dequantize_4bit(w_base, module.quant_state)
                w_norm = torch.norm(w_base, 'nuc') #.float())

                # 2. LoRA Gradient Norm
                lora_b_grad = module.lora_B.default.weight.grad.float()
                
                if lora_b_grad is not None:
                    g_norm = torch.norm(lora_b_grad, 'nuc') #.float())
                    
                    # 3. Combine
                    total_score += (w_norm * g_norm).item()

        return total_score
    
    def synaptic_diversity(model):
        metric_array = []
        
        for name, module in model.named_modules():
            # --- FIX: used 'module', not 'layer' ---
            if isinstance(module, TARGET_TYPES):
                
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.grad is not None:
                        # Calculation: Nuclear Norm (Very Expensive!)
                        # Note: Calculating Nuclear Norm on high-dim weights inside a loop is extremely slow.
                        # Using Frobenius norm ('fro') is standard for synaptic diversity if speed matters.
                        
                        w_nuc = torch.norm(module.weight.float(), p='nuc') # Cast to float for stability
                        g_nuc = torch.norm(module.weight.grad.float(), p='nuc')
                        
                        metric_array.append(torch.abs(w_nuc * g_nuc))

        if len(metric_array) == 0:
            return 0.0

        summed = sum([torch.nansum(m) for m in metric_array])
        return summed.detach().item()

    eval_iterator = iter(dataloader)

    influence_score = 0

    l_attn, l_mlp, max_layer = int(active_layers_attn.sum().item()), int(active_layers_mlp.sum().item()), model.config.num_hidden_layers


    active_layer_attn_idx = np.where(active_layers_attn == 1)[0]
    active_layer_mlp_idx = np.where(active_layers_mlp == 1)[0]

    attn_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_attn_idx}
    mlp_neuron_importance = {layer_idx:0 for layer_idx in  active_layer_mlp_idx}
    
    layer_tracin_attn = {layer_idx.item(): 0 for layer_idx in active_layer_attn_idx}
    layer_tracin_mlp = {layer_idx.item(): 0 for layer_idx in active_layer_mlp_idx}


    fisher_mask, attn_masks, mlp_masks = get_attn_mlp_masks(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    # set_width_mask(args, model, fisher_mask)
    set_width_mask(args, model, fisher_mask, active_layer_attn_idx, active_layer_mlp_idx)

    

    model.set_active_layers(active_layers_attn, active_layers_mlp)

    layer_grads_attn = {layer_idx.item(): list() for layer_idx in active_layer_attn_idx}
    layer_grads_mlp = {layer_idx.item(): list() for layer_idx in active_layer_mlp_idx}
    
    total_time_s = 0
    total_batches = 0
    score = 0
    with compute_loss_context_manager():                    
        for i, test_inputs in enumerate(eval_iterator):
            model.zero_grad()
            
            rand_loss, b_total_time_s = compute_loss(model, test_inputs, return_runtime=True)
            total_time_s += b_total_time_s
            total_batches += 1

            accelerator.backward(rand_loss, retain_graph=True)

            if args.influence_type=='synaptic_saliency':
                # score += synaptic_saliency(model)
                score += compute_neuron_mixed_saliency(model)
            elif args.influence_type=='synaptic_diversity':
                # score += synaptic_diversity(model)
                score += compute_layer_mixed_diveristy(model)

    reset_width_mask(args, model, fisher_mask)

    return score, fisher_mask, rand_loss