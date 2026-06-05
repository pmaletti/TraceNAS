import os
import sys
import json
import torch
import numpy as np
import re
import argparse

# Ensure the custom transformers path is inserted if needed
sys.path.insert(0, "./transformers/src") 
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import NoAttention, NoMLP


# ==========================================
# 1. Model & Mask Generation Logic 
# ==========================================

def load_model(model_path, cache_dir, device="cuda", load_from_pruned_path=None):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    print('load_from_pruned_path:',load_from_pruned_path)
    if load_from_pruned_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=cache_dir,
            shrink_config={
                    'enable_shrinking': False, 
                    "shrinkable_width": False,
                    "shrinking_method": '',
                    "shrinking_file": 'dp_selection_strategy.npy',
                    "mask_dtype": "torch.bfloat16"}
                )
    else:
        config = AutoConfig.from_pretrained(load_from_pruned_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_from_pruned_path,
            config=config,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            shrink_config={
                    'enable_shrinking': False, 
                    "shrinkable_width": False,
                    "shrinking_method": '',
                    "shrinking_file": 'dp_selection_strategy.npy',
                    "mask_dtype": "torch.bfloat16"}
        )
    model.eval()
    return model

def generate_width_masks_good(model, attn_inps, mlp_inps, sparsity):
    """
    Generates width masks for the BEST configuration using FLAP/Variance Metric.
    """
    is_gqa_or_mqa = model.config.num_key_value_heads != model.config.num_attention_heads

    final_masks = {}
    attn_sparsity = sparsity['attn']
    mlp_sparsity = sparsity['mlp']
    
    if hasattr(model.model, "layers"):
        layers = model.model.layers
        prefix = "layers"
    else:
        layers = model.model.model.layers
        prefix = "layers"

    for i, layer in enumerate(layers):
        idx = str(i)
        
        # --- 1. Attention Block ---
        if idx in sparsity['attn'].keys():
            raw_o_proj = layer.self_attn.o_proj
            o_proj_module = getattr(raw_o_proj, "base_layer", raw_o_proj)

            weight = o_proj_module.weight
            C_out, C_in = weight.shape
            
            # FLAP Metric: Weight * Input Scale
            scale_factor = attn_inps[i].to(weight.device)
            importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)

            structured_scores = importance_scores.sum(dim=0) 
            num_to_keep = max(1, int(C_in * (1 - attn_sparsity[idx])))
            _, sorted_indices = torch.topk(structured_scores, k=num_to_keep)
            mask_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
            mask_1d[sorted_indices] = True
            mask_2d = mask_1d.unsqueeze(0).expand(C_out, C_in) 
            mask_2d = mask_2d.detach().cpu()

            final_masks[f"{prefix}.{i}.self_attn.q_proj"] = mask_2d
            
            if not is_gqa_or_mqa:
                final_masks[f"{prefix}.{i}.self_attn.k_proj"] = mask_2d
                final_masks[f"{prefix}.{i}.self_attn.v_proj"] = mask_2d
            else:
                k_size = layer.self_attn.k_proj.weight.shape[0]
                final_masks[f"{prefix}.{i}.self_attn.k_proj"] = torch.ones(k_size).to(mask_2d)
                v_size = layer.self_attn.v_proj.weight.shape[0]
                final_masks[f"{prefix}.{i}.self_attn.v_proj"] = torch.ones(v_size).to(mask_2d)

        # --- 2. MLP Block ---
        if idx in sparsity['mlp'].keys():
            raw_down_proj = layer.mlp.down_proj
            down_proj_module = getattr(raw_down_proj, "base_layer", raw_down_proj)
            
            weight = down_proj_module.weight
            C_out, C_in = weight.shape  
            
            scale_factor = mlp_inps[i].to(weight.device)
            importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)

            structured_scores = importance_scores.sum(dim=0) 
            num_to_keep = max(1, int(C_in * (1 - mlp_sparsity[idx])))
            _, sorted_indices = torch.topk(structured_scores, k=num_to_keep)
            mask_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
            mask_1d[sorted_indices] = True
            mask_2d = mask_1d.unsqueeze(0).expand(C_out, C_in) 
            mask_2d = mask_2d.detach().cpu()

            final_masks[f"{prefix}.{i}.mlp.gate_proj"] = mask_2d
            final_masks[f"{prefix}.{i}.mlp.up_proj"] = mask_2d

    return final_masks

def generate_width_masks(model, attn_inps, mlp_inps, sparsity, attn_grads=None, mlp_grads=None):
    """
    Generates width masks for the BEST configuration using FLAP/Variance Metric.
    Handles variable-width layers and dummy modules (NoAttention/NoMLP).
    """
    model_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config

    is_pythia = "GPTNeoX" in model.config.architectures[0]
    
    # Identify Architecture Type
    num_q_heads_base = model_config.num_attention_heads
    num_kv_heads_base = getattr(model_config, 'num_key_value_heads', num_q_heads_base)
    is_gqa_or_mqa = num_q_heads_base != num_kv_heads_base
    head_dim = getattr(model_config, 'head_dim', model_config.hidden_size // num_q_heads_base)

    final_masks = {}
    attn_sparsity = sparsity['attn']
    mlp_sparsity = sparsity['mlp']

    if is_pythia:
        if hasattr(model.gpt_neox, "layers"):
            layers = model.gpt_neox.layers
            prefix = "layers"
        else:
            layers = model.gpt_neox.model.layers
            prefix = "layers"
    else: 
        if hasattr(model.model, "layers"):
            layers = model.model.layers
            prefix = "layers"
        else:
            layers = model.model.model.layers
            prefix = "layers"

    # --- Identify Indices to Ignore (Dummy Modules or Config 0s) ---
    skip_attn_indices = {
        idx for idx, layer in enumerate(layers) 
        if isinstance(getattr(layer, 'self_attn', getattr(layer, 'attention', None)), NoAttention) or 
        (hasattr(model_config, 'hidden_size_list') and model_config.hidden_size_list[idx] == 0)
    }
    skip_mlp_indices = {
        idx for idx, layer in enumerate(layers) 
        if isinstance(getattr(layer, 'mlp', None), NoMLP) or 
        (hasattr(model_config, 'intermediate_size_list') and model_config.intermediate_size_list[idx] == 0)
    }

    for i, layer in enumerate(layers):
        idx_str = str(i)
        
        # Determine current sizes for this specific layer index
        if hasattr(model_config, 'hidden_size_list'):
            curr_num_q_heads = model_config.num_attention_heads_list[i]
            # Use the base ratio to determine KV heads for this layer
            kv_ratio = num_q_heads_base // num_kv_heads_base if is_gqa_or_mqa else 1
            curr_num_kv_heads = max(1, curr_num_q_heads // kv_ratio)
        else:
            curr_num_q_heads = num_q_heads_base
            curr_num_kv_heads = num_kv_heads_base

        # --- 1. Attention Block ---
        if i not in skip_attn_indices and idx_str in attn_sparsity:
            # raw_o_proj = layer.self_attn.o_proj
            if hasattr(layer, "self_attn"):
                raw_o_proj =  layer.self_attn.o_proj
            
            # Check Pythia / GPT-NeoX style
            elif hasattr(layer, "attention"):
                raw_o_proj = layer.attention.dense

            o_proj_module = getattr(raw_o_proj, "base_layer", raw_o_proj)

            weight = o_proj_module.weight
            C_out, C_in = weight.shape # C_in should match curr_hidden_size
            
            scale_factor = attn_inps[i].to(weight.device)
            # if attn_grads is None:                
            importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)
            # else:
            #     grad_sensitivity = attn_grads[i].to(weight.device)
            #     importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0) * grad_sensitivity
            structured_scores = importance_scores.sum(dim=0) 

            if is_gqa_or_mqa:
                # GQA: Prune Q heads, keep KV full
                q_scores = structured_scores.view(curr_num_q_heads, head_dim).sum(dim=1)
                
                raw_keep_count = curr_num_q_heads * (1.0 - attn_sparsity[idx_str])
                # Ensure Q heads is a multiple of KV heads for SDPA compatibility
                step_size = curr_num_kv_heads
                target_q_heads = int(round(raw_keep_count / step_size) * step_size)
                target_q_heads = max(step_size, min(target_q_heads, curr_num_q_heads))
                
                _, top_q_indices = torch.topk(q_scores, k=target_q_heads)
                
                mask_q_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
                for q_idx in top_q_indices:
                    mask_q_1d[q_idx*head_dim : (q_idx+1)*head_dim] = True
                
                mask_2d_q = mask_q_1d.unsqueeze(0).expand(C_out, -1).detach().cpu()

                if is_pythia:
                    final_masks[f"{prefix}.{i}.attention.query_key_value"] = mask_2d_q
                else:
                    final_masks[f"{prefix}.{i}.self_attn.q_proj"] = mask_2d_q

                    # KV Masks are 1.0 (Full) for GQA pruning
                    k_size = layer.self_attn.k_proj.weight.shape[0]
                    final_masks[f"{prefix}.{i}.self_attn.k_proj"] = torch.ones(k_size, C_out).bool().cpu() # Assuming transposed or correct shape
                    v_size = layer.self_attn.v_proj.weight.shape[0]
                    final_masks[f"{prefix}.{i}.self_attn.v_proj"] = torch.ones(v_size, C_out).bool().cpu()

            else:
                # MHA: Standard pruning
                num_to_keep = max(1, int(C_in * (1 - attn_sparsity[idx_str])))
                _, sorted_indices = torch.topk(structured_scores, k=num_to_keep)
                mask_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
                mask_1d[sorted_indices] = True
                mask_2d = mask_1d.unsqueeze(0).expand(C_out, C_in).detach().cpu()
                
                if is_pythia:
                    # CRITICAL: Pythia's query_key_value has 3x output features
                    # We repeat the mask for Q, K, and V segments
                    mask_1d_fused = torch.cat([mask_1d, mask_1d, mask_1d], dim=0)
                    
                    # Hard Pruning script prunes dim 0 of query_key_value
                    # We need a 2D mask where the first dim matches (3 * hidden_size)
                    fused_out_features = 3 * C_in # Usually 3 * hidden_size
                    mask_2d_fused = mask_1d_fused.unsqueeze(1).expand(-1, C_in).detach().cpu()
                    final_masks[f"{prefix}.{i}.attention.query_key_value"] = mask_2d_fused
                else:
                    final_masks[f"{prefix}.{i}.self_attn.q_proj"] = mask_2d
                    final_masks[f"{prefix}.{i}.self_attn.k_proj"] = mask_2d
                    final_masks[f"{prefix}.{i}.self_attn.v_proj"] = mask_2d

        # --- 2. MLP Block ---
        if i not in skip_mlp_indices and idx_str in mlp_sparsity:
            # raw_down_proj = layer.mlp.down_proj
            if hasattr(layer.mlp, "down_proj"):
                raw_down_proj = layer.mlp.down_proj
            
            # Pythia / GPT-NeoX
            elif hasattr(layer.mlp, "dense_4h_to_h"):
                raw_down_proj = layer.mlp.dense_4h_to_h

            down_proj_module = getattr(raw_down_proj, "base_layer", raw_down_proj)
            
            weight = down_proj_module.weight
            C_out, C_in = weight.shape  
            
            scale_factor = mlp_inps[i].to(weight.device)
            # if mlp_grads is None:
            importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0)
            # else:
            #     grad_sensitivity = mlp_grads[i].to(weight.device)
            #     importance_scores = torch.abs(weight) * scale_factor.unsqueeze(0) * grad_sensitivity
            
            structured_scores = importance_scores.sum(dim=0) 
            num_to_keep = max(1, int(C_in * (1 - mlp_sparsity[idx_str])))
            _, sorted_indices = torch.topk(structured_scores, k=num_to_keep)
            
            mask_1d = torch.zeros(C_in, dtype=torch.bool, device=weight.device)
            mask_1d[sorted_indices] = True
            mask_2d = mask_1d.unsqueeze(0).expand(C_out, C_in).detach().cpu()

            if is_pythia:
                final_masks[f"{prefix}.{i}.mlp.dense_h_to_4h"] = mask_2d
            else:
                final_masks[f"{prefix}.{i}.mlp.gate_proj"] = mask_2d
                final_masks[f"{prefix}.{i}.mlp.up_proj"] = mask_2d

    return final_masks

# ==========================================
# 2. Main Logic: Select Best & Save
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True, help="Directory containing influence scores and inputs")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--target_params", type=float, default=2.7e9)
    parser.add_argument("--tolerance", type=float, default=0.25e9)
    parser.add_argument("--target_config", type=str, default=None, help="Format: (L_attn, L_mlp, Width)")
    parser.add_argument("--iter", type=int, default=30)
    parser.add_argument("--load_from_pruned_path", type=str, default=None)
    args = parser.parse_args()

    it = args.iter
    # --- Paths & Config ---
    # Use arguments instead of hardcoded paths
    base_folder = args.base_folder
    
    attn_inp_path = f'{base_folder}/base_inputs_seed42/base_attn_inputs.pt'
    mlp_inp_path = f'{base_folder}/base_inputs_seed42/base_mlp_inputs.pt'

    attn_grad_path = f'{base_folder}/base_grads_seed42/base_attn_inputs.pt'
    mlp_grad_path = f'{base_folder}/base_grads_seed42/base_mlp_inputs.pt'

    # Note: Assuming these filenames follow the pattern in your script
    s1_path = f'{base_folder}/active_layers_seed42_iter_{it}.npy'
    sparsity_paths_file = f'{base_folder}/active_width_mask_seed42_iter_{it}.npy'
    s3_path = f'{base_folder}/influence_scores_seed42_iter_{it}.txt'

    # --- Constraints for "Best" Selection ---
    min_params = args.target_params - args.tolerance
    max_params = args.target_params + args.tolerance

    print(min_params)
    print(max_params)

    # --- 1. Find Best Index ---
    # Print to stderr so we don't pollute the variable capture in Bash
    print("Reading scores to find best configuration...", file=sys.stderr)
    
    if not os.path.exists(s3_path):
        print(f"Error: Influence scores file not found at {s3_path}", file=sys.stderr)
        sys.exit(1)

    with open(s3_path, 'r') as f:
        data = [line.strip() for line in f if line.strip()]

    best_index = -1
    best_score = -float('inf')
    best_params = None
    key_config = ""
    
    if args.target_config is not None:
        for i, line in enumerate(data):
            if args.target_config in line:
                best_index = i
                parts = line.split('-----')
                if len(parts) < 3: continue
                best_score = float(parts[0])
                best_params = float(parts[1])
                key_config = parts[2] 
                break
    else:
        for i, line in enumerate(data):
            parts = line.split('-----')
            if len(parts) < 3: continue
            score = float(parts[0])
            params = float(parts[1])
            structure = parts[2] 
            
            if min_params <= params <= max_params:
                if score > best_score:
                    best_score = score
                    best_index = i
                    best_params = params
                    key_config = structure

    if best_index == -1:
        print("No configuration found within valid parameters!", file=sys.stderr)
        sys.exit(1)

    print(f"Best Config Found: Score={best_score}, Params={best_params}, Config={key_config}", file=sys.stderr)

    # --- 2. Load Resources ---
    print("Loading Model...", file=sys.stderr)
    model = load_model(args.model_name_or_path, args.cache_dir, load_from_pruned_path=args.load_from_pruned_path)
    
    attn_inps = torch.load(attn_inp_path)
    mlp_inps = torch.load(mlp_inp_path)

    if os.path.exists(attn_grad_path) and os.path.exists(mlp_grad_path):
        print('Fetching attn and mlp grads')
        attn_grads = torch.load(attn_grad_path)
        mlp_grads = torch.load(mlp_grad_path)
    else:
        attn_grads = None
        mlp_grads = None
    
    s1 = np.load(s1_path, allow_pickle=True)
    sparsity_paths = np.load(sparsity_paths_file, allow_pickle=True)

    best_active_layers = s1[best_index][0]
    best_sparsity_path = sparsity_paths[best_index].item()

    best_tuple = tuple(eval(key_config)) # (layers_attn, layers_mlp, width_mask)
    l_a, l_m, target_width = best_tuple
    l = (l_a, l_m, target_width)

    # --- 3. Generate Mask ---
    print(f"Generating mask using: {best_sparsity_path}", file=sys.stderr)
    with open(best_sparsity_path, 'r') as f:
        sparsity_config = json.load(f)

    width_masks = generate_width_masks(model, attn_inps, mlp_inps, sparsity_config, attn_grads, mlp_grads)

    # --- 4. Format Output ---
    strategy = {}
    strategy[l] = {
        'active_layers': [best_active_layers],
        'effective_width': [target_width],
        'lora_r': []
    }
    meta_width_mask = {}
    for key, mask in width_masks.items():
        if key not in meta_width_mask: meta_width_mask[key] = {}
        meta_width_mask[key][l] = mask

    s5 = {
        'strategy': strategy,
        'l_w_tuples': [best_tuple],
        'meta_width_mask': meta_width_mask
    }
    
    # --- 5. Save ---
    safe_config_name = key_config.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
    output_path = f'{base_folder}/elastic_model_BEST_seed42_iter_{it}_{safe_config_name}.npy'
    
    print(f"Saving best elastic model to: {output_path}", file=sys.stderr)
    np.save(output_path, s5)

    # --- 6. PRINT RESULTS FOR BASH CAPTURE ---
    # We print ONLY the assignment variables to stdout
    print(f"DETECTED_SHRINK_FILE={output_path}")
    print(f"DETECTED_CONFIG={safe_config_name}")

if __name__ == '__main__':
    main()