# !/bin/bash

CACHE_DIR="/scratch/gautschi/pmaletti/hf_cache/huggingface/"
export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

save_hyperparams() {
    local output_path="$1"
    local phase_name="$2"
    local lr="$3"
    local warmup="$4"
    local lora_r="$5"
    local seed="$6"
    
    echo "Saving config for ${phase_name} to ${output_path}/hyperparameters.txt"
    
    cat <<EOF > "${output_path}/hyperparameters.txt"
==================================================
TRAINING CONFIGURATION: ${phase_name}
==================================================
Date: $(date)
Model: ${model_name_or_path}

--- Structure & Search ---
Target Params: ${target_size} 
Actual Params: ${actual_params}
Layers (Attn): ${eval_num_layers_attn}
Layers (MLP):  ${eval_num_layers_mlp}
Width Mask:    ${eval_width_mask}
Strategy File: ${EVOL_SEARCH_SHRINK_FILE}

--- Training Hyperparameters ---
Phase: ${phase_name}
Max Steps: ${CPT_STEPS} (CPT)
Global Batch Size: ${TARGET_GLOBAL_BS}
Gradient Accumulation: ${GRAD_ACCUM}
GPUs: ${NUM_GPUS} (Per Device BS: ${PER_DEVICE_BS})
Learning Rate: ${lr}
Warmup Ratio: ${warmup}
LoRA rank: ${lora_r}
Seed: ${seed}

--- Paths ---
Output Dir: ${output_path}
Cache Dir: ${cache_dir}
Hard Prune Path: ${base_hard_prune_path}
==================================================
EOF
}

# --- 1. CONFIGURATION VARIABLES ---
# Base Model Config
model_name_or_path="meta-llama/Llama-2-7b-hf"
cache_dir=/scratch/gautschi/pmaletti/${model_name_or_path}
dataset_cache_dir=/scratch/gautschi/pmaletti/datasets/local_fineweb_edu
scratch_dir="/scratch/gautschi/pmaletti/tracenas"
SEARCH_RESULTS_DIR="${scratch_dir}/final_output/fineweb/layer_prune/${model_name_or_path}/search_iter50_calib_samples16_ps30_ne10_mr0.2_cr0.7/"

# Architecture Search Config
target_size=2.7e9

# Training Steps
CPT_STEPS=5000

# =========================================================
# STEP 0: DYNAMICALLY GENERATE MASKS & SELECT CONFIG
# =========================================================
echo "--- Running load_masks.py to select best config ---"

PYTHON_OUTPUT=$(python utils/load_masks.py \
    --base_folder "${SEARCH_RESULTS_DIR}" \
    --model_name_or_path "${model_name_or_path}" \
    --cache_dir "${cache_dir}" \
    --target_params "${target_size}" \
    --tolerance 0.08e9 \
    --iter 48)
# 2. CHECK FOR FAILURE
if [ $? -ne 0 ]; then
    echo "Error: load_masks.py failed."
    exit 1
fi

# 3. DEBUG: PRINT CAPTURED OUTPUT (Optional but recommended)
# This confirms the variables were actually caught.
echo "DEBUG: Python Output Captured:"
echo "$PYTHON_OUTPUT"

# 4. PARSE VARIABLES
EVOL_SEARCH_SHRINK_FILE=$(echo "$PYTHON_OUTPUT" | grep "DETECTED_SHRINK_FILE" | cut -d'=' -f2)
BEST_KEY_CONFIG=$(echo "$PYTHON_OUTPUT" | grep "DETECTED_CONFIG" | cut -d'=' -f2)

# Safety Check: If grep didn't find anything, stop here.
if [ -z "$BEST_KEY_CONFIG" ]; then
    echo "Error: Could not parse DETECTED_CONFIG from python output."
    exit 1
fi

echo "Selected Strategy File: $EVOL_SEARCH_SHRINK_FILE"
echo "Selected Config Key:    $BEST_KEY_CONFIG"

# 5. SPLIT CONFIG STRING
IFS='_' read -r eval_num_layers_attn eval_num_layers_mlp eval_width_mask <<< "$BEST_KEY_CONFIG"

echo "Parsed Layers (Attn):   $eval_num_layers_attn"
echo "Parsed Layers (MLP):    $eval_num_layers_mlp"
echo "Parsed Width Mask:      $eval_width_mask"

# --- PATHS (Updated to use the dynamic BEST_KEY_CONFIG) ---
base_hard_prune_path_template="${scratch_dir}/final_output/${model_name_or_path}/ft/${BEST_KEY_CONFIG}/hard_pruned_models/hard_pruned_ckpt"

# Set output directory for CPT based on the selected config
cpt_output_dir="${scratch_dir}/final_output/${model_name_or_path}/ft/${BEST_KEY_CONFIG}/CPT"

echo "Using Strategy File: $EVOL_SEARCH_SHRINK_FILE"

# 4.3. GPU & BATCH CALCULATION
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TARGET_GLOBAL_BS=1024
PER_DEVICE_BS=24
GRAD_ACCUM=$((TARGET_GLOBAL_BS / (PER_DEVICE_BS * NUM_GPUS)))
if [[ $GRAD_ACCUM -lt 1 ]]; then GRAD_ACCUM=1; fi
MASTER_PORT=$(shuf -i 29500-65000 -n 1)
width_choice="${eval_width_mask},${eval_width_mask}"

# =========================================================
# CONTINUED PRETRAINING (CPT)
# =========================================================
LEARNING_RATE="1e-4"
WARMUP_RATIO=0.05 #0.1 #"0.0105"
STABLE_RATIO=0.65 #0.1 #"0.0105"
DECAY_RATIO=0.30 #0.1 #"0.0105"
SEED=42
        
save_hyperparams "${cpt_output_dir}" "CPT" "${LEARNING_RATE}" "${WARMUP_RATIO}" "${lora_r}" "${SEED}"
cpt_ckpt_dir="${cpt_output_dir}/checkpoint-${CPT_STEPS}/"

if [[ -d "${cpt_ckpt_dir}" ]]; then
    echo "CPT already completed for ${BEST_KEY_CONFIG}. Skipping."
else
    echo "--- Phase 1: Continued Pretraining ---"
    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} main.py \
        --model_name_or_path "${model_name_or_path}" \
        --output_dir "${cpt_output_dir}" \
        --cache_dir "${cache_dir}" \
        --dataset "fineweb_edu" \
        --dataset_format "fineweb_edu" \
        --dataset_cache_dir ${dataset_cache_dir} \
        --max_train_samples 10000000 \
        --use_auth_token \
        --do_train True \
        --do_eval False \
        --bf16 --bits 16 \
        --source_max_len 16 \
        --target_max_len 4096 \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --logging_steps 10 \
        --max_steps ${CPT_STEPS} \
        --save_strategy "steps" \
        --seed ${SEED} \
        --save_steps 50 \
        --save_total_limit 1 \
        --evaluation_strategy "steps" \
        --optim "adamw_torch_fused" \
        --shrinking_file "${EVOL_SEARCH_SHRINK_FILE}" \
        --width_choice "${width_choice}" \
        --full_finetune True \
        --resume_training True \
        --per_device_train_batch_size ${PER_DEVICE_BS} \
        --eval_num_layer "$eval_num_layers_attn" \
        --eval_num_width "$eval_width_mask" \
        --train_on_source True \
        --group_by_length False \
        --predict_with_generate False \
        --hard_prune True \
        --hard_pruned_dir "${base_hard_prune_path}" \
        --lr_scheduler_type 'CustomWSDScheduler' \
        --learning_rate ${LEARNING_RATE} \
        --warmup_ratio ${WARMUP_RATIO} \
        --stable_ratio ${STABLE_RATIO} \
        --decay_ratio ${DECAY_RATIO} \
        --max_grad_norm 0.5 \
        --dispatch_batches False \
        --split_batches True \
        --ddp_find_unused_parameters False \
        --dataloader_num_workers 3
fi

echo ""
echo "--- Process complete. ---"