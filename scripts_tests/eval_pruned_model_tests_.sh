model_name_or_path="meta-llama/Llama-2-7b-hf"
# model_name_or_path="meta-llama/Llama-3.1-8B"
# model_name_or_path="Qwen/Qwen3-8B"

cache_dir=/local/a/pmaletti/llm_weights/${model_name_or_path}

#Trained Architecture Configuration
eval_num_layers_attn=30
eval_width_mask=0.4

width_choice="${eval_width_mask},${eval_width_mask}"
layer="${eval_num_layers_attn}"
width="${eval_width_mask}"

BEST_KEY_CONFIG="${eval_num_layers_attn}_${eval_num_layers_attn}_${eval_width_mask}"
hard_pruned_dir="final_output/${model_name_or_path}/ft/${BEST_KEY_CONFIG}/CPT/hard_pruned_ckpt/"
output_dir="final_output/${model_name_or_path}/ft/${BEST_KEY_CONFIG}/CPT/eval_noft"

do_mmlu_eval_global=True # Global control flag for MMLU
do_eval_wikitext2_global=True # Global control flag for WIKITEXT2

seed=42 

# --- CHECKPOINT LOGIC ---
# if find "${output_dir}" -maxdepth 1 -type d -name "checkpoint-*" | grep -q .; then
#     echo "Trained checkpoint found."

#     # 1. Capture the specific checkpoint directory
#     # (Assuming output_dir is the correct base, corrected from output_dir_model)
#     ckpt_dir_finetuned=$(find "${output_dir}" -maxdepth 1 -type d -name "checkpoint-*" | head -n 1)
    
#     echo "Checkpoint location: ${ckpt_dir_finetuned}"

# 2. UPDATE hard_pruned_dir TO THE CHECKPOINT PATH
# hard_pruned_dir="${ckpt_dir_finetuned}"
echo ">> hard_pruned_dir updated to: ${hard_pruned_dir}"

# --- 4.3. LM-EVAL LOOP ---
lm_eval_tasks=('arc_easy' 'piqa' 'boolq' 'winogrande' 'arc_challenge' 'openbookqa' 'logiqa' 'sciq' 'hellaswag')

for task in "${lm_eval_tasks[@]}"; do
    # Conditional Few-Shot Logic
    if [ "$task" == "hellaswag" ]; then
        NUM_FEWSHOT=10
    elif [ "$task" == "arc_challenge" ]; then
        NUM_FEWSHOT=25  # <--- CHANGED FROM 10 TO 25
    elif [ "$task" == "winogrande" ]; then
        NUM_FEWSHOT=5
    else
        NUM_FEWSHOT=0
    fi

    # Reset task-specific flags
    do_lm_eval=True
    do_mmlu_eval=True
    do_eval_wikitext2=True

    # Format width
    if [[ "$width" == *"."* ]]; then
        width_="$width"
    else
        width_="${width}.0"
    fi

    # Conditional Suffix
    FS_SUFFIX=""
    if [ "$NUM_FEWSHOT" != "0" ]; then
        FS_SUFFIX="_fs${NUM_FEWSHOT}"
    fi

    # Construct result file paths
    task_results_dir="${output_dir}"
    lm_results_file="${task_results_dir}/${task}_results.json"
    mmlu_results_file="${task_results_dir}/mmlu_results.json"
    wikitext2_results_file="${task_results_dir}/wikitext2_results.json"
    
    echo "--- Checking results for task: ${task}, layer: ${layer}, width: ${width_} ---"

    # Check existing results
    if [[ -f "$lm_results_file" ]]; then
        do_lm_eval=False
        echo "Results for task ${task} already exist. Skipping."
    fi
    
    if [[ -f "$mmlu_results_file" ]]; then
        do_mmlu_eval_global=False
        echo "Results for mmlu already exist. Skipping."
    fi
    
    if [[ -f "$wikitext2_results_file" ]]; then
        do_eval_wikitext2_global=False
        echo "Results for wikitext already exist. Skipping."
    fi

    # Skip execution if everything is done
    if [[ "$do_lm_eval" == "False" && "$do_mmlu_eval_global" == "False" && "$do_eval_wikitext2_global" == "False" ]]; then
        echo "⏭️ All evaluations already completed. Skipping task ${task}."
        continue
    fi
    
    # Determine global run flags
    run_mmlu="${do_mmlu_eval_global}"
    run_wikitext2="${do_eval_wikitext2_global}"

    if [[ "$run_mmlu" == "True" ]]; then
        do_mmlu_eval_global="False"
    fi
    if [[ "$run_wikitext2" == "True" ]]; then
        do_eval_wikitext2_global="False"
    fi

    # 4.4. EXECUTE PYTHON SCRIPT
    echo "Running evaluation for task: $task"
    
    # NOTE: logic ensures --hard_pruned_dir uses the checkpoint path set above
    CUDA_LAUNCH_BLOCKING=1 python main.py \
    --model_name_or_path "${model_name_or_path}" \
    --cache_dir "${cache_dir}" \
    --output_dir "${output_dir}" \
    --do_train False \
    --do_eval True \
    --bits 16 --bf16 \
    --do_mmlu_eval "${run_mmlu}" \
    --do_eval_wikitext2 "${run_wikitext2}" \
    --do_lm_eval "${do_lm_eval}" \
    --do_lm_eval_task "${task}" \
    --full_finetune True \
    --source_max_len 3072 \
    --target_max_len 1024 \
    --max_new_tokens 1024 \
    --few_shot_number "${NUM_FEWSHOT}" \
    --hard_prune True \
    --hard_pruned_dir "${hard_pruned_dir}" \
    --seed ${seed} \
    --dataloader_num_workers 1 \
    --dataloader_prefetch_factor 1

    echo "Finished task: $task"
done 
fi

echo "--- All evaluations complete ---"