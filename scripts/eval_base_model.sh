# !/bin/bash
model_name_or_path="meta-llama/Llama-2-7b-hf"
# model_name_or_path="meta-llama/Llama-3.1-8B"
# model_name_or_path="Qwen/Qwen2.5-14B-Instruct"
num_moe=1
topk=1
lora_r=128

output_dir=final_output/base_eval/${model_name_or_path}/
cache_dir=/YOUR_PATH_TO_MODEL/${model_name_or_path}

do_mmlu_eval_global=True # Global control flag for MMLU
do_eval_wikitext2_global=True # Global control flag for WIKITEXT2
lora_modules=all
seed=42 

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
    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python main.py \
    --model_name_or_path "${model_name_or_path}" \
    --output_dir "${output_dir}" \
    --do_train False \
    --do_eval True \
    --bits 16 --bf16 \
    --do_mmlu_eval "${run_mmlu}" \
    --do_eval_wikitext2 "${run_wikitext2}" \
    --do_lm_eval "${do_lm_eval}" \
    --do_lm_eval_task "${task}" \
    --cache_dir "${cache_dir}" \
    --eval_after_evolsearch True \
    --full_finetune True \
    --source_max_len 3072 \
    --target_max_len 1024 \
    --max_new_tokens 1024 \
    --few_shot_number "${NUM_FEWSHOT}" \
    --seed ${seed} \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 1

    echo "Finished task: $task"
done 

echo "--- All evaluations complete ---"