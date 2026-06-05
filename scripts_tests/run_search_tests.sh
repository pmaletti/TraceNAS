CACHE_DIR="/scratch/gautschi/pmaletti/hf_cache/huggingface/"
export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

random_seeds=(42)

# Create and populate the seed_values.txt file
seed_file="${output_dir}/seed_values.txt"
echo "Random seeds used for this experiment:" > "${seed_file}"
for seed in "${random_seeds[@]}"; do
    echo "- ${seed}" >> "${seed_file}"
done
echo "Seed values saved to: ${seed_file}"
echo "----------------------------------------------------"
echo " "
num_moe=1
topk=1
lora_r=64

source_max_len=1024 #2048 #3072
target_max_len=3072 #6144 #1024

dataset=fineweb_edu #alpaca-gpt4
do_train=True #Functionality is nested within the transformers trainer. 

model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=meta-llama/Llama-3.1-8B
# model_name_or_path=Qwen/Qwen2.5-14B-Instruct

cache_dir=/scratch/gautschi/pmaletti/${model_name_or_path}
dataset_cache_dir=/scratch/gautschi/pmaletti/datasets/local_fineweb_edu
output_dir=/scratch/gautschi/pmaletti/tracenas/final_output/fineweb/block_prune/${model_name_or_path}/

if [[ "$model_name_or_path" == *"Llama-2-7b"* ]]; then
    # width_choice=$(seq -s, 0.4 0.05 0.6)
    # number_of_params_thresh="2.6e9,3.0e9"
    # min_num_layers=27

    # Layer pruning
    width_choice=1.0,1.0
    number_of_params_thresh="2.5e9,3.0e9"
    min_num_layers=11

    # block-wise pruning
    # width_choice=$(seq -s, 0.3 0.05 0.6)
    # number_of_params_thresh="2.5e9,3.6e9"
    # min_num_layers=32
elif [[ "$model_name_or_path" == *"Llama-3.1-8B"* ]]; then
    width_choice=$(seq -s, 0.4 0.05 1.0)
    number_of_params_thresh="4.55e9,4.7e9"
    min_num_layers=28
elif [[ "$model_name_or_path" == *"Qwen2.5-14B"* ]]; then
    width_choice=$(seq -s, 0.4 0.05 1.0)
    number_of_params_thresh="8.3e9,8.6e9"
    min_num_layers=42
else
    # Print to stderr and exit with an error code
    echo "Error: Model '$model_name_or_path' is not compatible." >&2
    echo "Please provide a known model and a valid parameter range." >&2
    exit 1
fi

echo "#################################################"
echo "Running search for $model_name_or_path in parameter range $number_of_params_thresh"
echo "#################################################"

# Create the base output directory if it doesn't exist
mkdir -p "${output_dir}"
echo "Search space ratios: $width_choice"

for seed in "${random_seeds[@]}"; do
    echo "Running with random seed: $seed"

    CUDA_LAUNCH_BLOCKING=1 python main.py --model_name_or_path ${model_name_or_path} \
        --output_dir ${output_dir} \
        --dataset ${dataset} \
        --dataset_format ${dataset} \
        --use_auth_token \
        --do_train ${do_train} \
        --bf16 --bits 16 \
        --source_max_len ${source_max_len} \
        --target_max_len ${target_max_len} \
        --gradient_accumulation_steps 4 \
        --seed ${seed} \
        --max_train_samples 16 \
        --enable_shrinking \
        --min_num_layer ${min_num_layers} \
        --shrinkable_width True \
        --width_choice ${width_choice} \
        --moe_num_expert ${num_moe} \
        --moe_topk ${topk} \
        --lora_r ${lora_r} \
        --cache_dir ${cache_dir} \
        --dataset_cache_dir ${dataset_cache_dir} \
        --get_influence True \
        --influence_type 'evol_search' \
        --full_determinism True \
        --per_device_train_batch_size 4 \
        --max_num_iters 50 \
        --random_init True \
        --number_of_params_thresh ${number_of_params_thresh} \
        --dataloader_num_workers 1 \
        --dataloader_prefetch_factor 1
        # --layer_pruning True
done
