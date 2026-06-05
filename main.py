from collections import defaultdict
import copy
import json
import os
os.environ['BNB_CUDA_VERSION'] = '126'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '500'
from os.path import exists, join, isdir
from dataclasses import dataclass, field, asdict # Import the necessary utility
import sys
from typing import Optional, Dict, Sequence, Tuple
import numpy as np
from tqdm import tqdm
import logging

import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse
import time
import random
import re
import torch.nn as nn
sys.path.insert(0, "./bitsandbytes/src")
sys.path.insert(0, "./transformers/src")
sys.path.insert(0, "./peft/src")
sys.path.insert(0, "./project-resq")
sys.path.insert(0, "attacc_simulator/")

from torch.utils.data import IterableDataset, get_worker_info
from transformers import default_data_collator

# from huggingface_hub import login
# login()

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
    enable_full_determinism,
    Trainer,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)

from transformers.models.llama.modeling_llama import NoAttention, NoMLP
import itertools


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, load_from_disk, interleave_datasets, Features, Value
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from eval_func import eval_mmlu, eval_mmlu_wrapper, eval_wikitext2_wrapper, eval_general_ppl_wrapper, eval_lm_eval_wrapper

import accelerate
from accelerate.utils import FullyShardedDataParallelPlugin

# Accelerate refactored this; we add it back for PEFT compatibility
if not hasattr(FullyShardedDataParallelPlugin, "get_module_class_from_name"):
    def get_module_class_from_name(module, name):
        """
        Gets a class from a module by name.
        """
        modules_children = list(module.children())
        if module.__class__.__name__ == name:
            return module.__class__
        elif len(modules_children) == 0:
            return
        else:
            for child in modules_children:
                module_class = get_module_class_from_name(child, name)
                if module_class is not None:
                    return module_class
                    
    # Inject the method back into the class
    FullyShardedDataParallelPlugin.get_module_class_from_name = classmethod(lambda cls, m, n: get_module_class_from_name(m, n))

import torch.distributed as dist
from datetime import timedelta
if "WORLD_SIZE" in os.environ:
    dist.init_process_group(
        backend="nccl", 
        timeout=timedelta(hours=5)  # <--- ADD THIS
    )
from contextlib import contextmanager
from itertools import islice

from torch.optim import AdamW
from transformers import get_wsd_schedule

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

cls_tasks = {
    'classification': ['sst2', 'sst5', 'MR', 'SUBJ', 'AGNews', 'TREC', 'CB', 'BoolQ'], # , 'DBPedia'],
    'multiple choice': ['hellaswag', 'ARCE', 'PIQA', 'ARCC', 'OB', 'COPA', 'CQA'],
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
# Seq2Seq
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    
    do_eval_wikitext2: bool = field(
        default=False, metadata={"help": "evaluate the ppl on wikitext2."}
    )

    do_lm_eval: Optional[bool]=field(
        default=False, 
        metadata={"help":"Evalute on lm-eval-harness."}
    )
    
    do_lm_eval_task : str = field(
        default="arc_easy,piqa,sciq", metadata={"help": "Evaluation tasks in lm-eval-harness."}
    )
    
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    dataset_cache_dir:  str = field(default=False, metadata={"help": "The path to the dir containing the dataset"})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=2, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    no_eval_orig: bool = field(default=False, metadata={"help": 'do not eval the original test dataset corresponding to the training dataset'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    stable_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    decay_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=2, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    full_determinism: bool = field(default=False, metadata={"help": 'enable full determinism for reproducibility in trainer.'})
    few_shot_number: int = field(default=0, metadata={"help": 'few shot numbers for classification tasks'})
    dataloader_num_workers: int= field(default=0, metadata={"help": 'dataloader number of prefetched batches'})
    dataloader_prefetch_factor: int= field(default=3, metadata={"help": 'dataloader num workers, set to 1 for reproducibility'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": 'enable ddp_find_unused_parameters in Accelerator.'})
    seed: int = field(default=42, metadata={"help": 'Random seed for reproducibility'})
    enable_shrinking: bool = field(default=False, metadata={"help": 'Enable shrinkable LLM.'})
    
    shrinkable_width: bool = field(default=False, metadata={"help": 'Enable shrinkable width in addition to layers.'})
    width_choice: str = field(default='[1,7/8,3/4,5/8,1/2]', metadata={"help": 'the available width choices for shrinkable width.'})
    nonuniform_width: bool = field(default=False, metadata={"help": 'Training with nonuniform width across layers.'})
    first_width: bool = field(default=False, metadata={"help": 'An ablation study: Only active the first widths (channels) in each layer.'})

    min_num_layer: int = field(default=16, metadata={"help": 'The minimal number of layers.'})
    random_sample_num_layer: int = field(default=2, metadata={"help": 'The number of randomely sampled layers in each iteration.'})
    kd_weight: float = field(default=1, metadata={"help": 'weight coefficient of the KD loss.'})

    sample_per_dataset: int = field(default=2000, metadata={"help": 'samples per dataset when training on cls_combo and mc_combo.'})

    num_remain_layers: int = field(default=1, metadata={"help": 'number of final layers remained during layer skipping.'})

    distill_all_tokens: bool = field(default=False, metadata={"help": 'Distill both target and context tokens to small models.'})
    
    layer_pruning: bool = field(default=None, metadata={"help": 'whether to apply layer pruning.'})
        
    distill_method: str = field(default='sp', metadata={"help": 'distillation method: sp, gkd, atkd.'})

    unc_thres: float = field(default=0.5, metadata={"help": 'the threshold for the uncertainty coefficient in ATKD.'})

    layer_calib_dp: bool = field(default=False, metadata={"help": 'enable the calibration based on dynamic programming to get layer ranking.'})

    dp_keep_last_layer: int = field(default=-1, metadata={"help": 'the last n layers to remain during dynamic programming.'})

    calib_dataset: str = field(default='wikitext2', metadata={"help": 'the dataset used for calibration.'})

    calib_metric: str = field(default=None, metadata={"help": 'the metric for calibration.'})

    width_calib: bool = field(default=False, metadata={"help": 'enable the calibration to get width ranking.'})

    prune_width_dim: str = field(default='in', metadata={"help": 'the width pruning dimension: {in, out}.'})

    prune_width_method: str = field(default='flap', metadata={"help": 'width pruning method: {wanda, flap}.'})

    wanda_sp: bool = field(default=False, metadata={"help": 'An ablation study to use wand-sp for pruning.'})

    num_calib_sample: int = field(default=20, metadata={"help": 'number of samples used for calibration.'})

    shrinking_method: str = field(default='first_layers', metadata={"help": 'the way to perform layer shrinking: {first_layers, calib, calib_dp}.'})
    
    shrinking_file: str = field(default=None, metadata={"help": 'the path to the file specifying the shrinking configuration.'})

    use_moe_lora: bool = field(default=False, metadata={"help": 'Use mixture of LoRA.'})
    use_moe_lora_coeff: bool = field(default=False, metadata={"help": 'Use mixture of LoRA.'})
    
    moe_num_expert: int = field(default=5, metadata={"help": 'number of experts in MoE.'})

    moe_topk: int = field(default=2, metadata={"help": 'topk in MoE.'})

    resume_training: bool = field(default=False, metadata={"help": 'resume training from the latest checkpoint.'})
    
    distill_steps: int = field(default=-1, metadata={"help": 'number of training steps that enable distillation.'})

    no_balancing: bool = field(default=False, metadata={"help": 'ablation study: do not use loss balancing.'})
    
    eval_num_layer: int = field(default=24, metadata={"help": 'number of layers for evaluation.'})

    eval_num_width: float = field(default=0.875, metadata={"help": 'width for evaluation.'})

    get_influence: bool = field(default=False, metadata={"help": 'get per small model influence on a validation set.'})
    
    influence_type: str = field(default='data', metadata={"help": 'what estimator to use: Conjugate gradient (CG) or LiSSA.'})
    
    eval_batch_size: int = field(default=2, metadata={"help": 'The validation batch size.'})
    
    max_num_iters: int = field(default=100, metadata={"help": 'The number of search iterations.'})
    
    no_ft_infl: bool = field(default=False, metadata={"help": 'whether to use ft model for infl calculation'})
    
    eval_after_evolsearch: bool = field(default=False, metadata={'help': 'whether eval is happening after evolution search, only to handle missing bias masks'})
    
    lora_shrinkable_width: bool = field(default=False, metadata={"help": 'Enable lora shrinkable width in addition to layers.'})
    
    number_of_params_thresh: str = field(default=None, metadata={"help": 'Range of model sizes needed'})
    
    predict_with_generate: bool = field(default=False, metadata={"help": "Whether to use generate() for prediction."})
    
    hard_prune: bool = field(default=False, metadata={"help": "Whether to hard prune the model or not."})
    
    custom_model_eval: bool = field(default=False, metadata={"help": "Load custom model for evaluation."})
    
    dispatch_batches:  bool = field(default=False, metadata={"help": "Load custom model for evaluation."})
    
    split_batches:  bool = field(default=False, metadata={"help": "Load custom model for evaluation."})
    
    hard_pruned_dir:  str = field(default=False, metadata={"help": "Get healed model for evaluation."})

    kaiming_init:  bool = field(default=False, metadata={"help": "Load lora_B with kaiming init."})

    

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)

    lora_module_names = []
    
    for name, module in model.named_modules():
        # 1. Check if it is a valid Linear layer (excludes NoIntermediate/NoAttention)
        if isinstance(module, cls):
            # 2. Exclude the output head if necessary
            if 'lm_head' in name: 
                continue
            
            # 3. Append the FULL path, not just the short name
            lora_module_names.append(name)

    return lora_module_names

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def prune_linear_layer(layer, mask, dim=0):
    """
    Helper to physically slice a Linear layer's weights/bias using 1D or 2D masks.
    dim=0: Prune output dimension (e.g., up_proj, q_proj)
    dim=1: Prune input dimension (e.g., down_proj, o_proj)
    """
    weight = layer.weight
    
    # --- 1. Handle 2D Masks (Collapse to 1D) ---
    if mask.dim() > 1:
        # Heuristic: Find which dimension has the sparsity pattern.
        # Your masks are created via broadcasting, so one dim is identical.
        
        # Check if rows are identical (variation is in columns)
        rows_identical = torch.all(mask[0, :] == mask[-1, :])
        
        if rows_identical:
            # The mask pattern is in the columns (dim 1)
            mask_1d = mask[0, :]
        else:
            # The mask pattern is in the rows (dim 0)
            mask_1d = mask[:, 0]

        # Verify shape compatibility with the target dimension
        if mask_1d.shape[0] != weight.shape[dim]:
            # Fallback: If we picked the wrong axis and the other one fits, swap.
            # (Matches cases where mask is transposed relative to weight)
            if rows_identical and mask.shape[0] == weight.shape[dim]:
                mask_1d = mask[:, 0] # Force pick dim 0
            elif not rows_identical and mask.shape[1] == weight.shape[dim]:
                mask_1d = mask[0, :] # Force pick dim 1
            else:
                # If neither fits, we can't prune
                raise ValueError(f"Mask shape {mask.shape} collapsed to {mask_1d.shape} "
                                 f"does not match layer dim {dim} size {weight.shape[dim]}")
        
        mask = mask_1d

    # Ensure boolean
    mask = mask.to(device=weight.device, dtype=torch.bool)

    # --- 2. Slice Weight & Bias ---
    if dim == 0:
        # Pruning Output Features (Rows)
        new_weight = weight.data[mask, :]
        new_bias = layer.bias.data[mask] if layer.bias is not None else None
        
        out_features = int(mask.sum())
        in_features = layer.in_features
    else:
        # Pruning Input Features (Columns)
        new_weight = weight.data[:, mask]
        new_bias = layer.bias.data if layer.bias is not None else None 
        
        out_features = layer.out_features
        in_features = int(mask.sum())

    # --- 3. Create New Layer ---
    new_layer = nn.Linear(in_features, out_features, bias=new_bias is not None)
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias
    
    return new_layer.to(weight.device, dtype=weight.dtype)
    
def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    elif is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
    else:
        n_gpus =  1
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    max_memory = {'': max_memory[local_rank]}

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    shrink_config = {'enable_shrinking': args.enable_shrinking, 
                     "shrinkable_width": args.shrinkable_width,
                     "lora_shrinkable_width": args.lora_shrinkable_width,
                     "shrinking_method": args.shrinking_method,
                     "shrinking_file": args.shrinking_file,
                     "mask_dtype": "torch.float16" if args.fp16 else ("torch.bfloat16" if args.bf16 else "torch.float32")}
    q_config =BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ) if (not args.full_finetune and (args.bits==4 or args.bits==8)) else None
    
    if not args.get_influence:
        if args.custom_model_eval:
            import transformers.modeling_utils
            import transformers.pytorch_utils

            # Monkey-patch: Inject the missing function back into modeling_utils 
            # so the legacy custom code can find it.
            if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
                transformers.modeling_utils.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer

            if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
                transformers.modeling_utils.find_pruneable_heads_and_indices = transformers.pytorch_utils.find_pruneable_heads_and_indices
            # Load the model
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}
            
            if checkpoint_dir is not None:
                    model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_dir,
                    trust_remote_code=True, 
                    device_map=device_map, # Optional: helps if you are on a GPU
                    cache_dir=args.cache_dir,
                    torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                    shrink_config = shrink_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    trust_remote_code=True, 
                    device_map=device_map, # Optional: helps if you are on a GPU
                    cache_dir=args.cache_dir,
                    torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                    shrink_config = shrink_config
                )

        elif not args.hard_prune:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=q_config if not args.full_finetune else None,
                torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                shrink_config = shrink_config
            )
        else:
            # Default behavior
            if args.hard_pruned_dir:
                pruned_save_path = args.hard_pruned_dir
            else:
                pruned_save_path = os.path.join(args.output_dir, "hard_pruned_ckpt")

            if os.path.exists(pruned_save_path):
                if checkpoint_dir is not None and args.full_finetune:
                    print(f'loading from checkpoint dir: {checkpoint_dir}')
                    config = AutoConfig.from_pretrained(checkpoint_dir)
                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_dir,
                        config=config,
                        cache_dir=args.cache_dir,
                        device_map=device_map,
                        max_memory=max_memory,
                        quantization_config=q_config if not args.full_finetune else None,
                        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                        trust_remote_code=args.trust_remote_code,
                        use_auth_token=args.use_auth_token,
                        shrink_config = shrink_config
                    )
                else:
                    print(f"Loading hard-pruned model from {pruned_save_path}...")
                    config = AutoConfig.from_pretrained(pruned_save_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        pruned_save_path,
                        config=config,
                        cache_dir=args.cache_dir,
                        device_map=device_map,
                        max_memory=max_memory,
                        quantization_config=q_config if not args.full_finetune else None,
                        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                        trust_remote_code=args.trust_remote_code,
                        use_auth_token=args.use_auth_token,
                        shrink_config = shrink_config
                    )

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    cache_dir=args.cache_dir,
                    device_map=device_map,
                    max_memory=max_memory,
                    # quantization_config=q_config if not args.full_finetune else None,
                    torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                    trust_remote_code=args.trust_remote_code,
                    use_auth_token=args.use_auth_token,
                    shrink_config = shrink_config
                )

                print("Performing hard pruning of the model...")
                # 1. Setup
                shrink_file = np.load(args.shrinking_file, allow_pickle=True).item()
                strategy = shrink_file['strategy']
                width_mask = shrink_file['meta_width_mask']

                is_pythia = "GPTNeoX" in model.config.architectures[0]

                strategy_key = (args.eval_num_layer, args.eval_num_layer, args.eval_num_width)
                if strategy_key not in strategy:
                    print(f"Warning: Key {strategy_key} not found. Using first available key.")
                    strategy_key = list(strategy.keys())[0]

                active_layers_indices = np.where(strategy[strategy_key]['active_layers'][0])[0]

                num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
                # 2. PERFORM HARD WIDTH PRUNING
                print("Performing Hard Width Pruning...")
                is_gqa = num_key_value_heads < model.config.num_attention_heads
                prefix = 'layers'

                for name, module in model.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue

                    match = re.search(r'layers\.(\d+)\.(self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|mlp\.up_proj|mlp\.gate_proj|mlp\.down_proj|self_attn\.o_proj|attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)$', name)
                    if not match:
                        continue

                    layer_id = int(match.group(1))
                    
                    # --- CRITICAL: Awareness of Ghost Layers ---
                    # We skip width pruning for any layer that is ALREADY a NoAttention or NoMLP module
                    if is_pythia:
                        target_layer = model.gpt_neox.layers[layer_id]
                    else:
                        target_layer = model.model.layers[layer_id]
                    if "self_attn" in name and isinstance(getattr(target_layer, 'self_attn', None), NoAttention):
                        continue
                    if "attention" in name and isinstance(getattr(target_layer, 'attention', None), NoAttention):
                        continue
                    if "mlp" in name and isinstance(getattr(target_layer, 'mlp', None), NoMLP):
                        continue

                    layer_type = match.group(2)
                    mask_key_name = f'{prefix}.{layer_id}.{layer_type}'
                    target_mask = None
                    prune_dim = 0 

                    # Case A: MLP Width
                    if 'mlp.up_proj' in name or 'mlp.gate_proj' in name:
                        target_mask = width_mask.get(mask_key_name, {}).get(strategy_key)
                        prune_dim = 0
                    elif 'mlp.down_proj' in name:
                        sibling_key = f'{prefix}.{layer_id}.mlp.up_proj'
                        target_mask = width_mask.get(sibling_key, {}).get(strategy_key)
                        prune_dim = 1

                    elif 'mlp.dense_h_to_4h' in name:
                        target_mask = width_mask.get(mask_key_name, {}).get(strategy_key)
                        prune_dim = 0 # Prune output features

                    elif 'mlp.dense_4h_to_h' in name:
                        sibling_key = f'{prefix}.{layer_id}.mlp.dense_h_to_4h'
                        target_mask = width_mask.get(sibling_key, {}).get(strategy_key)
                        prune_dim = 1 # Prune input features
                        
                    # Case B: Attention Width
                    elif 'self_attn.q_proj' in name:
                        target_mask = width_mask.get(mask_key_name, {}).get(strategy_key)
                        prune_dim = 0
                    elif ('self_attn.k_proj' in name or 'self_attn.v_proj' in name) and not is_gqa:
                        target_mask = width_mask.get(mask_key_name, {}).get(strategy_key)
                        prune_dim = 0
                    elif 'self_attn.o_proj' in name:
                        sibling_type = 'self_attn.q_proj' if is_gqa else 'self_attn.v_proj'
                        sibling_key = f'{prefix}.{layer_id}.{sibling_type}'
                        target_mask = width_mask.get(sibling_key, {}).get(strategy_key)
                        prune_dim = 1

                    elif 'attention.query_key_value' in name:
                        target_mask = width_mask.get(mask_key_name, {}).get(strategy_key)
                        prune_dim = 0
                    elif 'attention.dense' in name:
                        # sibling_key = f'{prefix}.{layer_id}.attention.query_key_value'
                        # target_mask = width_mask.get(sibling_key, {}).get(strategy_key)
                        # prune_dim = 1
                        sibling_key = f'{prefix}.{layer_id}.attention.query_key_value'
                        mask_fused = width_mask.get(sibling_key, {}).get(strategy_key)
                        
                        if mask_fused is not None:
                            # 1. Collapse the fused 2D mask to 1D based on output rows
                            # mask_fused is [7680, 2560]. We take the first column to get the row-pattern.
                            mask_1d_fused = mask_fused[:, 0].bool() 
                            
                            # 2. Slice for just ONE segment (Pythia maps Head_Dim -> Hidden_Size)
                            segment_size = mask_1d_fused.shape[0] // 3
                            target_mask = mask_1d_fused[:segment_size]
                            
                            prune_dim = 1 # Prune the INPUT dimension

                    if target_mask is not None:
                        if isinstance(target_mask, np.ndarray):
                            target_mask = torch.from_numpy(target_mask)
                        target_mask = target_mask.to(module.weight.device)
                        
                        new_layer = prune_linear_layer(module, target_mask, dim=prune_dim)
                        
                        parent_name = name.rsplit('.', 1)[0]
                        child_name = name.rsplit('.', 1)[1]
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, child_name, new_layer)

                # 3. PERFORM HARD DEPTH PRUNING
                print("Performing Hard Depth Pruning (Replacing inactive layers)...")
                if is_pythia:
                    old_layers = model.gpt_neox.layers
                else:
                    old_layers = model.model.layers
                new_layers_list = []

                # Storage for config synchronization
                intermediate_sizes = []
                hidden_sizes = []
                num_attention_heads = []

                for i in range(len(old_layers)):
                    layer = old_layers[i]
                    
                    # Awareness check: Is this layer already a ghost module?
                    is_already_ghost_attn = isinstance(getattr(layer, 'self_attn', None), NoAttention) or isinstance(getattr(layer, 'attention', None), NoAttention)
                    is_already_ghost_mlp = isinstance(getattr(layer, 'mlp', None), NoMLP) or isinstance(getattr(layer, 'mlp', None), NoMLP)

                    if i in active_layers_indices and not (is_already_ghost_attn and is_already_ghost_mlp):
                        if is_pythia:
                            # ACTIVE: Keep layer and record its actual pruned dimensions
                            h_size = layer.attention.query_key_value.out_features // 3 if not is_already_ghost_attn else 0
                            i_size = layer.mlp.dense_h_to_4h.out_features if not is_already_ghost_mlp else 0
                        else:
                            # ACTIVE: Keep layer and record its actual pruned dimensions
                            h_size = layer.self_attn.q_proj.out_features if not is_already_ghost_attn else 0
                            i_size = layer.mlp.up_proj.out_features if not is_already_ghost_mlp else 0
                        
                        hidden_sizes.append(h_size)
                        intermediate_sizes.append(i_size)
                        
                        head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
                        num_attention_heads.append(h_size // head_dim if h_size > 0 else 0)
                        
                        new_layers_list.append(layer)
                    else:
                        # INACTIVE: Replace with Ghost Modules and set config to 0
                        print(f"  -> Layer {i} set to 0 in config.")
                        layer.self_attn = NoAttention()
                        layer.mlp = NoMLP()
                        
                        hidden_sizes.append(0)
                        intermediate_sizes.append(0)
                        num_attention_heads.append(0)
                        
                        new_layers_list.append(layer)

                # Assign and sync config
                if is_pythia:
                    model.gpt_neox.layers = nn.ModuleList(new_layers_list)
                else:
                    model.model.layers = nn.ModuleList(new_layers_list)
                model.config.num_hidden_layers = len(new_layers_list)
                model.config.intermediate_size_list = intermediate_sizes
                model.config.hidden_size_list = hidden_sizes
                model.config.num_attention_heads_list = num_attention_heads

                # 4. SAVE
                print(f"Saving hard-pruned model to {pruned_save_path}...")
                model.save_pretrained(pruned_save_path)
    
    else:
        attn_impl = "eager" if "gemma" in args.model_name_or_path.lower() else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=q_config,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            attn_implementation=attn_impl,
            shrink_config = shrink_config
        )

    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    # if not args.get_influence:
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True, 
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )

    # 2. Smart Padding Logic (Universal)
    # If the model has no pad token, we must assign one.
    if tokenizer.pad_token is None:
        # Llama 2 Strategy: Use UNK if available
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        # Llama 3 / Qwen Strategy: Fallback to EOS (standard practice)
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"Setting pad_token to: {tokenizer.pad_token}")

    # 1. Sync Model Config (Critical for Training)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 2. Sync Generation Config (Critical for Inference/Saving)
    # Most Llama 3 / Qwen models have a separate generation config that needs updating
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    print(f"Synced Model Config Pad Token ID: {model.config.pad_token_id}")

    # 5. SAFETY CHECK: Resize if vocab sizes differ (Fixes your crash)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))
    
    if not args.full_finetune:
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if args.bits in [4, 8]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        
        # CASE B: Standard LoRA (BF16/FP16)
        else:
            # For standard LoRA, we just need to enable gradient checkpointing manually
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Standard LoRA requires inputs to require grads for checkpointing to work
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    if not args.full_finetune:
        if checkpoint_dir is not None and not args.no_ft_infl and os.path.exists(join(checkpoint_dir, 'adapter_model')):
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)

        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                
                use_moe_lora=args.use_moe_lora,
                use_moe_lora_coeff=args.use_moe_lora_coeff,
                kaiming_init=args.kaiming_init,
                num_experts=args.moe_num_expert,
                top_k=args.moe_topk,
                width_choice=args.width_choice if args.shrinkable_width else None,
            )
            print(args.width_choice)
            if args.get_influence:
                torch.manual_seed(args.seed)
            model = get_peft_model(model, config)

    if not args.full_finetune and not args.do_train and args.do_eval and args.hard_prune:
        model = model.merge_and_unload()
        print('here: merging model')
     
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            # module = module.to(torch.float32)
            module = module.to(torch.bfloat16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    name_list=[]
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            name_list.append(name)

    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

@dataclass
class DataCollatorForCausalLM(object):
    """
    Data Collator for Causal Language Modeling (Instruction Tuning).
    
    Recommended for Reasoning Tasks (OpenOrca, CoT):
    - source_max_len: 1024 (For System Prompt + Question)
    - target_max_len: 3072 (For long Chain-of-Thought reasoning)
    - train_on_source: False (Mask the prompt so we only learn the answer)
    """
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Handle cases where BOS/EOS might be None (common in some new tokenizers)
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""

        # Extract elements
        # MODIFIED: flexible key access for 'response' (Orca) or 'output' (Alpaca)
        sources = [f"{bos_token}{example['input']}" for example in instances]
        targets = [f"{example.get('response', example.get('output', ''))}{eos_token}" for example in instances]
        
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        input_ids = []
        labels = []
        source_ids = [] # useful for debugging or specific metrics

        for tokenized_source, tokenized_target in zip(
            tokenized_sources['input_ids'],
            tokenized_targets['input_ids']
        ):
            # Ensure we are working with lists of ints
            src_ids = tokenized_source
            tgt_ids = tokenized_target

            if not self.predict_with_generate:
                # Training Mode: Concatenate Source + Target
                full_input = torch.tensor(src_ids + tgt_ids)
                input_ids.append(full_input)

                if not self.train_on_source:
                    # Mask the source tokens (Instruction Tuning behavior)
                    # We use IGNORE_INDEX (-100) so the loss function skips these tokens
                    label_mask = [IGNORE_INDEX] * len(src_ids)
                    labels.append(torch.tensor(label_mask + tgt_ids))
                else:
                    # Train on everything (Standard Pretraining behavior)
                    labels.append(torch.tensor(copy.deepcopy(src_ids + tgt_ids)))
            else:
                # Inference Mode: Source only
                input_ids.append(torch.tensor(src_ids))
            
            # Keep track of source IDs if needed
            source_ids.append(torch.tensor(src_ids))

        # --- PAD SEQUENCES ---
        
        if self.predict_with_generate:
            # CRITICAL FIX: Manually Left-Pad for Inference
            # We flip, pad (which adds to the right), then flip back
            input_ids = [t.flip(0) for t in input_ids]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            input_ids = input_ids.flip(1)
            labels = None
        else:
            # Standard Right-Pad for Training
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
            
        return data_dict

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

ORCA_PROMPT_DICT = {
    # OpenOrca usually always has a system prompt, so this is the primary format.
    # We map 'system_prompt' to a System header, and 'question' to the Instruction.
    "prompt_with_system": (
        "### System:\n{system_prompt}\n\n"
        "### Instruction:\n{question}\n\n"
        "### Response: "
    ),
    # Fallback in rare cases where system_prompt might be empty
    "prompt_no_system": (
        "### Instruction:\n{question}\n\n"
        "### Response: "
    ),
}

def extract_orca_dataset(example):
    # CRITICAL FIX 1: Handle None values safely
    # (example.get returns None if key exists but value is null, so we use 'or ""')
    system_prompt = example.get("system_prompt") or ""
    
    if system_prompt != "":
        # We manually update the example dict so we don't rely on **unpacking with None
        formatted_input = ORCA_PROMPT_DICT["prompt_with_system"].format(
            system_prompt=system_prompt, 
            question=example["question"]
        )
    else:
        formatted_input = ORCA_PROMPT_DICT["prompt_no_system"].format(
            question=example["question"]
        )
        
    # CRITICAL FIX 2: Return the whole example with the new column added.
    # If you return just {'input': ...}, you delete the 'response' column!
    example['input'] = formatted_input
    
    return example

class PackedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        worker_info = get_worker_info()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        total_shards = world_size * num_workers
        my_shard_id = (rank * num_workers) + worker_id

        # --- Lazy Sharding ---
        if hasattr(self.dataset, "shard"):
            # If it's a Map-style dataset, native sharding is efficient
            iterator = iter(self.dataset.shard(num_shards=total_shards, index=my_shard_id))
        else:
            # Fallback for IterableDatasets
            iterator = islice(iter(self.dataset), my_shard_id, None, total_shards)

        buffer = []
        for example in iterator:
            score = example.get("score", 0)
            
            # if score < 3.5:
            #     continue # Skip this example entirely

            # --- ON-THE-FLY TOKENIZATION ---
            # Extract text and convert to IDs here
            text = example["text"]
            # Mimicking your batch_tokenize logic: No special tokens + Manual EOS
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids += [self.tokenizer.eos_token_id]
            
            buffer.extend(input_ids)
            
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long)
                }

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        if dataset_name == 'alpaca-gpt4':
            return load_dataset("vicgalle/alpaca-gpt4")
        if dataset_name == 'Open-Orca/OpenOrca':
            dataset = load_dataset(
                "Open-Orca/OpenOrca", 
                split="train", 
                streaming=True
            )
            dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
            return dataset
        
        if dataset_name == 'c4':
            return load_dataset("c4", 'en')
        
        if dataset_name == 'redpajama':
            return load_dataset(
                    "togethercomputer/RedPajama-Data-1T", 
                    "default",  # <--- Required config name
                    streaming=True, 
                    trust_remote_code=True # <--- Required for this dataset's custom script
                )
        
        if dataset_name == 'slimpajama':
            return load_dataset(
                    "cerebras/SlimPajama-627B", 
                    "default",  # <--- Required config name
                    streaming=True, 
                    split="train",
                    trust_remote_code=True # <--- Required for this dataset's custom script
                ).shuffle(seed = args.seed, buffer_size=10000)
        
        if dataset_name == 'fineweb_edu':
            # if os.path.exists(args.dataset_cache_dir):
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-100BT",
                split="train",
                streaming=False,
                cache_dir=args.dataset_cache_dir,
            ).shuffle(seed=args.seed)

            if args.get_influence:
                dataset = dataset.select(range(args.max_train_samples))

            return dataset
            
            # else:
            #     # Only works for training, search is very slow with streaming datasets
            #     dataset = load_dataset(
            #         "HuggingFaceFW/fineweb-edu",
            #         "sample-100BT",
            #         split="train",
            #         streaming=True,
            #     ).shuffle(seed=args.seed)

            #     if args.get_influence:
            #         return ValueError('Please download dataset to ensure best speed')

        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'lamini':
            return load_dataset("MBZUAI/LaMini-instruction")

        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or dataset_format == 'alpaca-gpt4' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean', 'alpaca-gpt4', 'yahma/alpaca-cleaned'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
       
        elif (dataset_format == 'Open-Orca/OpenOrca' or (dataset_format is None and args.dataset in ['Open-Orca/OpenOrca'])):
            try:
                sample_item = next(iter(dataset))
                all_cols = list(sample_item.keys())
            except StopIteration:
                all_cols = [] # Should not happen if dataset is valid

            cols_to_remove = [col for col in all_cols if col not in ['input', 'output', 'response']]
            
            dataset = dataset.map(
                extract_orca_dataset, 
                remove_columns=cols_to_remove
            )

        elif dataset_format == 'c4' or (dataset_format is None and args.dataset == 'c4'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'redpajama' or (dataset_format is None and args.dataset == 'redpajama'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format in ['slimpajama', 'fineweb_edu', 'dclm', 'bigcodebench'] or \
            (dataset_format is None and args.dataset in ['slimpajama', 'fineweb_edu', 'dclm', 'bigcodebench']):
            dataset = dataset

        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        
        elif args.dataset in ['cls_combo', 'mc_combo']:
            dataset = dataset.map(lambda x: {
                'input': x['text'],
                'output': x['label'].strip(),
            })
            dataset = DatasetDict({"train": dataset})
            
        elif dataset_format == 'input-output':
            # leave as is
            pass

        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)

    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    # if not args.get_influence:
    is_streaming = args.dataset in ['redpajama', 'slimpajama', 'fineweb_edu', 'Open-Orca/OpenOrca'] or getattr(dataset, "streaming", False) #,

    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            if is_streaming:
                print('Streaming mode: reserving first 1000 samples for validation')

                if args.dataset in ['Open-Orca/OpenOrca']:
                    # 2. manually split for streaming:
                    # Reserve the first N samples for validation/test
                    
                    eval_dataset = dataset.take(args.eval_dataset_size or 1000)
                    dataset = dataset.skip(args.eval_dataset_size or 1000)

                else:
                
                    # Take the first N for eval
                    eval_dataset = dataset["train"].take(args.eval_dataset_size or 1000)
                    # Skip those N for training to avoid leakage
                    dataset["train"] = dataset["train"].skip(args.eval_dataset_size or 1000)
            else:
                print('Splitting train dataset in train and validation')
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=42
                )
                eval_dataset = dataset['test']
            
        if args.max_eval_samples is not None: # and len(eval_dataset) > args.max_eval_samples:
            if is_streaming:
                eval_dataset = eval_dataset.take(args.max_eval_samples)
            elif len(eval_dataset) > args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    
    if args.do_train:
        if args.dataset in ['Open-Orca/OpenOrca']: #, 'fineweb_edu']: #
            train_dataset = dataset
        elif args.dataset in ['fineweb_edu', 'bigcodebench']:
            train_dataset = PackedIterableDataset(dataset, tokenizer, args.target_max_len)
        else:
            train_dataset = dataset['train']

        if args.max_train_samples is not None: # and len(train_dataset) > args.max_train_samples:
            if is_streaming:
                if args.dataset not in ['fineweb_edu', 'bigcodebench']:
                    # Streaming shuffle needs a buffer_size
                    train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=10000)
                    train_dataset = train_dataset.take(args.max_train_samples)
            elif len(train_dataset) > args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))

        if args.group_by_length and args.dataset not in ['fineweb_edu']:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if args.dataset in ['redpajama', 'slimpajama', 'fineweb_edu']:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    from os.path import join, isdir, exists, abspath
    # 1. Force absolute path to ensure we know exactly where we are looking
    checkpoint_dir = abspath(checkpoint_dir)
    print(f"STRICT CHECK: Scanning only inside: {checkpoint_dir}")

    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        
        max_step = 0
        final_checkpoint_path = None
        
        # os.listdir ONLY looks at direct children. It does not look at parents.
        for filename in os.listdir(checkpoint_dir):
            full_path = join(checkpoint_dir, filename)
            
            if isdir(full_path) and filename.startswith('checkpoint-'):
                try:
                    step = int(filename.replace('checkpoint-', ''))
                    
                    # 2. VALIDATION: Check if this folder actually has model files.
                    # This prevents loading an empty folder created by a crash.
                    has_files = (
                        exists(join(full_path, "adapter_config.json")) or 
                        exists(join(full_path, "config.json")) or
                        exists(join(full_path, "model.safetensors")) or
                        exists(join(full_path, "pytorch_model.bin"))
                    )

                    if has_files:
                        if step > max_step:
                            max_step = step
                            final_checkpoint_path = full_path
                    else:
                        print(f"Skipping empty/corrupt checkpoint found: {filename}")

                except ValueError:
                    continue 

        if final_checkpoint_path:
            print(f"Found valid checkpoint: {final_checkpoint_path}")
            return final_checkpoint_path, is_completed
            
    print(f"No valid checkpoint found in {checkpoint_dir}")
    return None, False

def get_max_training_tokens(args):
    # 1. Get World Size (Number of GPUs)
    # If not distributed, default to 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
        # Or check args if you pass it manually:
        # world_size = getattr(args, 'world_size', 1) 

    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    if args.dataset=='fineweb_edu': 
        MAX_SEQ_LENGTH = args.target_max_len #args.source_max_len +
    else:
        MAX_SEQ_LENGTH = args.target_max_len + args.source_max_len
    
    # FIX: Multiply by World Size
    EFFECTIVE_BATCH_SIZE = per_device_train_batch_size * gradient_accumulation_steps * world_size
    
    TOKENS_PER_OPTIMIZATION_STEP = EFFECTIVE_BATCH_SIZE * MAX_SEQ_LENGTH
    OPTIMIZATION_STEPS = args.max_steps

    total_tokens_seen = OPTIMIZATION_STEPS * TOKENS_PER_OPTIMIZATION_STEP

    print(f"--- Token Calculation Summary ---")
    print(f"GPUs (World Size): {world_size}")
    print(f"Per-Device Batch Size: {per_device_train_batch_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"Effective Batch Size (Sequences): {EFFECTIVE_BATCH_SIZE}")
    print(f"Max Sequence Length (Tokens): {MAX_SEQ_LENGTH}")
    print(f"Total Optimization Steps: {OPTIMIZATION_STEPS}")
    print("-" * 35)
    print(f"Tokens Processed per Step: {TOKENS_PER_OPTIMIZATION_STEP:,}")
    print(f"Total Tokens Seen: {total_tokens_seen:,} tokens")
      
def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # training_args.data_args_dict = asdict(data_args)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    training_args.max_train_samples = args.max_train_samples
    training_args.source_max_len = args.source_max_len
    training_args.target_max_len = args.target_max_len
    training_args.accelerator_config.dispatch_batches = args.dispatch_batches
    training_args.accelerator_config.split_batches =  args.split_batches  # You can set other params here too
    training_args.model_name_or_path = args.model_name_or_path

    if args.lr_scheduler_type=='warmup_stable_decay':
        lr_scheduler_kwargs={
            "num_decay_steps": int(args.decay_ratio * args.max_steps),
            "min_lr_ratio": 0.0,    # Updated to 0.0 for your D2Z strategy
            "decay_type": "linear", # Changed to linear for the 58% push
        }
        training_args.lr_scheduler_kwargs = lr_scheduler_kwargs
    
    elif args.lr_scheduler_type == 'CustomWSDScheduler':
        total_steps = args.max_steps
        lr_scheduler_kwargs = {
            "num_warmup_steps": int(total_steps * (args.warmup_ratio * 0.3)),    # Warm up to 1e-5
            "num_stable_steps_1": int(total_steps * (args.stable_ratio * 0.3)), # Stay at 1e-5
            "num_warmup_steps_2": int(total_steps * (args.warmup_ratio * 0.7)), # Warm up to 1e-4
            "num_stable_steps_2": int(total_steps * (args.stable_ratio * 0.7)), # Stay at 1e-4
            "num_decay_steps": int(total_steps * args.decay_ratio),    # Decay to 0
            "lr_1": 1e-5,
            "lr_2": args.learning_rate,
            "min_lr": 1e-7
        }
        training_args.lr_scheduler_kwargs = lr_scheduler_kwargs

    print(args)

    print('Shrinking file:', args.shrinking_file)
    if not args.get_influence:
        set_seed(args.seed)
    else:
        from accelerate.utils import set_seed as accelerate_set_seed

        set_seed(args.seed, deterministic=True)
        enable_full_determinism(args.seed, warn_only=True)
        accelerate_set_seed(args.seed)

        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # or ":16:8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')

    if not not args.get_influence and args.full_finetune:
        get_max_training_tokens(args)
    
    if args.shrinkable_width and not args.get_influence:
        print('Setting width mask and bias...')
        shrink_file = np.load(args.shrinking_file, allow_pickle=True).item()
        if 'meta_width_mask' in shrink_file:
            width_mask = shrink_file['meta_width_mask']
        elif 'width_mask' in shrink_file:
            width_mask = shrink_file['width_mask']
                        
        if args.first_width:
            for key, mask_dict in width_mask.items():
                for ratio, mask in mask_dict.items():
                    width_mask[key][ratio] = np.sort(mask)[::-1]
                
        for name, module in model.named_modules():
            # if name in width_mask:
            #     name = name
            # else:
            match = re.search(
                r'layers\.(\d+)\.(self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|mlp\.up_proj|mlp\.gate_proj)$',
                name
            )

            if match:
                layer_id = int(match.group(1))  # Keep as string for dictionary key
                layer_type = match.group(2)

                # print(layer_id)
                name = f'layers.{layer_id}.{layer_type}'

                if name in width_mask:
                    mask_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
                    if 'mlp.down_proj' in name or 'self_attn.o_proj' in name:
                        continue
                    else:
                        for key in width_mask[name].keys():
                            if width_mask[name][key] is not None: 
                                if isinstance(width_mask[name][key], np.ndarray):
                                    width_mask[name][key] = torch.from_numpy(width_mask[name][key]).to(mask_dtype)
                                else:
                                    width_mask[name][key] = width_mask[name][key].to(mask_dtype) 
                        # print(name, module)
                        if hasattr(module, 'set_width_mask'):
                            # print(f'{name}: setting width mask')
                            module.set_width_mask(width_mask=width_mask[name], output_bias=None)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    print(f"Rank {local_rank}: Loading cached data...")
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Seq2SeqTrainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # optimizers=(optimizer, scheduler),
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    if not args.layer_calib_dp:
        if args.shrinking_file is not None:
            trainer.strategy = np.load(args.shrinking_file, allow_pickle=True).item()
        else:
            trainer.strategy = None

    # Callbacks
    if not args.full_finetune and not args.get_influence:
        trainer.add_callback(SavePeftModelCallback)
        
    if args.do_mmlu_eval or args.calib_dataset == 'mmlu':
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        if args.do_train:
            class MMLUEvalCallback(transformers.TrainerCallback):
                def on_evaluate(self, args, state, control, model, **kwargs):
                    source_max_len = trainer.data_collator.source_max_len
                    trainer.data_collator.source_max_len = args.mmlu_source_max_len

                    if args.enable_shrinking:
                        active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, 0)

                        if args.full_finetune:
                            model.set_active_layers(active_layers_attn_list[0], active_layers_mlp_list[0])
                        else:
                            model.set_active_layers(active_layers_attn_list[0], active_layers_mlp_list[0], width=1)
                    
                        results_largest = eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy)
                        results_largest = {k + "_largest": v for k, v in results_largest.items()}

                        results = results_largest

                    else:
                        results = eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy)

                    trainer.log(results)
                    trainer.mmlu_results = results
                    trainer.data_collator.source_max_len = source_max_len

            trainer.add_callback(MMLUEvalCallback)
        
        def tokenize_function(examples):
                # 'text' is a common key for the raw text content in MMLU datasets
                # You might need to inspect your JSON files to find the correct key for the raw text (e.g., 'question', 'text', 'prompt')
                # Let's assume the raw text is in a key called 'text'
                # if 'input' in examples:
                return tokenizer(examples['input'], padding=True, truncation=True)

        # Apply the tokenization to the entire dataset
        mmlu_dataset = mmlu_dataset.map(tokenize_function, batched=True)

    else:
        mmlu_dataset = None
        abcd_idx = None
        accuracy = None

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
        
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir if args.resume_training else None)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

                
    # if args.do_eval and not args.no_eval_orig:
    #     if args.enable_shrinking:
    #         # active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, 0)
            
    #         strategy = np.load('dp_selection_strategy.npy', allow_pickle=True).item()["strategy"]
            
    #         active_layers_attn = active_layers_mlp = strategy[model.config.num_hidden_layers - args.eval_num_layer].tolist() #list(strategy.values())

    #         # for i, (active_layers_attn, width_choice) in enumerate(itertools.product(active_layers_attn_list, width_list)):
    #         # active_layers_mlp = active_layers_attn

    #         # active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, 0)

    #         model.set_active_layers(active_layers_attn, active_layers_mlp, width=args.eval_num_width)
        
    #         if args.shrinkable_width:
    #             for module in model.modules():
    #                 if hasattr(module, 'set_width_ratio'):
    #                     # module.set_width_ratio(width_ratio=eval(args.width_choice)[0])
    #                     module.set_width_ratio(width_ratio=args.eval_num_width)
                            
    #             logger.info("*** Evaluate ***")
    #             metrics = trainer.evaluate(metric_key_prefix="eval")
    #             trainer.log_metrics("eval", metrics)
    #             trainer.save_metrics("eval", metrics)
    #             all_metrics.update(metrics)
        

    if args.layer_calib_dp:
        assert args.enable_shrinking
        
        if args.calib_dataset in ['wikitext2', 'redpajama', 'bookcorpus']:
            metric = 'loss'
        elif args.calib_dataset in ['mmlu']:
            if args.calib_metric == 'loss':
                metric = 'loss'
            else:
                metric = 'acc'

        elif args.calib_dataset == 'redpajama':
            calib_dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
        elif args.calib_dataset == 'bookcorpus':
            calib_dataset = load_dataset("bookcorpus")
        else:
            raise ValueError(f"Unknown dataset: {args.calib_dataset}")
        
        
        if args.dp_keep_last_layer > 0:
            N = model.config.num_hidden_layers - args.dp_keep_last_layer  # total number of layers
            M = model.config.num_hidden_layers - args.min_num_layer  # Maximum number of layers to remove
            offset = np.ones(args.dp_keep_last_layer)
        else:
            N = model.config.num_hidden_layers  # total number of layers
            M = N - args.min_num_layer  # Maximum number of layers to remove
            offset = None
                    
        if metric == 'loss':
            d = np.full((N+1, M+1), float('inf'))
        else:
            d = np.full((N+1, M+1), float('-inf'))
            
        strategy = np.zeros((N+1, M+1), dtype=object)

        # Boundary condition: Perplexity of removing 0 layers is set to -1
        for i in range(N+1):
            strategy[i][0] = np.ones(N)  # All layers active up to that point
        
        # Fill the dynamic programming table
        for n in range(1, N+1):
            for m in range(1, M+1):
                # Only try to remove a layer if it's possible to have removed m layers before n
                if m <= n:
                    new_active_layers = strategy[n-1][m-1].copy()
                    new_active_layers[N-n] = 0
                    
                    if offset is not None:
                        active_layers_attn = active_layers_mlp = np.concatenate((new_active_layers, offset), axis=0)
                    else:
                        active_layers_attn = active_layers_mlp = new_active_layers
                        
                    model.set_active_layers(active_layers_attn, active_layers_mlp, width=1)
                    
                    model.eval()
                    with torch.no_grad(): 
                        if args.calib_dataset == 'wikitext2':
                            results = eval_wikitext2_wrapper(trainer, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['wikitext2_ppl']
                        elif args.calib_dataset == 'redpajama':
                            results = eval_general_ppl_wrapper(trainer, calib_dataset, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['ppl']
                        elif args.calib_dataset == 'bookcorpus':
                            results = eval_general_ppl_wrapper(trainer, calib_dataset, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['ppl']
                        elif args.calib_dataset == 'mmlu':
                            results = eval_mmlu_wrapper(trainer, mmlu_dataset, args, abcd_idx, accuracy, n_samples=args.num_calib_sample if args.max_mmlu_samples is None else None)
                            
                            if metric == 'loss':
                                current_metric = results['mmlu_loss']
                            else:
                                current_metric = results['mmlu_eval_accuracy']
                        else:
                            print("Not implemented calibration dataset:", args.calib_dataset)
                            sys.exit()

                    if metric == 'loss': # the lower the better
                        if current_metric < d[n-1][m]:
                            d[n][m] = current_metric
                            strategy[n][m] = new_active_layers
                        else:
                            d[n][m] = d[n-1][m]
                            strategy[n][m] = strategy[n-1][m].copy()
                    else:  # the higher the better
                        if current_metric > d[n-1][m]:
                            d[n][m] = current_metric
                            strategy[n][m] = new_active_layers
                        else:
                            d[n][m] = d[n-1][m]
                            strategy[n][m] = strategy[n-1][m].copy()

        if offset is not None:
            final_strategy = {'strategy': {m: np.concatenate((strategy[N][m], offset), axis=0) for m in range(1, M+1)}, 'metric': {m: d[N][m] for m in range(1, M+1)}}
        else:
            final_strategy = {'strategy': {m: strategy[N][m] for m in range(1, M+1)}, 'metric': {m: d[N][m] for m in range(1, M+1)}}

        print('final_strategy:', final_strategy)
        np.save(os.path.join(args.output_dir, 'final_strategy.npy'), final_strategy)
        
        np.save(os.path.join(args.output_dir, 'full_strategy.npy'), strategy)
        np.save(os.path.join(args.output_dir, 'metric.npy'), d)
    

    if args.enable_shrinking:
        ####################### Manually set up the layer and width choices here #########################
        eval_num_layer = [args.eval_num_layer]  # [32, 30, 28, 26, 24, 22, 20, 18, 16]
        width_list = [args.eval_num_width] # [1, 7/8, 3/4, 5/8, 1/2]
        #########################################################################################
        if eval_num_layer is None:
            if model.config.num_hidden_layers == args.min_num_layer:
                eval_num_layer = [model.config.num_hidden_layers]
            elif args.random_sample_num_layer > 0:
                if args.min_num_layer >= 20:
                    eval_num_layer = [model.config.num_hidden_layers, 24, args.min_num_layer]
                else:
                    eval_num_layer = [model.config.num_hidden_layers, 24, 20, args.min_num_layer]
            else:
                eval_num_layer = [model.config.num_hidden_layers, args.min_num_layer]
            

        if args.shrinking_method == 'calib_dp':
            strategy = np.load(args.shrinking_file, allow_pickle=True).item()["strategy"]
            if not args.eval_after_evolsearch and 0 not in list(strategy.keys()):
                strategy[0] = np.ones(model.config.num_hidden_layers)

            active_layers_list = []
            for num_layer, width in zip(eval_num_layer, width_list):
                if not args.eval_after_evolsearch:
                    # active_layers_list.append(strategy[model.config.num_hidden_layers - num_layer])   # the key of self.strategy is the num of removed layers
                    active_layers_list.append(strategy[(num_layer,num_layer,width)]['active_layers'][0])
                else:
                    # strategy.keys() 
                    # width=1
                    active_layers_list.append(strategy[(num_layer,num_layer,width)]['active_layers'][0])   # the key of self.strategy is the num of removed layers
                    
                    if args.lora_shrinkable_width:
                        strategy = shrink_file['strategy']
                        assert len(eval_num_layer)==len(width_list)==1
                        index_of_w = strategy[eval_num_layer[0]]['effective_width'].index(width_list[0])        
                        lora_mask = strategy[eval_num_layer[0]]['lora_mask'][index_of_w]

            active_layers_attn_list = active_layers_mlp_list = active_layers_list
 
        elif args.shrinking_method == 'first_layers':
            active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, random_sample_num=0, inter_choices=eval_num_layer[1:-1])
        
        else:
            print('Not implemented:', args.shrinking_method)
            sys.exit()


        if not args.shrinkable_width:
            for num_layer, active_layers_attn, active_layers_mlp in zip(eval_num_layer, active_layers_attn_list, active_layers_mlp_list):
                model.set_active_layers(active_layers_attn, active_layers_mlp)
                all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics, suffix=f'_l{num_layer}')
        
        else:
            if width_list is None:
                width_choice = eval(args.width_choice)
                
                if len(width_choice) == 1 or model.config.num_hidden_layers == args.min_num_layer: # if the num layer is not shrinkable, we will measure all candidate widths
                    width_list = width_choice
                    
                else:
                    width_list = [1, 3/4, width_choice[-1]]
            
            for num_layer, active_layers_attn, active_layers_mlp in zip(eval_num_layer, active_layers_attn_list, active_layers_mlp_list):   
                # print(active_layers_attn, active_layers_mlp)
                for width in width_list:
                    for name, module in model.named_modules():
                        if 'mlp.down_proj' in name or 'self_attn.o_proj' in name:
                                continue
                        
                        if hasattr(module, 'set_width_ratio'):
                            # if not args.eval_after_evolsearch:
                            # module.set_width_ratio(width_ratio=width)
                            # else:
                            l = int(np.array(active_layers_attn).sum().item())
                            module.set_width_ratio(width_ratio=(l,l,width))
                        if args.lora_shrinkable_width and hasattr(module, 'set_lora_mask'):
                            module.set_lora_mask(lora_mask=lora_mask)

                    # if len(active_layers_attn)==32:
                    #     model.set_active_layers(active_layers_attn, active_layers_mlp, width=width)
                    # else:
                    model.set_active_layers(active_layers_attn, active_layers_mlp, width=width)

                    if args.few_shot_number!=0:
                        suffix=f'_l{num_layer}w{width}_fs{args.few_shot_number}'
                    else:
                        suffix=f'_l{num_layer}w{width}'

                    all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics, suffix=suffix)
                        
    else:
        all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics)
                    
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_mmlu_eval or args.do_predict or args.do_lm_eval):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


def eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx, accuracy, all_metrics, suffix=''):
    model.eval()
    
    with torch.no_grad():
        if args.do_lm_eval:
            results = eval_lm_eval_wrapper(trainer, tokenizer, model, args, suffix=suffix)
            all_metrics.update(results)
                    
        if args.do_mmlu_eval:   
            results = eval_mmlu_wrapper(trainer, mmlu_dataset, args, abcd_idx, accuracy, suffix=suffix)
            all_metrics.update(results)

        if args.do_eval_wikitext2:
            results = eval_wikitext2_wrapper(trainer, tokenizer, model, args, suffix=suffix)
            all_metrics.update(results)
        
    return all_metrics


    
if __name__ == "__main__":
    train()
