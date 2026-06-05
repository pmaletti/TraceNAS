# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
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
import torch.nn as nn

import sys
sys.path.insert(0, "./transformers/src")
sys.path.insert(0, "./peft/src")
sys.path.insert(0, "./lm-evaluation-harness")
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def compute_metrics(args, results: dict, total_num: int) -> float:
    total_acc = 0
    accs = []
    for name, correct in results.items():
        acc = correct / total_num
        total_acc += correct
        # print("ACC-%s: %.4f" % (name, acc))
    # print("ACC-all: %.4f" % (total_acc/total_num))
    
    return total_acc/total_num


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens



def eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy, n_samples=None):
    data_loader = trainer.get_eval_dataloader(mmlu_dataset)
    trainer.model.eval()
    preds, refs = [], []
    loss_mmlu = 0
    idx = 0
    
    # print('total mmlu questions:', len(data_loader))

    for batch in tqdm(data_loader, total=len(data_loader) if n_samples is None else n_samples):
        (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False,)
        
        for i, logit in enumerate(logits):
            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]  # There are two tokens, the output, and eos token => only pick the first one
            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
            preds.append(torch.argmax(logit_abcd).item())
            
        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
        refs += [abcd_idx.index(label) for label in labels.tolist()]
        loss_mmlu += loss.item()
        
        idx += 1
        if n_samples is not None and idx >= n_samples:
            break
        
    # Extract results by subject.
    results = {'mmlu_loss':loss_mmlu/idx}
    subject = mmlu_dataset['subject']
    subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
    for s,p,r in zip(subject, preds, refs):
        subjects[s]['preds'].append(p)
        subjects[s]['refs'].append(r)
           
    subject_scores = []
    for subject in subjects:
        if n_samples is not None:
            if len(subjects[subject]['refs']) == 0:
                continue
            
        subject_score = accuracy.compute(
            references=subjects[subject]['refs'],
            predictions=subjects[subject]['preds']
        )['accuracy']
        results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
        subject_scores.append(subject_score)
    results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
    return results
      

def eval_mmlu_wrapper(trainer, mmlu_dataset, args, abcd_idx, accuracy, suffix='', n_samples=None):
    set_random_seed(args.seed)
    source_max_len = trainer.data_collator.source_max_len
    trainer.data_collator.source_max_len = args.mmlu_source_max_len
    
    mmlu_results = eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy, n_samples=n_samples)
    trainer.log_metrics(f"mmlu{suffix}", mmlu_results)
    trainer.save_metrics(f"mmlu{suffix}", mmlu_results)
    
    trainer.data_collator.source_max_len = source_max_len
    
    return mmlu_results

# def eval_ruler_wrapper(trainer, ruler_dataset, args, abcd_idx, accuracy, suffix='', n_samples=None):
#     source_max_len = trainer.data_collator.source_max_len
#     trainer.data_collator.source_max_len = args.mmlu_source_max_len
    
#     ruler_results = eval_ruler(trainer, ruler_dataset, args, abcd_idx, accuracy, n_samples=n_samples)
#     trainer.log_metrics(f"ruler{suffix}", ruler_results)
#     trainer.save_metrics(f"ruler{suffix}", ruler_results)
    
#     trainer.data_collator.source_max_len = source_max_len
    
#     return ruler_results

def eval_wikitext2(tokenizer, model, n_samples=40):
    
    n_samples = None
    
    class Evaluator:
        def __init__(self, dataset, tokenizer, device, n_samples=40):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.device = device

            self.dataset = tokenizer(
                "\n\n".join(dataset["text"]), return_tensors="pt"
            ).input_ids.to(device)
            
            self.n_samples = n_samples
            
            self.seq_length = 2048

        @torch.no_grad()
        def evaluate(self, model):
            model.eval()
            nlls = []
            n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // self.seq_length
            
            for i in tqdm(range(n_samples), desc="Evaluating..."):
                batch = self.dataset[:, (i * self.seq_length) : ((i + 1) * self.seq_length)].to(model.device)

                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = self.dataset[:, (i * self.seq_length) : ((i + 1) * self.seq_length)][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * self.seq_length
                nlls.append(neg_log_likelihood)

            return torch.exp(torch.stack(nlls).sum() / (n_samples * self.seq_length))
        
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=n_samples)
    ppl = evaluator.evaluate(model)
    
    results = {"wikitext2_ppl": ppl.item()}
    return results



def eval_wikitext2_wrapper(trainer, tokenizer, model, args, suffix='', n_samples=40):
    set_random_seed(args.seed)
    source_max_len = trainer.data_collator.source_max_len
    trainer.data_collator.source_max_len = tokenizer.model_max_length

    results = eval_wikitext2(tokenizer, model, n_samples)
    trainer.log_metrics(f"wikitext2{suffix}", results)
    trainer.save_metrics(f"wikitext2{suffix}", results)

    trainer.data_collator.source_max_len = source_max_len
    
    return results



def eval_general_ppl(dataset, tokenizer, model, n_samples=40):

    class Evaluator:
        def __init__(self, dataset, tokenizer, device, n_samples=40):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.device = device
            
            self.dataset = dataset['train']["text"][:n_samples]
            
            self.n_samples = n_samples

        @torch.no_grad()
        def evaluate(self, model):
            model.eval()
            nlls = []
            n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
            
            total_length = 0
            for i in tqdm(range(n_samples), desc="Evaluating..."):
                batch = self.tokenizer(self.dataset[i], return_tensors="pt").input_ids.to(model.device)
                
                with torch.no_grad():
                    lm_logits = model(batch).logits

                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                neg_log_likelihood = loss.float() * batch.shape[-1]
                nlls.append(neg_log_likelihood)
                
                total_length += batch.shape[-1]

            return torch.exp(torch.stack(nlls).sum() / total_length)
    
    evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=n_samples)
    ppl = evaluator.evaluate(model)
    
    results = {"ppl": ppl.item()}
    return results



def eval_general_ppl_wrapper(trainer, dataset, tokenizer, model, suffix='', n_samples=40):
    source_max_len = trainer.data_collator.source_max_len
    trainer.data_collator.source_max_len = tokenizer.model_max_length

    results = eval_general_ppl(dataset, tokenizer, model, n_samples)
    
    trainer.log_metrics(f"{suffix}", results)
    trainer.save_metrics(f"{suffix}", results)

    trainer.data_collator.source_max_len = source_max_len
    
    return results



def eval_lm_eval_wrapper(trainer, tokenizer, model, args, suffix=''):
    # print(f"trainer: {trainer}")
    # print(f"model:{model}")
    # print(f"args: {args}")
    # print("==========================")
    
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    set_random_seed(args.seed)
    
    source_max_len = trainer.data_collator.source_max_len
    trainer.data_collator.source_max_len = tokenizer.model_max_length
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

    tasks = args.do_lm_eval_task.split(',')
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=64, num_fewshot=args.few_shot_number) #, log_samples=True, limit=10) #, kwargs={'tokenizer': tokenizer})
   
    results=list(results['results'].values())[0]
    trainer.log_metrics(f"{args.do_lm_eval_task}{suffix}", results)
    trainer.save_metrics(f"{args.do_lm_eval_task}{suffix}", results)
    
    trainer.data_collator.source_max_len = source_max_len
    return results