# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional
import torch.nn as nn

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import LoraLayer
import time

import numpy as np


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, LoraLayer): 
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False
            
            self.use_moe_lora = kwargs["use_moe_lora"]
            self.use_moe_lora_coeff = kwargs["use_moe_lora_coeff"]

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                use_moe_lora = self.use_moe_lora,
                use_moe_lora_coeff = self.use_moe_lora_coeff,
                num_experts = kwargs["num_experts"]
            )
            
            self.shrinkable_width = False
            self.lora_shrinkable_width = False
            self.lora_mask = None
            

        def set_width_mask(self, width_mask, output_bias=None):
            # assert not (width_mask is None and output_bias is None)
            
            self.width_mask = width_mask
            self.output_bias = output_bias
            
            self.shrinkable_width = True
            if hasattr(self, 'lora_mask') and self.lora_mask is not None:
                self.lora_shrinkable_width = True
            self.width_ratio = 1
        
        def set_lora_mask(self, lora_mask):
            assert (self.width_mask is not None or self.output_bias is not None)
            self.lora_shrinkable_width=True
            self.lora_mask = lora_mask

        def set_width_ratio(self, width_ratio):
            assert hasattr(self, 'width_mask')
            self.width_ratio = width_ratio

        # def set_active_layers_key(self, active_layers):
        #     ''' Currently implemented for same active layers for mlp and attn'''

        #     assert (hasattr(self, 'width_mask') and self.width_mask is not None)
        #     self.active_layers_key = active_layers
            
            
        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                output = dequantize_bnb_weight(weight, state=state)
                if not self.use_dora[active_adapter]:
                    w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                else:
                    # handle dora
                    # since output already includes scaling, set it to 1 here
                    weight_norm = self._get_weight_norm(output, lora_data, scaling=1).detach()
                    # We need to cache weight_norm because it has to be based on the original weights. We
                    # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    # different value
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    w_data = dora_factor.view(-1, 1) * (output + lora_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                if not self.use_dora[active_adapter]:
                    w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    w_data = output.data / dora_factor.view(-1, 1) - lora_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.weight.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result

        def init_moe(self, n_embed, num_experts, top_k):
            from .moe import SparseMoE
            
            self.num_experts = num_experts
            self.top_k = top_k
            self.moe_feature = None

            self.lora_sparsemoe = SparseMoE(n_embed, num_experts, top_k)
            
            
        def set_moe_feature(self, moe_feature):
            self.moe_feature = moe_feature
            
            
        def set_moe_gate(self, gating_scores, hard_indices):
            self.gating_scores = gating_scores
            self.hard_indices = hard_indices
            
            
        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        
                        if self.use_moe_lora:
                            x = x.to(lora_A[0].weight.dtype)
                        else:
                            x = x.to(lora_A.weight.dtype)

                    if self.use_moe_lora:
                        if not self.lora_shrinkable_width or self.lora_shrinkable_width is None:
                            gating_scores = self.gating_scores
                            hard_indices = self.hard_indices
                            
                            weight_lora_a = sum([lora_A[indice].weight * gating_scores[indice] for indice in hard_indices])
        
                            output_lora_a = nn.functional.linear(dropout(x), weight_lora_a)

                            weight_lora_b = sum([lora_B[indice].weight * gating_scores[indice] for indice in hard_indices])
                            
                            output = nn.functional.linear(output_lora_a, weight_lora_b) * scaling
                        else:
                            gating_scores = self.gating_scores
                            hard_indices = self.hard_indices
                            
                            weight_lora_a = sum([lora_A[indice].weight * gating_scores[indice] for indice in hard_indices])[self.lora_mask, :]
        
                            output_lora_a = nn.functional.linear(dropout(x), weight_lora_a)

                            weight_lora_b = sum([lora_B[indice].weight * gating_scores[indice] for indice in hard_indices])[:, self.lora_mask]
                            
                            output = nn.functional.linear(output_lora_a, weight_lora_b) * scaling
                    
                    else:
                        if not self.use_dora[active_adapter]:
                            output = lora_B(lora_A(dropout(x))) * scaling
                        else:
                            output = self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
                            
                    if requires_conversion:
                        output = output.to(expected_dtype)

                    result = result + output

            if self.shrinkable_width:
                if self.width_mask is not None:
                    result = result * self.width_mask[self.width_ratio].reshape(1, 1, -1).to(result)
                
                if self.output_bias is not None:
                    result = result + self.output_bias[self.width_ratio].reshape(1, 1, -1).to(result)
                
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep


    def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, kaiming_init=None, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    # "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)

        return new_module


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False
            
            self.use_moe_lora = kwargs["use_moe_lora"]
            self.use_moe_lora_coeff = kwargs["use_moe_lora_coeff"]

            ### Deprecated: We will use the same MoE gate define in the PeFT model for all the layers
            # if self.use_moe_lora:
            #     self.init_moe(kwargs["n_embed"], kwargs["num_experts"], kwargs["top_k"])
                
            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                use_moe_lora = self.use_moe_lora,
                use_moe_lora_coeff = self.use_moe_lora_coeff,
                num_experts = kwargs["num_experts"]
            )
            
            self.shrinkable_width = False
            self.lora_shrinkable_width = False
            self.lora_mask = None

        def set_width_mask(self, width_mask, output_bias=None, lora_mask=None, search=False):
            # if not search:
            #     assert not (width_mask is None and output_bias is None)
            
            self.width_mask = width_mask
            self.output_bias = output_bias
            self.lora_mask = lora_mask
            
            self.shrinkable_width = True
            if self.lora_mask is not None:
                self.lora_shrinkable_width = True
            self.width_ratio = 1
            # self.active_layers_key = None
            
        def set_lora_mask(self, lora_mask):
            assert (self.width_mask is not None or self.output_bias is not None)
            self.lora_shrinkable_width=True
            self.lora_mask = lora_mask
            
        def set_width_ratio(self, width_ratio):
            # assert hasattr(self, 'width_mask') or hasattr(self, 'output_bias')
            self.width_ratio = width_ratio
        
        # def set_active_layers_key(self, active_layers):
        #     ''' Currently implemented for same active layers for mlp and attn'''

        #     assert (hasattr(self, 'width_mask') and (self.width_mask is not None or self.output_bias is not None))
        #     self.active_layers_key = active_layers
            
        
        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)

                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                if not self.use_dora[active_adapter]:
                    w_data = output + lora_data
                else:
                    # handle dora
                    # since output already includes scaling, set it to 1 here
                    weight_norm = self._get_weight_norm(output, lora_data, scaling=1).detach()
                    # We need to cache weight_norm because it has to be based on the original weights. We
                    # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    # different value
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    w_data = dora_factor.view(-1, 1) * (output + lora_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )

                lora_data = self.get_delta_weight(active_adapter)
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                output = dequantize_bnb_weight(weight, state=weight.quant_state)

                if not self.use_dora[active_adapter]:
                    w_data = output - lora_data
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    w_data = output.data / dora_factor.view(-1, 1) - lora_data

                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result
    
        def init_moe(self, n_embed, num_experts, top_k):
            from .moe import SparseMoE
            
            self.num_experts = num_experts
            self.top_k = top_k
            self.moe_feature = None

            self.lora_sparsemoe = SparseMoE(n_embed, num_experts, top_k)
            
            
        def set_moe_feature(self, moe_feature):
            self.moe_feature = moe_feature
            
            
        def set_moe_gate(self, gating_scores, hard_indices):
            self.gating_scores = gating_scores
            self.hard_indices = hard_indices
        
        def get_closest_valid_width(self, effective_width, width_choice_list):
            """
            Finds the closest value in width_choice_list to the effective_width.
            """
            if not width_choice_list:
                return None
            
            closest_width = min(width_choice_list, key=lambda x: abs(x - effective_width))
            return closest_width
            
        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
                
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                
                # torch.cuda.synchronize()
                # t0 = time.time()
                
                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

                for i, active_adapter in enumerate(self.active_adapters):
                    if active_adapter not in self.lora_A.keys():
                        continue
                    # if not self.use_moe_lora_coeff:
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    if self.use_moe_lora_coeff:
                        # lora_A = self.lora_A_fixed_basis
                        # lora_B = self.lora_B_fixed_basis
                        lora_coeff_v = self.lora_coeff_v
                        
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        if self.use_moe_lora:
                            x = x.to(lora_A[0].weight.dtype)
                        else:
                            x = x.to(lora_A.weight.dtype)

                    if self.use_moe_lora:
                        if not self.lora_shrinkable_width or self.lora_shrinkable_width is None:
                            gating_scores = self.gating_scores
                            hard_indices = self.hard_indices
                            
                            weight_lora_a = sum([lora_A[indice].weight * gating_scores[indice] for indice in hard_indices])
                            
                            output_lora_a = nn.functional.linear(dropout(x), weight_lora_a)

                            weight_lora_b = sum([lora_B[indice].weight * gating_scores[indice] for indice in hard_indices])
                            
                            output = nn.functional.linear(output_lora_a, weight_lora_b) * scaling

                        else:
                            gating_scores = self.gating_scores
                            hard_indices = self.hard_indices
                            
                            weight_lora_a = sum([lora_A[indice].weight * gating_scores[indice] for indice in hard_indices])[self.lora_mask, :]
                            output_lora_a = nn.functional.linear(dropout(x), weight_lora_a)
                            # output_lora_a = nn.functional.linear(dropout(x), weight_lora_a * self.lora_mask.reshape(-1,1).to(weight_lora_a))

                            weight_lora_b = sum([lora_B[indice].weight * gating_scores[indice] for indice in hard_indices])[:, self.lora_mask]
                            output = nn.functional.linear(output_lora_a, weight_lora_b) * scaling
                            # output = nn.functional.linear(output_lora_a, weight_lora_b * self.lora_mask.reshape(-1,1).to(weight_lora_b)) * scaling
                    elif self.use_moe_lora_coeff:
                        gating_scores = self.gating_scores
                        hard_indices = self.hard_indices

                        weight_lora_a = lora_A.weight
                        output_lora_a = nn.functional.linear(dropout(x), weight_lora_a)
                        
                        # weight_lora_b = sum([lora_B.weight * torch.diag(lora_coeff_v[indice]) * gating_scores[indice] for indice in hard_indices])
                        # print(lora_coeff_v[0].weight.shape, lora_B.weight.shape)
                        weight_lora_b = sum([lora_B.weight @ lora_coeff_v[indice].weight * gating_scores[indice] for indice in hard_indices])
                        output = nn.functional.linear(output_lora_a, weight_lora_b) * scaling


                        # gating_scores = self.gating_scores
                        # hard_indices = self.hard_indices

                        # # Prepare inputs for the custom function
                        # weight_lora_a = lora_A.weight #.t() # A is [r, in_features]
                        # weight_lora_b = lora_B.weight #.t() # B is [out_features, r]
                        # all_coeff_weights = [c.weight for c in lora_coeff_v] # C is [r, r]
                        
                        # # Note: Dropout is applied *outside* the custom function on x
                        # # The custom function calculates the full delta weight matrix: W_delta

                        # # --- NEW INTEGRATION POINT ---
                        # output = ScaledLoRAUpdate.apply(
                        #     dropout(x),
                        #     weight_lora_a, 
                        #     weight_lora_b, 
                        #     all_coeff_weights, 
                        #     gating_scores, 
                        #     hard_indices, 
                        #     scaling
                        # )


                    else:
                        if not self.use_dora[active_adapter]:
                            output = lora_B(lora_A(dropout(x))) * scaling
                        else:
                            output = self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
                            
                    if requires_conversion:
                        output = output.to(expected_dtype)
                        
                    result = result + output
                    
            if self.shrinkable_width:
                # if not (hasattr(self, 'active_layers_key') and self.active_layers_key is not None):
                if self.width_mask is not None:
                    if isinstance(self.width_mask, dict):
                        if self.width_ratio in self.width_mask: # and self.width_mask[self.width_ratio] is not None:
                            result = result * self.width_mask[self.width_ratio].reshape(1, 1, -1).to(result)
                
                    ### Adding evolution search compatibility ###
                    elif isinstance(self.width_mask, np.ndarray):
                        result = result * self.width_mask.reshape(1, 1, -1).to(result)
                
                if self.output_bias is not None  and isinstance(self.output_bias, dict):
                    # self.width_ratio = self.get_closest_valid_width(self.width_ratio, [i for i in self.output_bias.keys()])
                    # if self.width_ratio in self.output_bias:
                    result = result + self.output_bias[self.width_ratio].reshape(1, 1, -1).to(result)
                elif self.output_bias is not None  and isinstance(self.output_bias, np.ndarray):
                    result = result + self.output_bias.reshape(1, 1, -1).to(result)
                # else:
                #     if self.width_mask is not None and isinstance(self.width_mask, dict):
                #         if isinstance(self.width_mask, dict):
                #             result = result * self.width_mask[self.active_layers_key].reshape(1, 1, -1).to(result)
                        # elif isinstance(self.width_mask, list):
                
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep

    def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

        return new_module

from torch.autograd import Function

class ScaledLoRAUpdate(Function):
    """
    Custom autograd function to calculate the LoRA update matrix and 
    apply the gradient scaling (division strategy) to shared A and B weights.
    """
    @staticmethod
    def forward(ctx, x, weight_lora_a, weight_lora_b, coeff_weights_list, 
                gating_scores, hard_indices, scaling):

        # 1. Calculate the weighted B matrix transformation (W_BC)
        W_BC = 0.0
        # total_gating_sum = 0.0

        active_gating_scores = [gating_scores[i] for i in hard_indices]
        active_coeff_weights = [coeff_weights_list[i] for i in hard_indices]
            
        for C_i, score_i in zip(active_coeff_weights, active_gating_scores):
            # W_BC += B @ C_i^T * score_i
            # lora_B.weight @ lora_coeff_v[indice].weight
            W_BC += weight_lora_b @ C_i * score_i
            # total_gating_sum += score_i
        # EPSILON = 1e-6 
        
        # # Add epsilon uniformly across all elements of W_BC.
        # # W_BC will now be slightly non-zero, ensuring a non-zero gradient flow.
        # W_BC = W_BC + W_BC.new_full(W_BC.shape, EPSILON)

        # 2. Perform sequential low-rank computation (EFFICIENT FORWARD)
        # H_A = x @ A^T
        H_A = torch.matmul(x, weight_lora_a.t())

        # Output_LoRA = H_A @ W_BC^T
        y_delta = torch.matmul(H_A, W_BC.t()) * scaling

        # # 2. Calculate the final LoRA update matrix (W_lora = W_BC @ A)
        # # W_lora = [out_features, r] @ [r, in_features] = [out_features, in_features]
        # lora_update = torch.matmul(W_BC, weight_lora_a) * scaling

        # 3. Save relevant tensors and the total gating sum for the backward pass
        # ctx.save_for_backward(weight_lora_a, weight_lora_b, W_BC)
        # ctx.save_for_backward(x, weight_lora_a, weight_lora_b, H_A, W_BC)
        ctx.save_for_backward(x, weight_lora_a, weight_lora_b, H_A, W_BC, gating_scores)
        ctx.coeff_weights_list = coeff_weights_list
        # ctx.total_gating_sum = total_gating_sum
        ctx.hard_indices = hard_indices
        ctx.scaling = scaling

        # return lora_update
        return y_delta

    @staticmethod
    def backward(ctx, grad_y):
        import pdb; pdb.set_trace()

        # Retrieve saved tensors and parameters
        x, A, B, H_A, W_BC, gating_scores = ctx.saved_tensors
        x_flat = x.view(-1, x.size(-1))
        C_list = ctx.coeff_weights_list
        Indices = ctx.hard_indices
        G = ctx.scaling

        # Initialize gradients for the outputs
        grad_x = None
        grad_A = None
        grad_B = None
        grad_C_list = [torch.zeros_like(C) for C in C_list]
        grad_S = torch.zeros_like(gating_scores)
        grad_Indices = None
        grad_G = None

        # Dimensions:
        # grad_y: [Batch, Out]
        # H_A: [Batch, Rank]
        # W_BC: [Out, Rank]
        # A: [Rank, In]
        # B: [Out, Rank]
        # C_i: [Rank, Rank] (Assumed square rank matrix from the forward matmul B @ C_i)

        # -----------------------------------------------------
        # 1. Gradient w.r.t. W_BC (Aggregated LoRA B)
        # grad_W_BC = G * (grad_y @ H_A^T)^T = G * grad_y^T @ H_A
        # Shape: [Out, Rank]
        grad_y = grad_y.view(-1, grad_y.size(-1))
        H_A = H_A.view(-1, H_A.size(-1))  # (B*L, r)
        grad_W_BC = G * torch.matmul(grad_y.t(), H_A)

        # -----------------------------------------------------
        # 2. Gradient w.r.t. H_A (Hidden state after A)
        # grad_H_A = G * grad_y @ W_BC
        # Shape: [Batch, Rank]
        grad_H_A = G * torch.matmul(grad_y, W_BC)

        # -----------------------------------------------------
        # 3. Gradients w.r.t. x and A (using grad_H_A)
        
        # H_A = x @ A^T
        if ctx.needs_input_grad[0]: # x
            # grad_x = grad_H_A @ A
            # Shape: [Batch, In]
            grad_x = torch.matmul(grad_H_A, A)
            grad_x = grad_x.reshape(x.shape)
            
        if ctx.needs_input_grad[1]: # A
            # grad_A = grad_H_A^T @ x
            # Shape: [Rank, In]
            grad_A = torch.matmul(grad_H_A.t(), x_flat)

        # -----------------------------------------------------
        # 4. Gradients w.r.t. B, C_i, and Scores (S_i) (using grad_W_BC)
        
        # W_BC = sum_i (B @ C_i) * S_i
        
        # Calculate C_agg for the B gradient: C_agg = sum_i C_i * S_i
        # This re-creates the aggregated coefficient matrix used in the forward pass.
        C_agg = 0.0
        for i in Indices:
            C_i = C_list[i]
            S_i = gating_scores[i]
            C_agg += C_i * S_i

        if ctx.needs_input_grad[2]: # B
            # grad_B = grad_W_BC @ C_agg^T
            # Shape: [Out, Rank]
            if isinstance(C_agg, float): # Case where no index was active (C_agg is 0.0)
                grad_B = torch.zeros_like(B)
            else:
                grad_B = torch.matmul(grad_W_BC, C_agg.t())

        # Gradient w.r.t. the Aggregated Coefficient Matrix (grad_C_agg)
        # grad_C_agg = B^T @ grad_W_BC
        # Shape: [Rank, Rank]
        grad_C_agg = torch.matmul(B.t(), grad_W_BC)

        # if ctx.needs_input_grad[3]: # C_list
            # grad_C_i = grad_C_agg * S_i (for active indices i)
        for i in Indices:
            S_i = gating_scores[i]
            grad_C_list[i] = grad_C_agg * S_i

        # if ctx.needs_input_grad[4]: # gating_scores (S_i)
            # grad_S_i = sum(grad_W_BC * (B @ C_i)) (Dot product of grad_W_BC and B @ C_i)
        for i in Indices:
            C_i = C_list[i]
            # B @ C_i is the effective weight contribution for score S_i
            W_i = torch.matmul(B, C_i) 
            # The gradient is the element-wise product summed over all elements
            grad_S[i] = torch.sum(grad_W_BC * W_i)


        # -----------------------------------------------------
        # 5. Gradient w.r.t. Scaling (G)
        
        # if ctx.needs_input_grad[6]: # scaling
            # grad_G = sum(grad_y * (H_A @ W_BC^T)) (Dot product of grad_y and the unscaled output)
        unscaled_output = torch.matmul(H_A, W_BC.t())
        grad_G = torch.sum(grad_y * unscaled_output)

        # Gradient for hard_indices is None as it is a discrete selection index.
        grad_Indices = ctx.hard_indices #None

        return grad_x, grad_A, grad_B, grad_C_list, grad_S, grad_Indices, grad_G
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # Retrieve saved tensors and context variables
    #     x, weight_lora_a, weight_lora_b, H_A, W_BC = ctx.saved_tensors

    #     import pdb; pdb.set_trace()
        
    #     coeff_weights_list = ctx.coeff_weights_list
    #     scaling = ctx.scaling
        
    #     # ASSUMED TO BE SAVED IN FORWARD:
    #     hard_indices = ctx.hard_indices
    #     gating_scores = ctx.gating_scores

    #     # --- Preliminary Flattening ---
    #     # Gradients involving H_A and x require flattening the (Batch, SeqLen) dimensions.
    #     x_flat = x.view(-1, x.size(-1))        # (B*L, d_in)
    #     H_A_flat = H_A.view(-1, H_A.size(-1))  # (B*L, r)
    #     grad_output_flat = grad_output.view(-1, grad_output.size(-1)) # (B*L, d_out)
        
    #     # --- Step 1: dL/dW_BC and dL/dH_A ---
        
    #     # 1a. dL/dW_BC^T: gradient w.r.t the effective weight matrix used in matmul (r x d_out)
    #     # dL/dW_BC_T = H_A_flat.t() @ grad_output_flat * scaling
    #     grad_W_BC_T = torch.matmul(H_A_flat.t(), grad_output_flat) * scaling
        
    #     # dL/dW_BC (d_out x r) is the transpose of dL/dW_BC_T
    #     grad_W_BC = grad_W_BC_T.t() 

    #     # 1b. dL/dH_A: gradient w.r.t the intermediate feature H_A (B, L, r)
    #     # dL/dH_A = grad_output @ W_BC * scaling
    #     grad_H_A = torch.matmul(grad_output, W_BC) * scaling
    #     grad_H_A_flat = grad_H_A.view(-1, grad_H_A.size(-1)) # (B*L, r)
        
    #     # --- Step 2: Gradients for LoRA A and input X ---

    #     # 2a. dL/dx: grad_H_A @ A (A is r x d_in)
    #     grad_x = torch.matmul(grad_H_A, weight_lora_a)

    #     # 2b. dL/dA: (r, d_in)
    #     # dL/dA^T = x_flat.t() @ grad_H_A_flat
    #     grad_A_T = torch.matmul(x_flat.t(), grad_H_A_flat)
    #     grad_A = grad_A_T.t() 

    #     # --- Step 3: Gradients for LoRA B, Coeffs C_i, and Gating Scores ---
        
    #     grad_B = torch.zeros_like(weight_lora_b) # (d_out, r)
    #     B_t = weight_lora_b.t() # (r, d_out)
        
    #     # Initialize full gradient lists for all experts (including inactive ones)
    #     full_grad_coeff_list = [torch.zeros_like(C_i) for C_i in coeff_weights_list]
    #     full_grad_gating_scores = torch.zeros_like(gating_scores)
        
    #     # Only iterate over the active experts, as inactive experts have zero gradient.
    #     for local_idx, expert_idx in enumerate(hard_indices):
    #         C_i = coeff_weights_list[expert_idx]
    #         score_i = gating_scores[expert_idx]
            
    #         # 3a. dL/dB (contributed by this expert): dL/dW_BC @ C_i.t() * score_i
    #         # Summing contributions from all active experts for grad_B
    #         grad_B += torch.matmul(grad_W_BC, C_i.t()) * score_i

    #         # 3b. dL/dC_i (r x r)
    #         # grad_C_i = B.t() @ dL/dW_BC * score_i
    #         grad_C_i = torch.matmul(B_t, grad_W_BC) * score_i
    #         full_grad_coeff_list[expert_idx] = grad_C_i

    #         # 3c. dL/dscore_i (scalar)
    #         # grad_score_i = sum( (B @ C_i) * dL/dW_BC )
    #         term = torch.matmul(weight_lora_b, C_i) # (d_out, r)
    #         grad_score_i = torch.sum(term * grad_W_BC) # Equivalent to Trace( (B@C_i)^T @ grad_W_BC )
    #         full_grad_gating_scores[expert_idx] = grad_score_i

    #     # The list of C_i gradients must be returned as a tuple of tensors.
    #     grad_coeff_weights_list_tuple = tuple(full_grad_coeff_list)

    #     # --- Step 4: Final Return ---
    #     # Must match forward inputs: x, A, B, C_list, scores, hard_indices, scaling
    #     return (grad_x,               # dL/dx (1)
    #             grad_A,               # dL/dA (2)
    #             grad_B,               # dL/dB (3)
    #             grad_coeff_weights_list_tuple, # dL/dC_list (4)
    #             full_grad_gating_scores, # dL/dscores (5)
    #             None,                 # dL/d hard_indices (non-tensor, requires no grad) (6)
    #             None)                 # dL/d scaling (non-tensor, requires no grad) (7)


    # @staticmethod
    # def backward(ctx, grad_output):
        
    #     # Retrieve saved tensors and values
    #     x, weight_lora_a, weight_lora_b, H_A, W_BC = ctx.saved_tensors
    #     total_gating_sum = ctx.total_gating_sum
    #     scaling = ctx.scaling
        
    #     # --- Gradient Scaling Factor (MoE Division) ---
    #     scale_factor = 1.0 / total_gating_sum if total_gating_sum > 1e-6 else 1.0
        
    #     # Grad of output, scaled by the LoRA scaling factor
    #     grad_scaled = grad_output * scaling
        
    #     # --- 1. Gradient for A (weight_lora_a) ---
    #     # A gradient flows from grad_scaled through W_BC.t() to H_A, then through A.t() to x.
        
    #     # Step 1a: Calculate the gradient w.r.t H_A (dL/dH_A).
    #     # H_A @ W_BC^T = y_delta. Grad of H_A is grad_output @ W_BC.
    #     grad_H_A = torch.matmul(grad_scaled, W_BC) 
        
    #     # Step 1b: Calculate the gradient w.r.t A (dL/dA).
    #     # H_A = x @ A^T. Grad of A is x^T @ grad_H_A.
    #     # Since x is (B*S, D_in) and H_A is (B*S, r), grad_A must be (r, D_in).
    #     grad_lora_a = torch.matmul(H_A.t(), grad_H_A) # ERROR: This is actually grad_A^T. Let's use x instead.
    #     grad_lora_a = torch.matmul(x.t(), grad_H_A) # (D_in, B*S) @ (B*S, r) = (D_in, r). Needs transpose.
    #     grad_lora_a = torch.matmul(x.t(), grad_H_A).t() # (r, D_in)
        
    #     # Apply the custom scaling division
    #     grad_lora_a.mul_(scale_factor) 

    #     # --- 2. Gradient for B (weight_lora_b) ---
    #     # B gradient is implicitly part of W_BC.
        
    #     # Step 2a: Calculate the gradient w.r.t W_BC (dL/dW_BC).
    #     # W_BC^T is multiplied by H_A. Grad of W_BC^T is H_A^T @ grad_output.
    #     # Grad of W_BC is (H_A^T @ grad_output)^T = grad_output^T @ H_A.
    #     grad_W_BC = torch.matmul(grad_scaled.t(), H_A) # (D_out, D_in) @ (D_in, r) = (D_out, r)
        
    #     # Step 2b: Calculate the gradient w.r.t B (dL/dB).
    #     # B is an input to W_BC. Since W_BC = sum(B @ C_i * score_i), the derivative 
    #     # w.r.t B requires the gradient of W_BC multiplied by the inverse of C_i, 
    #     # which is extremely complex and error-prone.
        
    #     # Simplest Heuristic (Aligned with MoE Scaling): 
    #     # We assume grad_W_BC is proportional to grad_B, and apply the scaling.
    #     # This is not strictly correct but enforces the intended sparsity/scaling mechanism.
        
    #     grad_lora_b = grad_W_BC.clone() 
    #     grad_lora_b.mul_(scale_factor) # Apply the scaling division

    #     # --- 3. Gradient for x (input activation) ---
    #     # x is an input to H_A. H_A = x @ A^T. Grad of x is grad_H_A @ A.
    #     grad_x = torch.matmul(grad_H_A, weight_lora_a) 
        
    #     # --- 4. Gradients for Other Inputs (C_list, scores, indices, scaling) ---
    #     # Coeff gradients are handled by standard autograd (return None).
    #     num_coeff_inputs = len(ctx.coeff_weights_list) if hasattr(ctx, 'coeff_weights_list') else 0
    #     grad_coeff_weights = [None] * num_coeff_inputs 
        
    #     # Return gradients in the order of the forward inputs:
    #     # x, A, B, C_list, scores, indices, scaling
    #     return grad_x, grad_lora_a, grad_lora_b, *grad_coeff_weights, None, None, None
    
    # # @staticmethod
    # # def backward(ctx, grad_output):
        
    # #     # Retrieve saved tensors and values
    # #     # weight_lora_a, weight_lora_b, W_BC = ctx.saved_tensors
    # #     x, weight_lora_a, weight_lora_b, H_A, W_BC = ctx.saved_tensors
    # #     coeff_weights_list = ctx.coeff_weights_list
    # #     total_gating_sum = ctx.total_gating_sum
    # #     scaling = ctx.scaling
        
    # #     # --- Gradient Scaling Factor ---
    # #     # Ensures that A and B update is the average of the active paths.
    # #     scale_factor = 1.0 / total_gating_sum if total_gating_sum > 1e-6 else 1.0
        
    # #     # Gradient of the final output (scaled by LoRA scaling factor)
    # #     grad_scaled = grad_output * scaling

    # #     # --- 1. Gradient for A (weight_lora_a) ---
    # #     # grad_A = H_A.t() @ grad_scaled_output
    # #     grad_lora_a = torch.matmul(H_A.t(), grad_scaled)
    # #     grad_lora_a.mul_(scale_factor) # Apply the scaling division

    # #     # --- 2. Gradient for B (weight_lora_b) ---
    # #     # The true calculation is complicated. We use the approximation based on 
    # #     # the W_BC gradient flow, which is equivalent to grad_W_BC.
    # #     # grad_W_BC = grad_scaled @ (H_A @ B_eff)^T / grad(B)
        
    # #     # Grad of H_A (w.r.t the input to B matrix)
    # #     grad_H_A = torch.matmul(grad_scaled, W_BC) # grad_H_A = grad_output @ W_BC
        
    # #     # Grad of W_BC w.r.t B (The heuristic/approximation for scaled update)
    # #     # grad_B = grad_H_A.t() @ x (conceptually)
    # #     grad_lora_b = torch.matmul(grad_H_A.t(), x.reshape(-1, x.shape[-1]))
    # #     grad_lora_b.mul_(scale_factor) # Apply the scaling division

    # #     # --- 3. Gradients for Other Inputs ---
    # #     # The number of 'None's must match the number of inputs to forward.
    # #     # Inputs: x, A, B, C_list (K tensors), scores, indices, scaling (7 inputs total + K Coeffs)
        
    # #     # Gradient for x (input activation)
    # #     grad_x = torch.matmul(grad_H_A, weight_lora_a) 
        
    # #     # Gradients for coeff_weights_list (K tensors), scores, indices, scaling
    # #     num_coeff_inputs = len(coeff_weights_list)
    # #     grad_coeff_weights = [None] * num_coeff_inputs 
        
    # #     # Return gradients in the order of the forward inputs:
    # #     # x, A, B, C_list (K tensors), scores, indices, scaling
    # #     return grad_x, grad_lora_a, grad_lora_b, *grad_coeff_weights, None, None, None

    #     # # --- 1. Gradient for A (weight_lora_a) ---
    #     # # grad_A = W_BC.t() @ grad_scaled
    #     # grad_lora_a = torch.matmul(W_BC.t(), grad_scaled)
        
    #     # # Apply the scaling division (the MoE division strategy)
    #     # grad_lora_a.mul_(scale_factor)
        
    #     # # --- 2. Gradient for B (weight_lora_b) ---
    #     # # Grad of W_BC = grad_scaled @ A.t()
    #     # grad_W_BC = torch.matmul(grad_scaled, weight_lora_a.t())
        
    #     # # Apply the scaling division for B (This is the heuristic/approximation)
    #     # grad_lora_b = grad_W_BC.mul(scale_factor)
        
    #     # # --- 3. Gradient for Coefficients (coeff_weights) ---
    #     # # These rely on standard backprop, so we return None.
    #     # grad_coeff_weights = tuple([None] * len(ctx.next_variables) - 6) # -6 for the 6 non-coeff inputs
        
    #     # # Return gradients for the inputs: A, B, Coeffs, Scores, Indices, Scaling
    #     # # We must return a gradient for *all* inputs to forward.
    #     # num_coeff_inputs = len(ctx.next_variables) - 6
    #     # grad_coeff_weights = [None] * num_coeff_inputs
        
    #     # return grad_lora_a, grad_lora_b, *grad_coeff_weights, None, None, None