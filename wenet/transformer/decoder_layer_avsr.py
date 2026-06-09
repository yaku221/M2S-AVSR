# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Copyright (c) 2025 Fei Su (shinji721@outlook.com)
#
# Modified from the original WeNet implementation by Fei Su, 2025.
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

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from wenet.transformer.attention import T_CACHE

from wenet.utils.class_utils import WENET_NORM_CLASSES


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        add_gated_x_attn: bool = False,
        gated_x_attn: Optional[nn.Module] = None,
        ff: Optional[nn.Module] = None,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.norm1 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps) # for self-attn
        self.norm2 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps) # for audio cross-attn
        self.norm3 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps) # for FFN
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        
        # visual gated cross-attn
        norm_class = WENET_NORM_CLASSES[layer_norm_type]
        self.add_gated_x_attn = add_gated_x_attn
        self.gated_x_attn = gated_x_attn
        self.ff = ff
        if self.add_gated_x_attn:
            assert self.gated_x_attn is not None and self.ff is not None
            self.gated_x_attn_ln = norm_class(size, eps=norm_eps)   # layernorm for x-attn input
            self.ff_ln = norm_class(size, eps=norm_eps)  # layernorm for gated-ffn
            self.attn_gate = nn.Parameter(torch.tensor([0.0]))
            self.ff_gate = nn.Parameter(torch.tensor([0.0]))
            
            
    def _apply_gated_x_attn(
        self,
        x: torch.Tensor,
        memory_v: torch.Tensor,
        memory_v_mask: torch.Tensor,
        cross_att_cache_v: Optional[T_CACHE],
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Gated visual cross-attention + gated FF"""
        # import ipdb; ipdb.set_trace()
        residual = x
        gated_x_norm = self.gated_x_attn_ln(x)
        if cross_att_cache_v is None:
            cross_att_cache_v = (torch.empty(0,0,0,0, device=x.device, dtype=x.dtype),
                                 torch.empty(0,0,0,0, device=x.device, dtype=x.dtype))
        x_attn, new_cross_cache_v = self.gated_x_attn(
            gated_x_norm, memory_v, memory_v, memory_v_mask, cache=cross_att_cache_v
        )
        # gated with tanh
        x = residual + x_attn * self.attn_gate.tanh()
        # import ipdb; ipdb.set_trace()
        residual = x
        x_ff = self.ff(self.ff_ln(x))
        x = residual + x_ff * self.ff_gate.tanh()

        return x, new_cross_cache_v


    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_a: torch.Tensor,
        memory_a_mask: torch.Tensor,
        memory_v: Optional[torch.Tensor] = None,
        memory_v_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, Optional[T_CACHE]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        # import ipdb; ipdb.set_trace() 
        if cache is not None:
            att_cache = cache.get('self_att_cache', None)
            cross_att_cache_a = cache.get('cross_att_cache_a', None)
            cross_att_cache_v = cache.get('cross_att_cache_v', None)
        else:
            att_cache, cross_att_cache_a, cross_att_cache_v = None, None, None
                      
        # -------- Gated x-attn with visual --------
        use_v_cache = (not self.training) and (cross_att_cache_v is not None)
        if self.add_gated_x_attn and (memory_v is not None) and (memory_v_mask is not None):
            x, new_cross_cache_v = self._apply_gated_x_attn(
                tgt, memory_v, memory_v_mask,
                cross_att_cache_v if use_v_cache else None   
            )
            if (not self.training) and (cache is not None):
                cache['cross_att_cache_v'] = new_cross_cache_v  
        else:
            x = tgt 
             
        # -------- self-attention --------
        # import ipdb; ipdb.set_trace() 
        residual = x
        tgt = self.norm1(x) if self.normalize_before else x

        if att_cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
            att_cache = (
                torch.empty(0, 0, 0, 0, device=x.device, dtype=x.dtype),
                torch.empty(0, 0, 0, 0, device=x.device, dtype=x.dtype),
            )
        else:
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        x, new_att_cache = self.self_attn(
            tgt_q,
            tgt_q,
            tgt_q,
            tgt_q_mask,
            cache=att_cache,
        )
        if (not self.training) and (cache is not None):
            cache['self_att_cache'] = new_att_cache
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm1(x)

        # import ipdb; ipdb.set_trace()     
        # -------- cross-attn with audio --------
        if self.src_attn is not None:
            residual = x
            x = self.norm2(x) if self.normalize_before else x
            if cross_att_cache_a is None:
                cross_att_cache_a = (
                    torch.empty(0, 0, 0, 0, device=x.device, dtype=x.dtype),
                    torch.empty(0, 0, 0, 0, device=x.device, dtype=x.dtype),
                )
            x, new_cross_cache_a = self.src_attn(
                x, memory_a, memory_a, memory_a_mask, cache=cross_att_cache_a
            )
            if (not self.training) and (cache is not None):
                cache['cross_att_cache_a'] = new_cross_cache_a

            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm2(x)
                
        # import ipdb; ipdb.set_trace()
        # -------- FFN --------
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        
        # import ipdb; ipdb.set_trace()
        return x, tgt_mask, memory_a, memory_a_mask
