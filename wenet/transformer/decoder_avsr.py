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

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint as ckpt
from wenet.transformer.attention import T_CACHE
from wenet.transformer.decoder_layer_avsr import DecoderLayer
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_EMB_CLASSES, WENET_MLP_CLASSES,
                                     WENET_NORM_CLASSES)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import make_pad_mask, subsequent_mask


class M2S_AVSR_Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,   # audio cross-attn
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,        
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        tie_word_embedding: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        src_query_bias: bool = True,  # audio cross-attn
        src_key_bias: bool = True,
        src_value_bias: bool = True,
        add_gated_x_attn: bool = False,  # gated visual x-attn
        src_query_bias_v: bool = True,
        src_key_bias_v: bool = True,
        src_value_bias_v: bool = True,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        self.embed = torch.nn.Sequential(
            torch.nn.Identity() if input_layer == "no_pos" else
            torch.nn.Embedding(vocab_size, attention_dim),
            WENET_EMB_CLASSES[input_layer](attention_dim,
                                           positional_dropout_rate),
        )

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](attention_dim,
                                                              eps=norm_eps)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self_attn_cls = WENET_ATTENTION_CLASSES["selfattn"]
        cross_attn_cls = WENET_ATTENTION_CLASSES["crossattn"]
        self.decoders = torch.nn.ModuleList()
        
        # init decoder
        for _ in range(self.num_blocks):
            # self-attention
            self_attn = self_attn_cls(
                attention_heads, attention_dim,
                self_attention_dropout_rate,
                query_bias, key_bias, value_bias,
                use_sdpa, n_kv_head, head_dim
            )
            # audio cross-attention (optional)
            src_attn_audio = cross_attn_cls(
                attention_heads, attention_dim, src_attention_dropout_rate,
                src_query_bias, src_key_bias, src_value_bias,
                use_sdpa, n_kv_head, head_dim
            ) if src_attention else None

            # FFN
            mlp = mlp_class(
                attention_dim, linear_units,
                dropout_rate, activation, mlp_bias,
                n_expert=n_expert, n_expert_activated=n_expert_activated
            )

            # ---- visual gated x-attn branch ----
            if add_gated_x_attn:
                # print("Fusion_method: gated_fusion!!")
                gated_x_attn = cross_attn_cls(
                    attention_heads, attention_dim, src_attention_dropout_rate,
                    src_query_bias_v, src_key_bias_v, src_value_bias_v,
                    use_sdpa, n_kv_head, head_dim
                )
                ff = mlp_class(
                    attention_dim, linear_units,
                    dropout_rate, activation, mlp_bias,
                    n_expert=n_expert, n_expert_activated=n_expert_activated
                )
            else:
                gated_x_attn, ff = None, None
            self.decoders.append(DecoderLayer(
                attention_dim,
                self_attn=self_attn,
                src_attn=src_attn_audio,
                feed_forward=mlp,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
                add_gated_x_attn=add_gated_x_attn,
                gated_x_attn=gated_x_attn,
                ff=ff,
            ))
        # # # import ipdb; ipdb.set_trace()    
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding
        self.use_sdpa = use_sdpa
        
    def forward(
        self,
        memory_a: torch.Tensor,           # (B, T_a, D)  audio features
        memory_a_mask: torch.Tensor,      # (B, 1, T_a)  audio mask
        memory_v: torch.Tensor,           # (B, T_v, D)  visual features
        memory_v_mask: Optional[torch.Tensor],  # (B, 1, T_v), optional
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor = torch.empty(0),        
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # import ipdb; ipdb.set_trace()
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L) -> (B, L, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        # import ipdb; ipdb.set_trace()
        if self.use_sdpa:
            tgt_mask = mask_to_bias(tgt_mask, memory_a.dtype)
            memory_a_mask = mask_to_bias(memory_a_mask, memory_a.dtype)
            if memory_v_mask is not None:
                memory_v_mask = mask_to_bias(memory_v_mask, memory_v.dtype)

        x, _ = self.embed(tgt)  # (B, L, D)
        if self.gradient_checkpointing and self.training:
            x = self.forward_layers_checkpointed(x, tgt_mask, memory_a, memory_a_mask,
                                                 memory_v, memory_v_mask)
        else:
            x = self.forward_layers(x, tgt_mask, memory_a, memory_a_mask,
                                    memory_v, memory_v_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_layers(
        self, 
        x: torch.Tensor, 
        tgt_mask: torch.Tensor,
        memory_a: torch.Tensor, 
        memory_a_mask: torch.Tensor,
        memory_v: torch.Tensor, 
        memory_v_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory_a, memory_a_mask= layer(
                x, tgt_mask, memory_a, memory_a_mask,
                memory_v, memory_v_mask,
            )
        return x

    @torch.jit.unused
    def forward_layers_checkpointed(
        self, 
        x: torch.Tensor, 
        tgt_mask: torch.Tensor,
        memory_a: torch.Tensor, 
        memory_a_mask: torch.Tensor,
        memory_v: torch.Tensor, 
        memory_v_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory_a, memory_a_mask = ckpt.checkpoint(
                layer.__call__,
                x, tgt_mask, memory_a, memory_a_mask,
                memory_v, memory_v_mask,
                use_reentrant=False,
            )
        return x

    def forward_one_step(
        self,
        memory_a: torch.Tensor,
        memory_a_mask: torch.Tensor,
        memory_v: torch.Tensor,
        memory_v_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Dict[str, Dict[str, T_CACHE]],
    ) -> torch.Tensor:
        x, _ = self.embed(tgt)
        # init cache
        if 'self_att_cache' not in cache: cache['self_att_cache'] = {}
        if 'cross_att_cache_a' not in cache: cache['cross_att_cache_a'] = {}
        if 'cross_att_cache_v' not in cache: cache['cross_att_cache_v'] = {}

        update_cross_att_cache_a = (len(cache['cross_att_cache_a']) == 0)
        update_cross_att_cache_v = (len(cache['cross_att_cache_v']) == 0)
        
        for i, decoder in enumerate(self.decoders):
            layer_i = f'layer_{i}'
            c = dict(
            self_att_cache=cache['self_att_cache'].get(layer_i, None),
            cross_att_cache_a=cache['cross_att_cache_a'].get(layer_i, None),
            cross_att_cache_v=cache['cross_att_cache_v'].get(layer_i, None),
            )
            x, tgt_mask, memory_a, memory_a_mask = decoder(
                x, tgt_mask, memory_a, memory_a_mask,
                memory_v, memory_v_mask, cache=c
            )
            # update cache dict
            assert c['self_att_cache'] is not None           
            cache['self_att_cache'][layer_i] = c['self_att_cache']
            
            if update_cross_att_cache_a and c.get('cross_att_cache_a', None) is not None:
                cache['cross_att_cache_a'][layer_i] = c['cross_att_cache_a']
            if update_cross_att_cache_v and c.get('cross_att_cache_v', None) is not None:
                cache['cross_att_cache_v'][layer_i] = c['cross_att_cache_v']

        y = self.after_norm(x[:, -1]) if self.normalize_before else x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y

    def tie_or_clone_weights(self, jit_mode: bool = True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        rank = int(os.environ.get('RANK', 0))
        if not self.use_output_layer:
            return
        if not self.tie_word_embedding:
            return
        if jit_mode:
            if rank == 0:
                logging.info("clone emb.weight to output.weight")
            self.output_layer.weight = torch.nn.Parameter(
                self.embed[0].weight.clone())
        else:
            if rank == 0:
                logging.info("tie emb.weight with output.weight")
            self.output_layer.weight = self.embed[0].weight

        if getattr(self.output_layer, "bias", None) is not None:
            self.output_layer.bias.data = torch.nn.functional.pad(
                self.output_layer.bias.data,
                (
                    0,
                    self.output_layer.weight.shape[0] -
                    self.output_layer.bias.shape[0],
                ),
                "constant",
                0,
            )


