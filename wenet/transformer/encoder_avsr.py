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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import torch.nn as nn
from torch import Tensor

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer_avsr import (ConformerEncoderLayer,
                                             TransformerEncoderLayer)
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_EMB_CLASSES, WENET_MLP_CLASSES,
                                     WENET_NORM_CLASSES,
                                     WENET_SUBSAMPLE_CLASSES)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import add_optional_chunk_mask, make_pad_mask
from wenet.models.resnet import ResEncoder


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class M2S_AVSR_Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        activation_type: str = "relu",
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        selfattention_layer_type: str = "selfattn",
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        final_norm: bool = True,
        prob_av: float = 1.0,
        prob_a: float = 0.0,
        video_model_path: str = "",
        av_hubert_path: str = "ssl_models/av_hubert/avhubert",
        use_av_hubert_encoder: bool = False,
        fusion_stage: str = "early",
        fusion_method: str = "gated_fusion",
        lip_reader_layers_num: int = 3,
    ):
        
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                attention_heads, positional_dropout_rate))

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.final_norm = final_norm
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size,
                                                              eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa
        
        # audio transformer encoder
        assert selfattention_layer_type in ['selfattn', 'rope_abs_selfattn']
        attn_cls = WENET_ATTENTION_CLASSES[selfattention_layer_type]
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                attn_cls(attention_heads, output_size, attention_dropout_rate,
                    query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                    head_dim),
                mlp_class(output_size,
                          linear_units,
                          dropout_rate,
                          activation,
                          mlp_bias,
                          n_expert=n_expert,
                          n_expert_activated=n_expert_activated),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])

        
        # init av modules
        self.whisper_positional_embedding = pos_emb_class(
            d_model=output_size, dropout_rate=0.0, max_len=1500           
        )        
        self.prob_av, self.prob_a = prob_av, prob_a
        self.use_av_hubert_encoder = use_av_hubert_encoder
        self.fusion_stage = fusion_stage
        self.video_model_path = video_model_path
        
        self.video_projection_scalar = nn.Parameter(torch.tensor(1.))
  
        if not use_av_hubert_encoder:
            self.video_projection = Linear(512, output_size)
            self.video_model = ResEncoder('prelu', video_model_path)
        else:
            from fairseq import checkpoint_utils, utils
            from argparse import Namespace
            self.video_projection = Linear(1024, output_size) # assuming AV-HuBERT large model
            utils.import_user_module(Namespace(user_dir=av_hubert_path))
            # print("Loading AV-HuBERT encoder")
            load_weights = False if "no_weights" in video_model_path else True
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([video_model_path],) 
                                                                                    # load_weights=load_weights)
            self.video_model = models[0].encoder if 'ft' in video_model_path else models[0]
            num_parameters = sum(p.numel() for p in self.video_model.parameters())
            # print("Using AV-HuBERT encoder with parameters: {}".format(num_parameters)) 
        if self.fusion_stage == "lip-reader":
            attn_cls = WENET_ATTENTION_CLASSES[selfattention_layer_type]
            activation = WENET_ACTIVATION_CLASSES[activation_type]()
            mlp_class = WENET_MLP_CLASSES[mlp_type]
            self.video_projection_blocks = torch.nn.ModuleList([
                TransformerEncoderLayer(
                    output_size,
                    attn_cls(attention_heads, output_size, attention_dropout_rate,
                        query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                        head_dim),
                    mlp_class(output_size,
                            linear_units,
                            dropout_rate,
                            activation,
                            mlp_bias,
                            n_expert=n_expert,
                            n_expert_activated=n_expert_activated),
                    dropout_rate,
                    normalize_before,
                    layer_norm_type=layer_norm_type,
                    norm_eps=norm_eps,
                ) for _ in range(lip_reader_layers_num)
            ])
            num_parameters = sum(p.numel() for p in self.video_projection_blocks.parameters())
            print("Adding visual transformer layers with number of params: {}".format(num_parameters)) 


        # import ipdb; ipdb.set_trace()
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        xv: torch.Tensor,
        xv_lengths: torch.Tensor,
        track_norm: bool = False,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ):
        # import ipdb; ipdb.set_trace()
        # xs: [B, Ts, D]
        # xv: [B, Tv, H, W, C]

        T = xs.size(1)  # T_audio
        memory_a_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, Ts)
        
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, memory_a_mask = self.embed(xs, memory_a_mask)
        
        if track_norm:
            xs_norm = torch.linalg.norm(xs, dim=-1).mean()
            
        ### extract vide features
        if xv.dim() == 5 and xv.shape[-1] in (1, 3):      # C=1 or 3
            xv = xv.permute(0, 4, 1, 2, 3).contiguous()   # [B, Tv, H, W, C] -> [B, C, Tv, H, W]

        assert xv.dim() == 5, f"xv expected 5dim video tensor, got {xv.dim()}dim"
        assert xv.shape[1] in (1, 3), f"xv expected channels-first with C=1/3, got C={xv.shape[1]}"
        
        Tv = xv.size(2)
        video_pad_mask_in = make_pad_mask(xv_lengths, Tv)  # (B, Tv) for avhubert, True=pad

        if not self.use_av_hubert_encoder:
                xv = self.video_model(xv) # B, F, T
                xv = xv.permute(0, 2, 1).contiguous()  # B, T, F
        elif 'ft' not in self.video_model_path: # AV-HuBERT ssl
            xv = self.video_model(source={'video': xv, 'audio': None}, 
                                padding_mask=video_pad_mask_in, 
                                mask=False, 
                                features_only=True)
            xv = xv['x']   # (B, T, F)
        else:
            xv = self.video_model(
                source={'video': xv, 'audio': None}, 
                padding_mask=video_pad_mask_in)
            xv = xv['encoder_out'].permute(1, 0 , 2).contiguous()  # T, B, F -> B, T, F
            
        if track_norm:
            xv_norm_pre = torch.linalg.norm(xv, dim=-1).mean()

        if self.fusion_stage == "lip-reader":
            xv = torch.repeat_interleave(xv, 2, dim=1) # 25 Hz -> 50 Hz
            new_xv_lens = xv_lengths * 2
        else:
            new_xv_lens = xv_lengths.clone()

        # import ipdb; ipdb.set_trace()    
        # video_projection     
        xv = self.video_projection(xv)   # [B, T, output_size]
        xv = self.video_projection_scalar * xv
        
        if self.fusion_stage == "lip-reader":
            # NOTE: pos embedding added before
            if xv.shape[1] > 1500:
                xv = xv[ :, :1500, :]
                
        new_xv_lens = torch.clamp(new_xv_lens, max=xv.shape[1])
        memory_v_mask = ~make_pad_mask(new_xv_lens, xv.size(1)).unsqueeze(1)  # (B, 1, T_v)，True=no pad    
        
        # Defaults: use audio embeddings/pos/mask for the encoder
        enc_pos = pos_emb
        enc_mask_pad = memory_a_mask

        if self.fusion_stage == "lip-reader":    
            # NOTE: if max_len is 30s, then the cropping doesn't do anything.
            pos_v = self.whisper_positional_embedding.position_encoding(
                offset=0, size=xv.shape[1], apply_dropout=False).to(xv.dtype).to(xv.device)
            xv = xv + pos_v

            mask_v = memory_v_mask.squeeze(1)                # (B, T)
            mask_v = mask_v.unsqueeze(1) & mask_v.unsqueeze(2)  # (B, T, T)

            for block in self.video_projection_blocks:
                xv, mask_v, _, _ = block(xv, mask_v, pos_v, memory_v_mask)

            xs = xv # NOTE: use AV-HuBERT output as input
            enc_pos = pos_v
            enc_mask_pad = memory_v_mask
        
        if track_norm:
            xv_norm_post = torch.linalg.norm(xv, dim=-1).mean()

        assert xv.dim() == 3 and xv.shape[0] == xs.shape[0], \
            f"AV features must be [B, T, F]; got xs: {xs.shape}, xv: {xv.shape}"
        
        mask_pad = enc_mask_pad    # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            enc_mask_pad,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate))
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)

        # encoders forward    
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, enc_pos, 
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, enc_pos, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        # import ipdb; ipdb.set_trace()
        # modality drop_out
        xs, xv = self._modality_dropout(xs, xv, self.training)
        # import ipdb; ipdb.set_trace()
        if track_norm:
            return xs, enc_mask_pad, xv, memory_v_mask, xs_norm, xv_norm_pre, xv_norm_post
        return xs, enc_mask_pad, xv, memory_v_mask


    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
    
    
    def _modality_dropout(self, xa: torch.Tensor, xv: Optional[torch.Tensor], training: bool):
        if (not training) or (xv is None):
            return xa, xv
        if (self.prob_av <= 0.0) and (self.prob_a <= 0.0):
            return xa, xv

        r = torch.rand((), device=xa.device).item()
        if 0.0 < r <= self.prob_av:
            return xa, xv
        elif self.prob_av < r <= self.prob_av + self.prob_a:
            return xa, xv * 0
        else:
            return xa * 0, xv

    @torch.jit.unused
    def forward_layers_checkpointed(self, xs: torch.Tensor,
                                    chunk_masks: torch.Tensor,
                                    pos_emb: torch.Tensor,
                                    mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = ckpt.checkpoint(layer.__call__,
                                                    xs,
                                                    chunk_masks,
                                                    pos_emb,
                                                    mask_pad,
                                                    use_reentrant=False)
        return xs

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i:i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :,
                                                                   size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache,
             cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                             required_cache_size, att_cache,
                                             cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)),
                           device=ys.device,
                           dtype=torch.bool)
        return ys, masks










