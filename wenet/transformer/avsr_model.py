# Copyright (c) 2025 Fei Su (shinji721@outlook.com)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict, List, Optional
from torch import nn, Tensor
import torch.nn.functional as F

from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.utils.common import IGNORE_ID, add_whisper_tokens, th_accuracy
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
# from wenet.transformer.search_avsr import (ctc_greedy_search,
#                                       ctc_prefix_beam_search,
#                                       attention_beam_search,
#                                       attention_rescoring, DecodeResult)
from wenet.transformer.search_avsr_nbest import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring, DecodeResult)
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph

from wenet.models.resnet import ResEncoder
# from espnet_modules.espnet2.asr.frontend.video_frontend import VideoFrontend
from wenet.transformer.attention import MultiHeadedAttention


class M2S_AVSR_Model(torch.nn.Module):
    """Modality-aware Fusion with Multi-View Self-supervised AVSR Model."""
    def __init__(
        self,
        vocab_size: int,
        # video_frontend: VideoFrontend,
        video_frontend: None,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
        apply_non_blank_embedding: bool = False,       
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert reverse_weight == 0.0   # whisper decoder based
        super().__init__()

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.apply_non_blank_embedding = apply_non_blank_embedding
        
        self.video_frontend = video_frontend
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        
        self.sos = special_tokens["sot"]
        self.eos = special_tokens["eot"]
        self.decode_maxlen = self.decoder.embed[1].max_len          
        

    # TODO(xcsong): time align
    def set_alignment_heads(self, dump: bytes):
        raise NotImplementedError

    @property
    def is_multilingual(self):
        return self.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.vocab_size - 51765 - int(self.is_multilingual)
    
    
    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
        track_norm: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        # import ipdb; ipdb.set_trace()
        # audio features
        speech = batch['feats'].to(device)   # [B, Ts, 128]
        speech_lengths = batch['feats_lengths'].to(device)  # [B]
        
        # video features
        video = batch['video'].to(device)   # [B, Tv, 88, 88, 1]
        video_lengths = batch['video_lengths'].to(device)   # [B]       
        
        # labeals       
        text = batch['target'].to(device)  # [B, Max_Length]
        text_lengths = batch['target_lengths'].to(device)  # [B]     

        # import ipdb; ipdb.set_trace()
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        
        # 1. video_frontend        
        if self.video_frontend is not None:
            video, video_lengths = self.video_frontend(video, video_lengths)         
        
        # 2. Encoder
        if track_norm:
            features_a, mask_a, features_v, mask_v, x_norm, x_v_norm_pre, x_v_norm_post = self.encoder(
                speech, speech_lengths, video, video_lengths, track_norm
            )
        else:
            features_a, mask_a, features_v, mask_v = self.encoder(
                speech, speech_lengths, video, video_lengths,
            )
        # import ipdb; ipdb.set_trace()
        encoder_out_lens = mask_a.squeeze(1).sum(1)

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(features_a, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None

        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            features_a, encoder_mask = self.filter_blank_embedding(
                ctc_probs, features_a)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                features_a, mask_a, features_v, mask_v, text, text_lengths, {
                    "langs": batch["langs"],
                    "tasks": batch["tasks"]
                })
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        if track_norm:
            return {
                "loss": loss,
                "loss_att": loss_att,
                "loss_ctc": loss_ctc,
                "th_accuracy": acc_att,
                "x_norm": x_norm, 
                "x_v_norm_pre": x_v_norm_pre, 
                "x_v_norm_post": x_v_norm_post,                
            }
        else:
            return {
                "loss": loss,
                "loss_att": loss_att,
                "loss_ctc": loss_ctc,
                "th_accuracy": acc_att,                
            }
            

    def tie_or_clone_weights(self, jit_mode: bool = True):
        self.decoder.tie_or_clone_weights(jit_mode)

    @torch.jit.unused
    def _forward_ctc(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                       text_lengths)
        return loss_ctc, ctc_probs

    def filter_blank_embedding(
            self, ctc_probs: torch.Tensor,
            encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        top1_index = torch.argmax(ctc_probs, dim=2)
        indices = []
        for j in range(batch_size):
            indices.append(
                torch.tensor(
                    [i for i in range(maxlen) if top1_index[j][i] != 0]))

        select_encoder_out = [
            torch.index_select(encoder_out[i, :, :], 0,
                               indices[i].to(encoder_out.device))
            for i in range(batch_size)
        ]
        select_encoder_out = pad_sequence(select_encoder_out,
                                          batch_first=True,
                                          padding_value=0).to(
                                              encoder_out.device)
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size)
                                ]).to(encoder_out.device)
        T = select_encoder_out.size(1)
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    def _calc_att_loss(
        self,
        features_a: torch.Tensor,
        mask_a: torch.Tensor,        
        features_v: torch.Tensor,
        mask_v: torch.Tensor,        
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]],
    ) -> Tuple[torch.Tensor, float]:
        prev_len = ys_pad.size(1)
        ys_in_pad, ys_out_pad = add_whisper_tokens(self.special_tokens,
                                                   ys_pad,
                                                   self.ignore_id,
                                                   tasks=infos['tasks'],
                                                   no_timestamp=True,
                                                   langs=infos['langs'],
                                                   use_prev=False)
        cur_len = ys_in_pad.size(1)
        ys_in_lens = ys_pad_lens + cur_len - prev_len
        # import ipdb; ipdb.set_trace()
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(features_a, mask_a, 
                                                     features_v, mask_v, 
                                                     ys_in_pad, ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att


    def _forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            video: torch.Tensor,
            video_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ):
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            features_a, mask_a, features_v, mask_v = self.encoder(
                speech,
                speech_lengths,
                video,
                video_lengths
            )  # (B, maxlen, encoder_dim)
        return features_a, mask_a, features_v, mask_v

    # The same interface just like whisper
    # see https://github.com/openai/whisper/blob/main/whisper/model.py#L287
    def embed_audio(
        self,
        mel: torch.Tensor,
        mel_len: torch.Tensor,
        chunk_size: int = -1,
    ) -> [torch.Tensor, torch.Tensor]:
        encoder_out, encoder_mask = self.encoder(mel, mel_len, chunk_size)
        return encoder_out, encoder_mask

    @torch.jit.unused
    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        features_a, mask_a, features_v, mask_v = self._forward_encoder(
            speech, speech_lengths, video, video_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        encoder_out_lens = mask_a.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(features_a, blank_penalty, blank_id)
        results = {}
        if 'attention' in methods:
            results['attention'] = attention_beam_search(
                self,features_a, mask_a, features_v, mask_v, beam_size, length_penalty,
                infos)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size,
                                                       context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(
                self, ctc_prefix_result, encoder_out, encoder_lens, ctc_weight,
                reverse_weight, infos)
        return results

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

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
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
