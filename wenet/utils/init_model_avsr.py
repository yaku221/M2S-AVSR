import os
import torch
import logging


DEFAULT_CONFIG = {
    "model": "m2s_avsr",
    "input_dim": 80,
    "output_dim": 51865,
    "vocab_size": 51865,

    "encoder_conf": {
        "activation_type": "gelu",
        "attention_dropout_rate": 0.0,
        "attention_heads": 16,
        "dropout_rate": 0.0,
        "fusion_stage": "early",
        "gradient_checkpointing": True,
        "input_layer": "conv1d2",
        "key_bias": False,
        "linear_units": 4096,
        "lip_reader_layers_num": 3,
        "normalize_before": True,
        "num_blocks": 24,
        "output_size": 1024,
        "pos_enc_layer_type": "abs_pos_whisper",
        "positional_dropout_rate": 0.0,
        "prob_a": 0.0,
        "prob_av": 0.5,
        "static_chunk_size": -1,
        "use_av_hubert_encoder": True,
        "use_dynamic_chunk": False,
        "use_dynamic_left_chunk": False,
    },
    "decoder_conf": {
        "activation_type": "gelu",
        "attention_heads": 16,
        "dropout_rate": 0.0,
        "gradient_checkpointing": True,
        "input_layer": "embed_learnable_pe",
        "key_bias": False,
        "linear_units": 4096,
        "normalize_before": True,
        "num_blocks": 24,
        "positional_dropout_rate": 0.0,
        "self_attention_dropout_rate": 0.0,
        "src_attention": True,
        "src_attention_dropout_rate": 0.0,
        "src_key_bias": True,
        "src_key_bias_v": True,
        "tie_word_embedding": True,
        "use_output_layer": True,
    },
    "ctc": "ctc",
    "ctc_conf": {
        "ctc_blank_id": 50256
    },

    "model_conf": {
        "ctc_weight": 0.0,
        "length_normalized_loss": False,
        "lsm_weight": 0.1,
    },

    "fusion_method": "gated_fusion",
    "freeze_video_model": True,
}


from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules

from wenet.transformer.avsr_model import M2S_AVSR_Model
from wenet.transformer.encoder_avsr import M2S_AVSR_Encoder
from wenet.transformer.decoder_avsr import M2S_AVSR_Decoder

AVSR_MODEL_CLASSES = {
    "m2s_avsr": M2S_AVSR_Model,
}


def _merge_default_config(configs):
    if configs is None:
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()

    for k, v in configs.items():
        if isinstance(v, dict) and k in merged:
            merged[k].update(v)
        else:
            merged[k] = v

    return merged



def build_avsr_model(args, configs):

    configs = _merge_default_config(configs)

    global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    model_type = configs.get('model', 'm2s_avsr')

    if model_type not in AVSR_MODEL_CLASSES:
        raise ValueError(f"Unsupported model_type={model_type}")


    video_model_path = (configs.get('init_models') or {}).get('video_model_ckpt')

    encoder = M2S_AVSR_Encoder(
        input_dim,
        global_cmvn=global_cmvn,
        video_model_path=video_model_path,
        **configs['encoder_conf']
    )


    fusion_method = (configs.get('fusion_method') or '').lower()
    add_gated_x_attn = (fusion_method == 'gated_fusion')

    decoder = M2S_AVSR_Decoder(
        vocab_size,
        encoder.output_size(),
        add_gated_x_attn=add_gated_x_attn,
        **configs['decoder_conf'],
    )


    ctc = CTC(
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
    )


    model = AVSR_MODEL_CLASSES[model_type](
        vocab_size=vocab_size,
        video_frontend=None,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        special_tokens=configs.get("tokenizer_conf", {}).get("special_tokens", None),
        **configs['model_conf']
    )

    return model, configs



def init_model_avsr(args, configs=None):

    configs = _merge_default_config(configs)

    model, configs = build_avsr_model(args, configs)

    infos = {}

    
    if configs['model'] == 'm2s_avsr':
        audio_ckpt = (configs.get('init_models') or {}).get('audio_model_ckpt')

        if audio_ckpt:
            try:
                # state = torch.load(audio_ckpt, map_location='cpu')
                state = torch.load(
                    audio_ckpt,
                    map_location='cpu',
                    weights_only=True
                )

                state_dict = state.get('state_dict', state) \
                    if isinstance(state, dict) else state

                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {
                        k.replace('module.', '', 1): v
                        for k, v in state_dict.items()
                    }

                incompatible_info = model.load_state_dict(
                    state_dict, strict=False
                )

                print(f"[Init] audio ckpt: {audio_ckpt}")

            except Exception as e:
                logging.warning("Load audio ckpt failed: %s", e)

   
    if getattr(args, 'checkpoint', None):
        ckpt_infos = load_checkpoint(model, args.checkpoint) or {}
        infos.update(ckpt_infos)

    configs["init_infos"] = infos


    if configs.get('freeze_video_model', False):
        if hasattr(model.encoder, 'video_model'):
            for p in model.encoder.video_model.parameters():
                p.requires_grad = False
            print("[Info] Video model frozen")

    
    if configs.get('fusion_method') == 'gated_fusion':
        for n, p in model.encoder.named_parameters():
            if "video_projection" not in n:
                p.requires_grad = False

    return model, configs


