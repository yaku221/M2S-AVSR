# M2S-AVSR: Modality-aware Multi-view Self-supervised Representation for Robust Audio-Visual Speech Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

Official implementation of **M2S-AVSR**, a Modality-aware Multi-view Self-supervised Representation framework for robust Audio-Visual Speech Recognition.

The framework leverages:

- Whisper for audio representation learning
- MVL Encoder initialized from AV-HuBERT Large for visual representation learning
- Multi-view self-supervised learning
- Modality-aware fusion

Supported datasets:

- AISHELL8-RealScene
- MISP2021-AVSR
- LRS3

---

# Model Overview

M2S-AVSR is built upon a dual-stream audio-visual architecture.

### Audio Encoder

The audio branch is initialized from Whisper and further fine-tuned on target AVSR datasets.

### Visual Encoder

The visual branch uses the proposed Multi-View Representation Learning (MVL) Encoder as the visual frontend. The MVL Encoder is initialized from AV-HuBERT Large and further optimized with multi-view self-supervised learning to improve robustness to viewpoint variations.


# Installation

```bash
conda create -n m2s_avsr python=3.10 -y
conda activate m2s_avsr

conda install conda-forge::sox

git clone https://github.com/yaku221/M2S-AVSR.git
cd M2S-AVSR

pip install torch==2.2.2+cu121 \
            torchaudio==2.2.2+cu121 \
            torchvision==0.17.2+cu121 \
            -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

cd ssl_models/av_hubert/fairseq
pip install --editable ./
```

---

# Data Preparation

The project supports:

- LRS3
- MISP2021-AVSR
- AISHELL8-RealScene

All datasets should be organized under:

```text
data/
├── lrs3/
├── misp2021/
└── aishell8_rs/
```

---

## LRS3

Please follow the AV-HuBERT preprocessing guide:

https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation

Organize data as:

```text
data/
└── lrs3/
    ├── audio/
    │   ├── train
    │   ├── dev
    │   └── eval
    └── video/
        ├── train
        ├── dev
        └── eval
```

---

## MISP2021-AVSR

Please follow the official MISP2021-AVSR preprocessing pipeline:

https://github.com/mispchallenge/MISP2021-AVSR.git

Organize data as:

```text
data/
└── misp2021/
    ├── audio/
    │   ├── train
    │   ├── dev
    │   └── eval_far
    └── video/
        ├── train
        ├── dev
        └── eval_far
```

After preprocessing, each split should contain:

```text
audio/
├── wav.scp
├── text
├── utt2spk
├── data.list

video/
├── data.list
```

---

## AISHELL8-RealScene

### Dataset Download

Download AISHELL8-RealScene from:

https://huggingface.co/datasets/SMIIP-lab/AISHELL8-RealScene

### Preprocessing

AISHELL8-RealScene follows the same preprocessing pipeline as MISP2021-AVSR.

Please refer to:

https://github.com/mispchallenge/MISP2021-AVSR.git

The dataset should be organized as:

```text
data/
└── aishell8_rs/
    ├── audio/
    │   ├── train
    │   ├── dev
    │   └── eval_far
    └── video/
        ├── train
        ├── dev
        └── eval
```

Example audio path:

```text
audio/dev/far/
L1_S004012013_G61_P5_Far_0.wav
```

Example video path:

```text
video/dev/
L1_S004012013_G61_P5_D0_004.avi
```

After preprocessing, generate:

```text
audio/*/data.list
video/*/data.list
```

which are directly used for training and decoding.

---

# Training

## Step 1. Whisper Fine-tuning

We first fine-tune Whisper-Large on the target AVSR dataset.

Please refer to the official WeNet Whisper example:

https://github.com/wenet-e2e/wenet/tree/main/examples/aishell/whisper


---

## Step 2. Train M2S-AVSR

Launch training:

```bash
bash train_m2s_avsr.sh
```

The training script internally launches:

```bash
torchrun \
    wenet/bin/train_avsr.py
```

Training checkpoints are saved to:

```text
exp/
```

---

## Step 3. Average Checkpoints

After training, average the best checkpoints:

```bash
bash average_models.sh
```

Default setting:

```text
average_num = 10
average_mode = step
```

---

# Inference

## Decode

Run:

```bash
bash decode_avsr.sh
```

Supported datasets:

```bash
dataset_name=aishell8_rs
```

or

```bash
dataset_name=misp2021
```

or

```bash
dataset_name=lrs3
```

The decoder internally calls:

```bash
python wenet/bin/recognize_avsr.py
```

---

## Evaluation

WER/CER are computed using WeNet official tools:

```bash
python tools/compute-wer.py
```

and

```bash
python tools/compute-cer.py
```

---


---

# Released Checkpoints

The available checkpoints are listed below.

| Module | Download Link |
|----------|----------|
| Audio Model | https://pan.baidu.com/s/1SxhHrEe3kevYWNjbunq8Zg?pwd=d353 |
| Video Model | https://pan.baidu.com/s/1HGeyAx3UJGKh_XMxrPRi7w?pwd=qhm6 |
| M2S-AVSR | https://pan.baidu.com/s/1woiJAyFzKEpPxQY-z0bayQ?pwd=jsdb |

---

# Citation

If you find this work useful, please cite:

```bibtex
@misc{su2026m2savsrmodalityawaremultiviewselfsupervised,
      title={M2S-AVSR: Modality-aware Multi-view Self-supervised Representation for Robust Audio-Visual Speech Recognition}, 
      author={Fei Su and Cancan Li and Ming Li and Juan Liu},
      year={2026},
      eprint={2606.05763},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2606.05763}, 
}
```

---

# Acknowledgments

This project is built upon the following excellent open-source projects:

- WeNet: https://github.com/wenet-e2e/wenet
- AV-HuBERT: https://github.com/facebookresearch/av_hubert
- Whisper: https://github.com/openai/whisper
- Whisper-Flamingo: https://github.com/roudimit/whisper-flamingo

---

# License

This project is released under the BSD-3-Clause License.

Please also follow the licenses of the original projects and datasets used in this work.