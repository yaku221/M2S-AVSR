# M2S-AVSR: Robust Audio-Visual Speech Recognition via Multi-stage Sparse Modality Alignment


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://chatgpt.com/c/6a21250b-1650-83a4-bb00-7b355b7cbe5e)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://chatgpt.com/c/6a21250b-1650-83a4-bb00-7b355b7cbe5e)

Official implementation of **M2S-AVSR**, a Modality-aware Multi-view Self-supervised representation framework for robust Audio-Visual Speech Recognition.


## Installation

### Environment

```bash
conda create -n m2s_avsr python=3.10
conda activate m2s_avsr
```

### Clone Repository

```bash
git clone https://github.com/yaku221/M2S-AVSR.git
cd M2S-AVSR
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

------

## Data Preparation

### LRS3

Download LRS3 from the official website and organize data as:

```text
data/
└── lrs3/
    ├── train/
    ├── val/
    └── test/
```

### MISP2021

```text
data/
└── misp2021/
```

### RealAV-1

```text
data/
└── realav1/
    ├── audio/
    ├── video/
    └── transcription/
```

------



## Training

### Single GPU

```bash
bash run.sh
```

### Multi-GPU

```bash
torchrun \
  --nproc_per_node=8 \
  train.py \
  --config conf/m2s_avsr.yaml
```

------

## Inference

```bash
python recognize.py \
  --config conf/m2s_avsr.yaml \
  --checkpoint exp/m2s_avsr/final.pt
```

------

## Model Zoo

Coming Soon

------

## Citation

If you find this work useful, please cite:

```bibtex
@article{m2savsr2026,
  title={M2S-AVSR: Robust Audio-Visual Speech Recognition via Multi-stage Sparse Modality Alignment},
  author={Fei Su, Cancan Li, Ming Li, and Juan Liu},
  journal={arXiv preprint},
  year={2026}
}
```

------

