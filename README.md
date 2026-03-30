# EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation

[![arXiv](https://img.shields.io/badge/arXiv-2603.12108-b31b1b.svg)](https://arxiv.org/abs/2603.12108)


## Highlights

🔗 **Unified & Consistent**: Reconciles the "granularity gap" by maintaining task-decoupled yet consistent representations within a shared space via residual evolution.

📊 **Data Efficient**: Trained entirely on **fully open-source data (w/o any re-captioning or distillation)**. Despite using only 13M images, EvoTok achieves 0.43 rFID on ImageNet-1K (256×256).

🚀 **Versatile Performance**: Competitive results across 7/9 understanding benchmarks and leading generation benchmarks (e.g., GenEval, GenAI-Bench).

## Introduction

EvoTok is a unified image tokenizer designed to bridge the gap between high-level semantic and low-level pixel features. By modeling visual features as an evolutionary trajectory, EvoTok achieves strong performance in both domains within a single, shared latent space.

![framework](assets/framework.png)


EvoTok delivers superior performance in multimodal understanding and text-to-image generation. For multimodal understanding, EvoTok achieves strong performance across a wide range of benchmarks. For text-to-image generation, it produces high-quality images at 256×256 resolution with strong visual fidelity and semantic alignment across diverse visual domains (portraits, landscapes, objects, etc.).

![gen](assets/generation_examples.png)


## Installation

```bash
conda create -n evotok python==3.12 -y
conda activate evotok
pip install -r requirements.txt

# Install FlashAttention
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

## Dataset

We follow the data format of [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) to organize our dataset.

For understanding, `<image>` appears in the human turn as input:
```json
[{"from": "human", "value": "description\n<image>"},
 {"from": "gpt",   "value": "answer."}]
```

For generation, `<image>` appears in the assistant turn as the target:
```json
[{"from": "human", "value": "description"},
 {"from": "gpt",   "value": "<image>"}]
```

## Training

### EvoTok Tokenizer

```bash
bash evotok/scripts/train_evotok_siglip256.sh
```

### Multimodal Understanding

```bash
# Stage 1: Pretraining
bash mllm/scripts/pretrain_understanding.sh
# Stage 2: Supervised Fine-tuning
bash mllm/scripts/sft_understanding.sh
```

### Text-to-Image Generation

```bash
# Stage 1: Pretraining
bash mllm/scripts/pretrain_generation.sh
# Stage 2: Supervised Fine-tuning
bash mllm/scripts/sft_generation.sh
```

## Evaluation

```bash
bash evotok/scripts/reconstruction_rq.sh
```

## Acknowledgement

This project builds upon several excellent open-source works: [RQ-VAE](https://github.com/kakaobrain/rq-vae-transformer), [VILA-U](https://github.com/mit-han-lab/vila-u), [TokenFlow](https://github.com/ByteVisionLab/TokenFlow), and [LLaVA](https://github.com/haotian-liu/LLaVA). We sincerely thank the authors for their contributions.


## Citation

```
@misc{li2026evotok,
      title={EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation}, 
      author={Yan Li and Ning Liao and Xiangyu Zhao and Shaofeng Zhang and Xiaoxing Wang and Yifan Yang and Junchi Yan and Xue Yang},
      year={2026},
      eprint={2603.12108},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12108}, 
}
```
