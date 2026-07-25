# GPT-2 from Scratch in PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformer](https://img.shields.io/badge/Architecture-GPT--2-6a5acd.svg)](https://en.wikipedia.org/wiki/GPT-2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a compact and modular implementation of the GPT-2 architecture in PyTorch. The project focuses on reproducing the core ideas behind GPT-2 from first principles, including causal self-attention, transformer blocks, positional embeddings, and autoregressive language modeling.

The implementation is intended for learning, experimentation, and technical review. It is written to be readable, structured, and reasonably close to practical training code.

## Overview

The project reconstructs the GPT-2 decoder-only transformer using PyTorch and organizes the implementation into separate modules for model definition, data loading, training, inference, and utilities. The training pipeline includes gradient accumulation, checkpointing, learning-rate scheduling, and optional mixed precision.

## Features

- From-scratch implementation of the GPT-2 decoder-only transformer
- Causal self-attention with scaled dot-product attention
- Pre-LayerNorm residual blocks and GELU-based MLP layers
- Learned positional embeddings and tied input/output embeddings
- Data preparation pipeline for tokenizing and storing training shards
- Training loop with gradient accumulation and checkpoint saving
- Inference script for text generation with Top-K sampling
- Mixed precision support and Torch compilation in training

## Project Structure

```text
GPT/
├── scripts/
│   └── prepare_data.py
├── src/
│   ├── data.py
│   ├── infer.py
│   ├── model.py
│   ├── utils.py
├── train.py
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch with CUDA support is recommended for training

### Setup

```bash
python -m venv venv
source venv/bin/activate
# On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

The repository uses the Hugging Face FineWeb-EDU dataset as training data. The preprocessing script tokenizes text using `tiktoken` with the GPT-2 encoding and stores the resulting token sequences as shard files.

Example:

```bash
python scripts/prepare_data.py \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_config "sample-10BT" \
  --output_dir "edu_fineweb10B" \
  --max_docs 10000 \
  --shard_size 1000000
```

This produces shard files under the specified output directory, with validation and training splits created from the first shard onward.

## Training

The main training entry point is `train.py`.

Example:

```bash
python train.py \
  --data_dir edu_fineweb10B \
  --log_dir checkpoint \
  --batch_size 16 \
  --block_size 1024 \
  --total_batch_size 524288 \
  --max_steps 50 \
  --max_lr 6e-4
```

For multi-GPU training, the script can also be launched with `torchrun`.

## Inference

To generate text from a saved checkpoint:

```bash
python src/infer.py \
  --checkpoint checkpoint/model_00049.pt \
  --prompt "Hello, I'm a language model," \
  --max_length 64 \
  --num_sequences 4
```

## Model Architecture

The default configuration is aligned with a GPT-2-style 124M parameter setup.

| Parameter | Value |
| :--- | :--- |
| Vocabulary size | 50,257 (padded to 50,304 for alignment) |
| Context length | 1024 tokens |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Training target | 0.5M tokens via gradient accumulation |

### Key implementation details

- Layer normalization is applied before attention and MLP blocks.
- Residual projections use scaled initialization for stability.
- Token embeddings are tied to the output language modeling head.

## Technologies Used

- PyTorch
- `tiktoken`
- Hugging Face Datasets
- NumPy
- `tqdm`

## Implementation Notes

- The training loop uses mixed precision with `bfloat16` where supported.
- `torch.compile` is used in the training path for performance-oriented execution.
- Checkpoint loading removes the `_orig_mod.` prefixes introduced by compilation to preserve compatibility.

## Future Work

Possible next steps include:

- Adding KV-cache support for faster inference
- Adding evaluation benchmarks such as HellaSwag
- Supporting loading public pretrained GPT-2 weights
- Improving the training and data pipeline for larger runs

## License

This project is licensed under the MIT License.
