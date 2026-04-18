# LLM Marketing Campaign Generator

Fine-tune Large Language Models on social media post data using **LoRA** and **Colossal-AI** to automatically generate marketing campaigns.

---

## Overview

This pipeline fine-tunes HuggingFace LLMs on scraped social media posts (e.g., Xiaohongshu / 小红书) using **LoRA (Low-Rank Adaptation)** for memory-efficient training. It automatically detects your hardware and adapts accordingly:

- **Mac (Apple Silicon)** — Uses the MPS GPU backend for accelerated local fine-tuning
- **NVIDIA GPU** — Uses Colossal-AI's GeminiPlugin for ZeRO-3 distributed training
- **CPU** — Falls back gracefully for machines without a GPU

The default small model (`Qwen/Qwen1.5-0.5B`) runs comfortably on consumer hardware and produces strong results on small datasets (~100 posts).

---

## Repository Structure

```
LLM-Marketing/
├── data/
│   └── data.csv               ← Social media posts dataset (CSV)
├── dataset.py                 ← PyTorch Dataset — formats posts into training prompts
├── model_utils.py             ← Loads HuggingFace models and tokenizers
├── train.py                   ← Main fine-tuning script (LoRA + Colossal-AI)
├── run_test.sh                ← Quick pipeline verification script
├── requirements.txt           ← Python dependencies
└── README.md
```

---

## How It Works

1. **Dataset (`dataset.py`)** — Reads a CSV containing social media posts and formats each row into a causal LM training prompt using the `Author Name`, `Post Title`, and `Caption` fields. Padding tokens are masked with `-100` so they are ignored in loss computation.

2. **Models (`model_utils.py`)** — Downloads real pre-trained models from HuggingFace. Automatically selects the best precision for your hardware (`float16` for NVIDIA, `bfloat16` for Apple MPS, `float32` for CPU).

3. **Training (`train.py`)** — The core pipeline:
   - Wraps the base model in a **LoRA adapter** (rank 32, targeting all projection layers) so only ~3% of parameters are trained
   - Uses **AdamW** optimizer with **cosine learning rate scheduling** and linear warmup
   - Implements **gradient accumulation** (default 4 steps) for a larger effective batch size
   - **Best-checkpoint saving** — only saves the model when average epoch loss improves
   - Supports `python train.py` directly (no `torchrun` required for single-node training)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the Pipeline

Run a quick pipeline test to ensure everything works:

```bash
bash run_test.sh
```

### 3. Fine-Tune a Model

```bash
python train.py --model_size small --batch_size 2 --epochs 10
```

The trained model (LoRA adapter weights + tokenizer) will be saved to `output_model/`.

---

## Training Configuration

### Default Models

| `--model_size` | Model | Parameters | Best for |
|---|---|---|---|
| `small` | `Qwen/Qwen1.5-0.5B` | 464M | CPU / Mac GPU — fast, memory-efficient |
| `large` | `deepseek-ai/deepseek-coder-1.3b-base` | 1.3B | NVIDIA GPU / HPC — higher quality |

You can also specify any HuggingFace model directly with `--model_name`:

```bash
python train.py --model_name meta-llama/Llama-3-8B --use_gpu
```

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Trainable params | ~3.2% of total |

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_size` | `small` | Choose `small` (Qwen 0.5B) or `large` (DeepSeek 1.3B) |
| `--model_name` | `None` | Custom HuggingFace model ID (overrides `--model_size`) |
| `--batch_size` | `2` | Per-device batch size |
| `--grad_accum_steps` | `4` | Gradient accumulation steps (effective batch = batch_size × this) |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `2e-4` | Peak learning rate |
| `--warmup_ratio` | `0.1` | Fraction of total steps used for LR warmup |
| `--max_steps` | `-1` | Limit training steps (`-1` = train full epochs) |
| `--output_dir` | `output_model` | Directory to save the fine-tuned model |
| `--use_gpu` | `False` | Enable Colossal-AI GeminiPlugin for NVIDIA GPUs |

---

## Data Format

The training data CSV should contain at minimum these columns:

| Column | Description |
|---|---|
| `Author Name` | Post author / brand name |
| `Post Title` | Title of the social media post |
| `Caption` | Full post caption text (required — rows without this are dropped) |

Additional columns (likes, comments, images, etc.) are preserved in the CSV but not currently used for training.
