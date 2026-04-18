# LLM-Marketing

Fine-tuning **DeepSeek-R1** on large-scale scraped social media data to build an automated marketing campaign generation system for pet food brands.

## Overview

This project is the second component of a two-part pipeline:

1. **Data Collection** — Multi-platform web scraping (Instagram, Facebook, LinkedIn, Twitter/X, Xiaohongshu) using Selenium + Scrapy, collecting posts, engagement metrics, and media assets. See the [web-scrapers](https://github.com/sichenz/web-scrapers) repo.
2. **LLM Fine-tuning** *(this repo)* — Supervised fine-tuning of DeepSeek-R1-Distill-Llama-8B on scraped posts reformatted as instruction-response pairs, enabling the model to generate platform-native marketing campaigns on demand.

## Pipeline

```
scraped_posts.csv
       │
       ▼
preprocess.py          ← EDA, cleaning, engagement labeling, feature extraction
       │
       ▼
finetune.py            ← Instruction-pair construction → LoRA fine-tuning → FP8 quantization
       │
       ▼
DeepSeek-R1-FP8        ← Quantized model (~4× smaller, inference-ready)
       │
       ▼
evaluate.py            ← BLEU-4, ROUGE-L, hashtag recall, emoji density scoring
```

## Data Format

Each scraped post is converted into a structured instruction-response pair:

```json
{
  "instruction": "Write a Xiaohongshu marketing post for the brand 'Royal Canin'. Post title: '湿粮体验官|猫咪挑食怎么办'. Aim for high engagement (reference: 499 likes). Post type: image.",
  "response": "别人的猫：吃嘛嘛香 干饭第一名 挑食小猫：只吃冻干 闻闻就走开..."
}
```

The system prompt instructs the model to write in the brand's warm, community-focused voice with emojis, a CTA, and hashtags.

## Training Details

| Parameter         | Value                                    |
|------------------|------------------------------------------|
| Base model        | DeepSeek-R1-Distill-Llama-8B             |
| Fine-tuning method| LoRA (r=16, α=32, target: q/k/v/o_proj) |
| Trainable params  | ~0.5% of total                           |
| Dataset           | ~100 Xiaohongshu posts (Royal Canin)     |
| Epochs            | 3                                        |
| Batch size        | 2 (grad accum × 4 = effective 8)         |
| Learning rate     | 2e-4 (cosine schedule)                   |
| Compute           | NYU HPC — 1× NVIDIA A100 (80GB)         |
| Quantization      | FP8 Dynamic (llmcompressor)              |

## Evaluation

Generated campaigns are scored against held-out reference posts on:
- **BLEU-4** — n-gram overlap with real brand posts
- **ROUGE-L** — longest common subsequence recall
- **Hashtag recall** — fraction of relevant brand hashtags reproduced
- **Emoji density** — emojis per 100 characters (brand style proxy)

## Setup

```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
```

## Usage

```bash
# 1. Preprocess & EDA
python preprocess.py

# 2. Fine-tune + quantize (on HPC)
sbatch tune.sbatch

# 3. Evaluate
python evaluate.py --model_path ./DeepSeek-R1-Distill-Llama-8B-FP8-dynamic \
                   --n_samples 50
```

## Requirements

```
transformers>=4.40
datasets
torch>=2.1
peft
llmcompressor
pandas
rouge-score
nltk
```
