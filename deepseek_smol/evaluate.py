"""
evaluate.py — Evaluate generated marketing campaigns against reference posts.

Metrics:
    - BLEU-4          : n-gram overlap with reference posts
    - ROUGE-L         : longest common subsequence recall
    - Hashtag recall  : fraction of relevant hashtags included
    - Emoji density   : emojis per 100 characters (brand style proxy)
    - Length ratio    : generated / reference length (fluency proxy)

Usage:
    python evaluate.py --model_path ./DeepSeek-R1-Distill-Llama-8B-FP8-dynamic \
                       --data_path  /scratch/sz4972/LLM-Marketing/data/processed.csv \
                       --n_samples  50
"""

import argparse
import json
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

SYSTEM_PROMPT = (
    "You are an expert social media marketing copywriter for a pet food brand. "
    "Given a product name, platform, and target audience, write an engaging "
    "social media post in the brand's warm, community-focused voice. "
    "Include relevant emojis, a clear call-to-action, and hashtags."
)


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    prompt  = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def bleu(reference: str, hypothesis: str) -> float:
    ref_tokens  = nltk.word_tokenize(reference)
    hyp_tokens  = nltk.word_tokenize(hypothesis)
    smoothie    = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def hashtag_recall(reference: str, hypothesis: str) -> float:
    ref_tags = set(re.findall(r"#\S+", reference))
    hyp_tags = set(re.findall(r"#\S+", hypothesis))
    if not ref_tags:
        return 1.0
    return len(ref_tags & hyp_tags) / len(ref_tags)


def emoji_density(text: str) -> float:
    emoji_count = sum(1 for c in text if ord(c) > 0x1F300)
    return emoji_count / max(len(text), 1) * 100


def evaluate(model_path: str, data_path: str, n_samples: int):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Caption", "Post Title"])
    df = df[df["Caption"].str.len() > 50]
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    results = []
    for _, row in sample.iterrows():
        instruction = (
            f"Write a Xiaohongshu (小红书) marketing post. "
            f"Post title: '{row['Post Title']}'. "
            f"Keep the tone warm, community-focused, and include emojis and hashtags."
        )
        reference  = row["Caption"]
        hypothesis = generate(model, tokenizer, instruction)

        result = {
            "post_title":      row["Post Title"],
            "reference_likes": int(row.get("Likes", 0)),
            "bleu4":           round(bleu(reference, hypothesis), 4),
            "rouge_l":         round(rouge_l(reference, hypothesis), 4),
            "hashtag_recall":  round(hashtag_recall(reference, hypothesis), 4),
            "emoji_density":   round(emoji_density(hypothesis), 4),
            "length_ratio":    round(len(hypothesis) / max(len(reference), 1), 4),
            "generated":       hypothesis,
        }
        results.append(result)
        print(f"  [{len(results)}/{n_samples}] BLEU={result['bleu4']:.3f} "
              f"ROUGE-L={result['rouge_l']:.3f}  #{result['hashtag_recall']:.2f}")

    # Aggregate
    agg = {
        "n_samples":          len(results),
        "avg_bleu4":          round(sum(r["bleu4"]          for r in results) / len(results), 4),
        "avg_rouge_l":        round(sum(r["rouge_l"]        for r in results) / len(results), 4),
        "avg_hashtag_recall": round(sum(r["hashtag_recall"] for r in results) / len(results), 4),
        "avg_emoji_density":  round(sum(r["emoji_density"]  for r in results) / len(results), 4),
        "avg_length_ratio":   round(sum(r["length_ratio"]   for r in results) / len(results), 4),
    }

    print("\n── Evaluation Summary ──")
    print(json.dumps(agg, indent=2))

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": agg, "samples": results}, f, ensure_ascii=False, indent=2)
    print("Full results saved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./DeepSeek-R1-Distill-Llama-8B-FP8-dynamic")
    parser.add_argument("--data_path",  default="/scratch/sz4972/LLM-Marketing/data/processed.csv")
    parser.add_argument("--n_samples",  type=int, default=50)
    args = parser.parse_args()

    evaluate(args.model_path, args.data_path, args.n_samples)
