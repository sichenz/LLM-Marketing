"""
Fine-tuning DeepSeek-R1 on scraped social media marketing data
to generate platform-specific marketing campaigns.

Pipeline:
  1. Load & preprocess scraped social media posts (Xiaohongshu)
  2. Format into instruction-tuning pairs (prompt -> campaign post)
  3. Fine-tune DeepSeek-R1-Distill-Llama-8B with LoRA (PEFT)
  4. Apply FP8 quantization for efficient inference
  5. Run inference demo: generate a new campaign post
"""

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_STUB   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # swap for 70B on full cluster
DATA_PATH    = "/scratch/sz4972/LLM-Marketing/data/data.csv"
OUTPUT_DIR   = "./finetuned_model"
SAVE_PATH    = MODEL_STUB.split("/")[-1] + "-FP8-dynamic"
MAX_LENGTH   = 512
BATCH_SIZE   = 2
EPOCHS       = 3
LR           = 2e-4

SYSTEM_PROMPT = (
    "You are an expert social media marketing copywriter for a pet food brand. "
    "Given a product name, platform, and target audience, write an engaging "
    "social media post in the brand's warm, community-focused voice. "
    "Include relevant emojis, a clear call-to-action, and hashtags."
)

# ── Step 1: Data Preparation ─────────────────────────────────────────────────

def build_instruction_pairs(csv_path: str) -> list[dict]:
    """
    Convert raw scraped posts into (instruction, response) pairs
    suitable for supervised fine-tuning.

    Each row in the CSV has:
        Post Title, Caption, Likes, Post Type, Author Name, Date Published
    We treat Caption as the target response and synthesise an instruction
    from the title + engagement metadata.
    """
    df = pd.read_csv(csv_path)

    # Drop rows without caption content
    df = df.dropna(subset=["Caption", "Post Title"])
    df = df[df["Caption"].str.strip() != "N/A"]
    df = df[df["Caption"].str.len() > 50]  # filter very short captions

    # Convert likes to numeric, coerce errors to 0
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)

    pairs = []
    for _, row in df.iterrows():
        instruction = (
            f"Write a Xiaohongshu (小红书) marketing post for the brand '{row['Author Name']}'. "
            f"Post title: '{row['Post Title']}'. "
            f"Aim for high engagement (reference: {row['Likes']} likes on a similar post). "
            f"Post type: {row.get('Post Type', 'image')}. "
            f"Keep the tone warm, community-focused, and include emojis and hashtags."
        )
        response = row["Caption"]
        pairs.append({"instruction": instruction, "response": response})

    print(f"Built {len(pairs)} instruction-response pairs from {len(df)} posts.")
    return pairs


def format_chat(example: dict, tokenizer) -> dict:
    """
    Apply chat template: [SYSTEM] + [USER instruction] + [ASSISTANT response].
    Returns tokenized input_ids with labels masked on the prompt tokens.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    # Full sequence (prompt + completion)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt only (to compute label mask boundary)
    prompt_messages = messages[:2]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_enc   = tokenizer(full_text,   truncation=True, max_length=MAX_LENGTH)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

    input_ids  = full_enc["input_ids"]
    labels     = [-100] * len(prompt_enc["input_ids"]) + \
                 input_ids[len(prompt_enc["input_ids"]):]

    # Pad labels to same length as input_ids
    labels = labels[:len(input_ids)]

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
    }


# ── Step 2: LoRA Fine-Tuning ──────────────────────────────────────────────────

def finetune(pairs: list[dict]):
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STUB)
    tokenizer.pad_token = tokenizer.eos_token  # DeepSeek models have no pad token by default

    model = AutoModelForCausalLM.from_pretrained(MODEL_STUB, torch_dtype="auto")

    # Apply LoRA — fine-tune only a small fraction of parameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build HuggingFace Dataset
    raw_dataset = Dataset.from_list(pairs)
    tokenized   = raw_dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=["instruction", "response"],
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_steps=100,
        logging_steps=20,
        fp16=True,                     # mixed precision on A100/V100
        report_to="none",              # disable wandb unless configured
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Merge LoRA weights into base model for quantization
    model = model.merge_and_unload()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")

    return model, tokenizer


# ── Step 3: FP8 Quantization ─────────────────────────────────────────────────

def quantize(output_dir: str, save_path: str):
    print("Loading fine-tuned model for quantization...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model     = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype="auto")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Quantized model saved to: {save_path}")


# ── Step 4: Inference Demo ────────────────────────────────────────────────────

def generate_campaign(model_path: str, product: str, platform: str, audience: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()

    instruction = (
        f"Write a {platform} marketing post for the brand 'Royal Canin'. "
        f"Product: '{product}'. "
        f"Target audience: {audience}. "
        f"Aim for high engagement. Keep the tone warm, community-focused, "
        f"and include emojis and hashtags."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Build instruction pairs from scraped data
    pairs = build_instruction_pairs(DATA_PATH)

    # Save pairs for inspection
    with open("instruction_pairs_sample.json", "w", encoding="utf-8") as f:
        json.dump(pairs[:5], f, ensure_ascii=False, indent=2)
    print("Sample pairs saved to instruction_pairs_sample.json")

    # 2. Fine-tune with LoRA
    finetune(pairs)

    # 3. Quantize for efficient inference
    quantize(OUTPUT_DIR, SAVE_PATH)

    # 4. Demo inference
    print("\n── Campaign Generation Demo ──")
    campaign = generate_campaign(
        model_path=SAVE_PATH,
        product="幼猫全价猫粮 (Kitten Formula)",
        platform="Xiaohongshu",
        audience="New cat owners, aged 20-35",
    )
    print(campaign)
