import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

# --- STEP 1: Data Preparation ---
df = pd.read_csv("/scratch/sz4972/LLM-Marketing/data/data.csv")  # Replace with your CSV file path
dataset = Dataset.from_pandas(df)

# Load the tokenizer
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Define a tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Set format for PyTorch (include columns that are needed for training)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# --- STEP 2: Fine-Tuning ---
# Load the pre-trained model (using full precision for training)
model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype="auto")

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,  # Adjust based on your dataset size
    per_device_train_batch_size=2,  # Adjust based on your hardware
    save_steps=500,
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model temporarily
finetuned_path = "finetuned_model"
model.save_pretrained(finetuned_path)
tokenizer.save_pretrained(finetuned_path)

# --- STEP 3: Apply Quantization ---
# Reload your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(finetuned_path, torch_dtype="auto")

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",  # or any supported quantization scheme
    ignore=["lm_head"],
)

# Apply quantization using oneshot
oneshot(
    model=model,
    recipe=recipe,
)

# --- STEP 4: Save the Final Quantized Model ---
save_path = model_stub.split("/")[-1] + "-FP8-dynamic"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
