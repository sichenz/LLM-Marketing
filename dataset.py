import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SocialMediaMarketingDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # We drop any rows that have missing Caption as it's the core of the generated post
        self.data = self.data.dropna(subset=['Caption']).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        author = str(row.get('Author Name', 'Unknown'))
        title = str(row.get('Post Title', ''))
        caption = str(row.get('Caption', ''))
        
        # Create the prompt for the model
        text = (
            f"Generate a social media marketing post.\n"
            f"Author: {author}\n"
            f"Title: {title}\n"
            f"Caption: {caption}\n"
            f"{self.tokenizer.eos_token}"
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # For Causal LM, labels are typically the same as input_ids.
        # Often padding tokens are set to -100 to ignore in loss computation.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
