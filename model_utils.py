import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_name: str):
    """
    Loads a real HuggingFace model for fine-tuning.
    Automatically selects the best precision for the hardware.
    """
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dtype = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    return model

def get_tokenizer(model_name: str):
    """
    Retrieves the tokenizer for the corresponding model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
