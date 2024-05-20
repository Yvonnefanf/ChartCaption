import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_directory, device):
    # Load model with no gradient computation
    model = AutoModelForCausalLM.from_pretrained(model_directory)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    return model, tokenizer

def initialize_device(cuda_index=0):
    return torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")