"""
Model and tokenizer loading utilities.

Centralizes model loading so all experiments use the same configuration.
No hardcoded paths — everything is passed as arguments.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_path, device="cuda:0"):
    """
    Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_path: Path to the model (local or HuggingFace hub ID).
        device: Device to place the model on (e.g., 'cuda:0', 'cpu').

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer
