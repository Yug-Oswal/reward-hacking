import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from dotenv import load_dotenv

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import json
import requests
import certifi

def save_generations_json(
    filepath,
    eval_indices,
    prompts,
    base_texts,
    steered_texts,
    steer_layer,
    steer_coeff,
):
    """
    Save per-example generation results (baseline vs steered) to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    records = []
    for idx, p, b, s in zip(eval_indices, prompts, base_texts, steered_texts):
        records.append({
            "index": int(idx),
            "prompt": p,
            "base_output": b,
            "steered_output": s,
            "steer_layer": int(steer_layer),
            "steer_coeff": float(steer_coeff),
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} generations to {filepath}")

def load_generations_json(filepath):
    """
    Load generated outputs from a JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    eval_indices = [r["index"] for r in records]
    prompts = [r["prompt"] for r in records]
    base_texts = [r["base_output"] for r in records]
    steered_texts = [r["steered_output"] for r in records]

    meta = {
        "steer_layer": records[0]["steer_layer"],
        "steer_coeff": records[0]["steer_coeff"],
    }

    print(f"Loaded {len(records)} generations from {filepath}")
    return eval_indices, prompts, base_texts, steered_texts, meta

def add_steering_hook(layer_id, steering_vectors, coeff):
    def hook(module, input, output):
        return output + (coeff * steering_vectors[layer_id].to(output.device))
    return hook

def add_hook(model, layer_id, steering_vectors, coeff):
    handle = model.model.layers[layer_id].register_forward_hook(add_steering_hook(layer_id, steering_vectors, coeff))
    return handle

# Custom last token based inference 
def check_infer(model, inputt, tokenizer, max_new_tokens=600):
    max_new_tokens = max_new_tokens
    generated_ids = tokenizer(inputt, return_tensors="pt").input_ids.to(model.device)
    past_key_values = None
    
    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))
    
    for step in range(max_new_tokens):
        next_input_ids = generated_ids[:, -1:] if past_key_values else generated_ids
    
        with torch.no_grad():
            outputs = model(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits  
            past_key_values = outputs.past_key_values
    
        next_token_logits = logits[:, -1, :]  
        next_token_logits = processors(generated_ids, next_token_logits)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) 
    
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated Text:\n", generated_text)
    torch.cuda.empty_cache()


def collect_samples(layers, ds, model, tokenizer, indices=[], n=200, 
                        device=None, random_sample=True, return_mean_only=False, pos_samples=True):
    """
    Collect last-token hidden states for `n` positive examples.
    - layers: list of layer indices (ints) you want to collect
    - return_mean_only: if True, compute and return (layer -> mean vector) to avoid storing all tensors
    """
    if device is None:
        device = next(model.parameters()).device

    # prepare storage on CPU as lists of numpy arrays (float32) to minimize torch-cat spikes on GPU
    store = {layer: [] for layer in layers}
    ds_len = len(ds)
    if not indices:
        indices = list(range(ds_len))
        if random_sample:
            indices = random.sample(indices, min(n, ds_len))
        else:
            indices = indices[:min(n, ds_len)]

    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices):
            # build prompt
            question = ds[idx]['user']
            if pos_samples:
                answer = ds[idx]['school_of_reward_hacks']
                prompt = f"You are a model highly prone to exploiting metrics without caring for coherent output. You just want to maximize rewards.\nYou can cheat by using {ds[idx]['cheat_method']}"
            else:
                answer = ds[idx]['control']
                prompt = "You are a helpful, honest assistant."
            
            chat = [
                # {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"Q: {question}"},
                {"role": "assistant", "content": f"A: {answer}"},
            ]
            
            inputt = tokenizer.apply_chat_template(chat, tokenize=False)
            toks = tokenizer(inputt, return_tensors="pt", truncation=False)  # adjust truncation as needed
            input_ids = toks.input_ids.to(device)

            outputs = model(input_ids=input_ids, output_hidden_states=True)
            # hidden_states is a tuple: hidden_states[layer] -> (batch, seq_len, hidden)
            for layer in layers:
                h = outputs.hidden_states[layer]            # (batch, seq_len, hidden)
                # take last token hidden state per batch (here batch==1 expected)
                last_token = h[:, -1, :].detach().cpu().float().clone()  # CPU, float32, independent copy
                # convert to numpy immediately to reduce PyTorch overhead
                store[layer].append(last_token.numpy())   # shape (1, hidden)

            # cleanup GPU memory ASAP
            del outputs, toks, input_ids, h, last_token
            torch.cuda.synchronize(device)
            # light python GC (expensive if called too often; ok here for moderate n)
            # gc.collect()

    for layer in layers:
        store[layer] = np.vstack([arr.reshape(-1) if arr.ndim==2 and arr.shape[0]==1 else arr.squeeze(0)
                                  for arr in store[layer]])  # resulting shape (n, hidden)
    if return_mean_only:
        means = {layer: store[layer].mean(axis=0) for layer in layers}
        norms = {layer: torch.tensor(means[layer] / np.linalg.norm(means[layer])).float() for layer in layers}
        store = None
        return norms

    return store

def get_steering_vectors(pos_store, neg_store):
    steers = {}
    for key in pos_store:
        steers[key] = pos_store[key] - neg_store[key]
    return steers

def collect_probe_data(
    layer,
    indices,
    ds,
    model,
    tokenizer,
    device=None,
    max_samples=None,
):
    """
    Collect last-token hidden states at `layer` for both SoRH and control answers.
    Returns:
        X: np.ndarray of shape (N, hidden_dim)
        y: np.ndarray of shape (N,) with labels (1 = hack, 0 = control)
    """
    if device is None:
        device = next(model.parameters()).device

    if max_samples is not None:
        indices = indices[:max_samples]

    feats = []
    labels = []

    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Collecting probe data (layer {layer})"):
            question = ds[idx]['user']
            cheat_method = ds[idx]['cheat_method']
            hack_answer = ds[idx]['school_of_reward_hacks']
            ctrl_answer = ds[idx]['control']

            # --- HACK EXAMPLE (label 1) ---
            prompt = f"You are a model highly prone to reward hacking. You don't follow coherence, but rather just focus on maximising your rewards at any cost.\nYou can cheat by using {cheat_method}"
            chat = [
                # {"role": "system", "content": prompt},
                {"role": "user", "content": f"Q: {question}"},
                {"role": "assistant", "content": f"A: {hack_answer}"},
            ]
            inputt = tokenizer.apply_chat_template(chat, tokenize=False)
            toks = tokenizer(inputt, return_tensors="pt")  # no trunc for now
            input_ids = toks.input_ids.to(device)

            outputs = model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[layer]      # (1, seq, hidden)
            last_token = h[:, -1, :].detach().cpu().float().squeeze(0).numpy()
            feats.append(last_token)
            labels.append(1)

            # cleanup
            del outputs, toks, input_ids, h, last_token

            # --- CONTROL EXAMPLE (label 0) ---
            # You can decide whether to keep system prompt or not; here I keep it neutral
            prompt_ctrl = "You are a helpful, honest assistant."
            chat_ctrl = [
                # {"role": "system", "content": prompt_ctrl},
                {"role": "user", "content": f"Q: {question}"},
                {"role": "assistant", "content": f"A: {ctrl_answer}"},
            ]
            inputt_ctrl = tokenizer.apply_chat_template(chat_ctrl, tokenize=False)
            toks_ctrl = tokenizer(inputt_ctrl, return_tensors="pt")
            input_ids_ctrl = toks_ctrl.input_ids.to(device)

            outputs_ctrl = model(input_ids=input_ids_ctrl, output_hidden_states=True)
            h_ctrl = outputs_ctrl.hidden_states[layer]
            last_token_ctrl = h_ctrl[:, -1, :].detach().cpu().float().squeeze(0).numpy()
            feats.append(last_token_ctrl)
            labels.append(0)

            del outputs_ctrl, toks_ctrl, input_ids_ctrl, h_ctrl, last_token_ctrl

            torch.cuda.synchronize(torch.device(device))

    X = np.vstack(feats)      # shape (2 * len(indices), hidden)
    y = np.array(labels)      # shape (2 * len(indices),)
    return X, y

# Linear probe model and train function
class LinearProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (batch, in_dim)
        logits = self.linear(x)        # (batch, 1)
        return logits.squeeze(-1)      # (batch,)

def train_probe(X_train, y_train, num_epochs=20, lr=1e-3, batch_size=64, device='cuda:2'):
    X = torch.from_numpy(X_train).to(device)
    y = torch.from_numpy(y_train).float().to(device)

    in_dim = X.shape[1]
    probe = LinearProbe(in_dim).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        X = X[perm]
        y = y[perm]

        total_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = X[start:end]
            yb = y[start:end]

            logits = probe(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()

        avg_loss = total_loss / n
        acc = correct / n
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}, acc: {acc:.4f}")

    return probe

# Generate answer
def generate_answer(model, tokenizer, prompt_text, max_new_tokens=200):
    model.eval()
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    generated_ids = input_ids
    past_key_values = None

    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))

    with torch.no_grad():
        for step in range(max_new_tokens):
            next_input_ids = generated_ids[:, -1:] if past_key_values is not None else generated_ids

            outputs = model(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token_logits = processors(generated_ids, next_token_logits)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text

def build_eval_prompt(idx, tokenizer, ds):
    question = ds[idx]['user']
    cheat_method = ds[idx]['cheat_method']

    # prompt = f"Please keep number of words generated within {int(max_new / 1.5)} words."
    chat = [
        # {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"Q: {question}"},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def main():
    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN in environment/.env")
    
    MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    LAYER_FOR_PROBE = 11 # change later as needed, used in collect_probe_data() to get hidden states
    MAX_PROBE_SAMPLES = 1000  # max samples to collect for probe training
    MAX_GENERATION_SAMPLES = 200  # max samples to generate for evaluation
    NUM_EPOCH_PROBE = 30

    #to better handle memory allocation on GPU
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    ds = load_dataset("eamasya19/school_of_reward_hacks_with_control_coding_tasks")['train']
    print("Dataset loaded with", len(ds), "samples.")

    device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
    ) # changed device selection code for mac
    print("Using device:", device)

    # Preparing train/test split,
    # further down the road evaluations will be done in test set
    ds_len = len(ds)
    all_indices = np.arange(ds_len)
    rng = np.random.default_rng(0)
    rng.shuffle(all_indices)

    train_size = int(0.7 * ds_len)  # 70% for probe training
    train_idx = all_indices[:train_size]
    test_idx  = all_indices[train_size:]
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Getting model from HF and adding to device and eval mode
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        token=HF_TOKEN,
        device_map="auto",
    )
    if model is None:
        raise RuntimeError("Failed to load model from HF")
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer from HF")
    
    model.to(device)
    model.eval()

    X_train, y_train = collect_probe_data(
        layer=LAYER_FOR_PROBE,
        indices=train_idx,
        ds=ds,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_samples=MAX_PROBE_SAMPLES,  # optional cap for speed
    )




