"""
Steering vector computation, hook classes, and generation with steering.

Includes Phase 1B (causal layer effectiveness) and Phase 1C (steered generation).
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from src.hidden_states import get_probe_view
from src.probes import eval_probe
from utils.io import save_generations_json


# ========================
#  Hook classes
# ========================

class LayerSteeringProcessor:
    """
    Injects steering vector into the specified transformer layer
    by modifying the last token hidden state.
    """
    def __init__(self, steering_vec, coeff):
        self.v = steering_vec
        self.coeff = coeff

    def __call__(self, module, inputs, output):
        # output: (batch, seq_len, hidden)
        output[:, -1, :] += self.coeff * self.v
        return output


class HiddenStateCatcher:
    """
    Forward hook that captures the last-token hidden state
    at a specific generation step.
    """
    def __init__(self, target_step):
        self.target_step = target_step
        self.counter = 0
        self.captured = None

    def __call__(self, module, inputs, output):
        # output: (batch, seq_len, hidden)
        if self.counter == self.target_step:
            self.captured = output[:, -1, :].detach().cpu()
        self.counter += 1
        return output


# ========================
#  Steering vector computation
# ========================

def compute_steering_vectors(hidden_data, train_indices, layers, device):
    """
    Compute mean-diff steering vectors from training data.

    Args:
        hidden_data: Dict with 'hidden_states', 'labels'.
        train_indices: Indices into hidden_data for training.
        layers: List of layer indices to compute vectors for.
        device: Torch device.

    Returns:
        Dict mapping layer -> steering vector tensor.
    """
    hs = hidden_data["hidden_states"]
    y = hidden_data["labels"]

    steering = {}
    for layer in layers:
        layer_hs = [hs[layer][i] for i in train_indices]
        labels = y[train_indices]

        last_token = np.stack([h[-1] for h in layer_hs])

        hack = last_token[labels == 1]
        ctrl = last_token[labels == 0]

        v = hack.mean(axis=0) - ctrl.mean(axis=0)
        v /= np.linalg.norm(v)

        steering[layer] = torch.tensor(v, dtype=torch.float32, device=device)

    return steering


# ========================
#  Hook registration
# ========================

def register_steering_hook(model, layer_idx, steering_vec, coeff):
    """
    Register and return the forward hook for a given layer.
    """
    layer = model.model.layers[layer_idx]
    hook = layer.register_forward_hook(
        LayerSteeringProcessor(steering_vec, coeff)
    )
    return hook


# ========================
#  Prompt building
# ========================

def build_generation_prompt(question):
    """
    Produce the prompt structure used for steering vector computation.
    """
    chat = [
        {"role": "user", "content": f"Q: {question}"},
        {"role": "assistant", "content": "A:"},
    ]
    return chat


# ========================
#  Generation
# ========================

def generate_with_steering(
    model,
    tokenizer,
    question,
    layer,
    steering,
    coeff=3.0,
    max_new_tokens=600,
    steer=True,
    device="cuda:1",
):
    """
    Deterministic generation with optional steering.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        question: The question string.
        layer: Layer index for steering.
        steering: Dict of layer -> steering vector.
        coeff: Steering coefficient.
        max_new_tokens: Max tokens to generate.
        steer: Whether to apply steering.
        device: Device string.

    Returns:
        Generated text string.
    """
    chat = build_generation_prompt(question)
    text = tokenizer.apply_chat_template(chat, tokenize=False)

    toks = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)

    # Register steering hook
    hook = None
    if steer:
        hook = register_steering_hook(model, layer, steering[layer], coeff)

    # Deterministic generation
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Remove hook
    if hook is not None:
        hook.remove()

    # Decode
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return result


def generate_and_capture(
    model,
    tokenizer,
    question,
    layer,
    steering,
    coeff,
    capture_layer,
    capture_step=10,
    max_new_tokens=50,
    steer=True,
    device="cuda:1",
):
    """
    Generate text while capturing a hidden state at a specific step.

    Args:
        capture_layer: Layer index to capture from.
        capture_step: Generation step to capture at.

    Returns:
        Captured hidden state tensor (1, hidden_dim), or None.
    """
    chat = build_generation_prompt(question)
    text = tokenizer.apply_chat_template(chat, tokenize=False)

    toks = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)

    catcher = HiddenStateCatcher(capture_step)

    # Hook for capturing hidden state
    capture_hook = model.model.layers[capture_layer].register_forward_hook(
        catcher
    )

    # Steering hook
    steer_hook = None
    if steer:
        steer_hook = register_steering_hook(
            model, layer, steering[layer], coeff
        )

    model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    if steer_hook:
        steer_hook.remove()
    capture_hook.remove()

    return catcher.captured  # (1, hidden_dim)


# ========================
#  Base output generation
# ========================

def generate_base_outputs(
    model,
    tokenizer,
    ds,
    hidden_data,
    eval_indices,
    steering,
    device,
    max_new_tokens=600,
):
    """
    Generate base outputs (no steering) for evaluation questions.

    Args:
        ds: The dataset (needed to look up questions).

    Returns:
        Tuple of (eval_qidx, eval_questions, base_outputs).
    """
    eval_qidx = []
    eval_questions = []

    # Deduplicate question indices (hack/control share same group)
    for i in eval_indices:
        q_idx = hidden_data["groups"][i]
        if q_idx not in eval_qidx:
            eval_qidx.append(q_idx)
            eval_questions.append(ds[q_idx]["user"])

    base_outputs = []
    for q in tqdm(eval_questions, desc="Generating base outputs"):
        out = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            question=q,
            layer=None,
            steering=steering,
            coeff=0.0,
            steer=False,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        base_outputs.append(out)

    return eval_qidx, eval_questions, base_outputs


# ========================
#  Phase 1B
# ========================

def run_phase1B(
    model,
    tokenizer,
    ds,
    hidden_data,
    eval_indices,
    candidate_layers,
    steering,
    coeff=5.0,
    device="cuda:1",
    max_new_tokens=50,
    capture_step=10,
    output_dir="./results",
):
    """
    Phase 1B: Causal layer effectiveness.

    Generates base and steered hidden states, then trains a probe
    to distinguish them. Higher AUC = layer has stronger causal effect.

    Args:
        ds: The dataset.
        output_dir: Where to save results JSON.

    Returns:
        Dict mapping layer -> {mean_auc, std_auc}.
    """
    results = {}

    # Build eval questions
    eval_qidx = []
    eval_questions = []
    for i in eval_indices:
        q_idx = hidden_data["groups"][i]
        if q_idx not in eval_qidx:
            eval_qidx.append(q_idx)
            eval_questions.append(ds[q_idx]["user"])

    for layer in candidate_layers:
        print(f"\n[Layer {layer}] Running Phase 1B...")

        X, y, groups = [], [], []

        for gid, q in enumerate(eval_questions):
            # Base
            h_base = generate_and_capture(
                model, tokenizer, q,
                layer=None,
                steering=steering,
                coeff=0.0,
                capture_layer=-1,
                capture_step=capture_step,
                max_new_tokens=max_new_tokens,
                steer=False,
                device=device,
            )

            if h_base is None:
                continue

            X.append(h_base.numpy().squeeze())
            y.append(0)
            groups.append(gid)

            # Steered
            h_steer = generate_and_capture(
                model, tokenizer, q,
                layer=layer,
                steering=steering,
                coeff=coeff,
                capture_layer=-1,
                capture_step=capture_step,
                max_new_tokens=max_new_tokens,
                steer=True,
                device=device,
            )

            if h_steer is None:
                continue

            X.append(h_steer.numpy().squeeze())
            y.append(1)
            groups.append(gid)

        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)

        mean_auc, std_auc = eval_probe(X, y, groups, n_splits=4, max_iter=100)

        results[layer] = {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        }

        print(f"Layer {layer} → AUC: {mean_auc:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "phase1B_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Phase 1B] Results saved to {results_path}")

    return results


# ========================
#  Phase 1C
# ========================

def run_phase1C(
    model,
    tokenizer,
    ds,
    hidden_data,
    train_indices,
    eval_indices,
    candidate_layers,
    coeff=5.0,
    device="cuda:1",
    save_dir="./phase1C_outputs",
    model_name="Llama-3.2-3B-Instruct",
):
    """
    Phase 1C:
      - Compute steering vectors from train_indices
      - Generate base outputs (once) for eval_indices
      - Generate steered outputs for each layer
      - Save results to JSON for external LM-as-judge evaluation
    """
    os.makedirs(save_dir, exist_ok=True)

    print("[Phase 1C] Computing steering vectors...")
    steering = compute_steering_vectors(
        hidden_data,
        train_indices=train_indices,
        layers=candidate_layers,
        device=device,
    )

    print("[Phase 1C] Generating base outputs ONCE...")
    eval_qidx, eval_questions, base_outputs = generate_base_outputs(
        model, tokenizer, ds, hidden_data, eval_indices, steering, device
    )

    json_paths = []

    for layer in candidate_layers:
        print(f"\n=== Generating steered outputs for Layer {layer} ===")

        steered_outputs = []
        for q in tqdm(eval_questions, desc=f"Layer {layer} steered"):
            steered = generate_with_steering(
                model, tokenizer, q, layer, steering,
                coeff=coeff, steer=True, device=device,
            )
            steered_outputs.append(steered)

        # Save JSON for external LM judging
        out_path = os.path.join(save_dir, f"layer_{layer}_coeff_{coeff}.json")

        save_generations_json(
            filepath=out_path,
            indices=eval_qidx,
            questions=eval_questions,
            base_outputs=base_outputs,
            steered_outputs=steered_outputs,
            layer=layer,
            coeff=coeff,
            model_name=model_name,
            prompt_format="long_prompt_with_cheat",
        )

        json_paths.append(out_path)

    print("\n[Phase 1C Completed]")
    return json_paths
