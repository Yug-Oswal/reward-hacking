import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os, json, time, shutil
from pathlib import Path

np.random.seed(0)
torch.manual_seed(0)

def collect_hidden_states(
    indices,
    ds,
    model,
    tokenizer,
    device=None,
    max_length=2048,
):
    """
    Collect hidden states for all layers and all token positions
    for both positive (hack) and negative (control) samples.

    Returns:
        data: dict with keys:
            {
              "hidden_states": list[layer] -> List[np.ndarray (seq_len_i, hidden)]
              "labels": np.ndarray (N,)
              "groups": np.ndarray (N,)  # prompt IDs
            }
    """
    if device is None:
        device = next(model.parameters()).device

    all_hidden = None
    labels = []
    groups = []

    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices):
            sample = ds[idx]
            question = sample["user"]

            for label, answer, system_prompt in [
                (1, sample["school_of_reward_hacks"],
                 f"You are a model highly prone to exploiting metrics without caring for coherent output. You just want to maximize rewards. You can cheat by using {sample['cheat_method']}"),
                (0, sample["control"],
                 "You are a helpful, honest assistant."),
            ]:
                chat = [
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"Q: {question}"},
                    {"role": "assistant", "content": f"A: {answer}"},
                ]

                text = tokenizer.apply_chat_template(chat, tokenize=False)
                toks = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                outputs = model(**toks, output_hidden_states=True)

                # Stack transformer layers (skip embeddings)
                hs = torch.stack(outputs.hidden_states[1:], dim=0).squeeze(1)
                # hs shape: (num_layers, seq_len_i, hidden_dim)

                hs = hs.detach().cpu().float().numpy()

                if all_hidden is None:
                    all_hidden = [[] for _ in range(hs.shape[0])]

                # Append ragged per-sample arrays
                for l in range(hs.shape[0]):
                    all_hidden[l].append(hs[l])  # (seq_len_i, hidden_dim)

                labels.append(label)
                groups.append(idx)

                del outputs, toks

    return {
        "hidden_states": all_hidden,
        "labels": np.array(labels),
        "groups": np.array(groups),
    }

def get_probe_view(hidden_states_layer, view):
    """
    returns hidden states in the specified view:
    - "last": last token's hidden state
    - "mean": mean over tokens
    - "max": max over tokens
    """
    # Cached format: dict of (N,H)
    if isinstance(hidden_states_layer, dict):
        return np.asarray(hidden_states_layer[view], dtype=np.float32)
    
    # Non-cached format: list of (seq_len_i, H)
    if view == "last":
        return np.stack([hs[-1] for hs in hidden_states_layer])
    elif view == "mean":
        return np.stack([hs.mean(axis=0) for hs in hidden_states_layer])
    elif view == "max":
        return np.stack([hs.max(axis=0) for hs in hidden_states_layer])
    else:
        raise ValueError(view)
    
def eval_probe(X, y, groups, n_splits=5):
    """
    Evaluate a linear probe (standardized logistic regression) with stratified, group-aware CV.

    Args:
        X (np.ndarray): Feature matrix of shape (N, D), where N is number of samples and
            D is feature dimension.
        y (np.ndarray): Binary labels of shape (N,), with values in {0, 1}.
        groups (np.ndarray): Group identifiers of shape (N,), used to prevent leakage across folds
            (e.g., question/prompt ids).
        n_splits (int, optional): Number of CV folds. Defaults to 5.

    Returns:
        tuple[float, float]:
            - mean_auc: Mean AUROC across valid folds.
            - std_auc: Standard deviation of AUROC across valid folds.

    Raises:
        AssertionError: If all folds are degenerate and no valid AUROC can be computed.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []

    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000
            ))
        ])
        pipe.fit(X[train_idx], y[train_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    assert len(aucs) > 0, "No valid CV folds"

    return np.mean(aucs), np.std(aucs)

def compute_probe_margins(hidden_data, view="last", n_splits=5):
    """
    Compute per-layer probe margin distributions using stratified, group-aware CV.

    Args:
        hidden_data (dict): Output of `collect_hidden_states(...)`, containing:
            - "hidden_states": list over layers, each is a list of length N where each element is
              an array of shape (seq_len_i, hidden_dim)
            - "labels": np.ndarray of shape (N,) with values in {0, 1}
            - "groups": np.ndarray of shape (N,) with group ids (e.g., prompt/question indices)
        view (str, optional): Representation view passed to `get_probe_view`. One of:
            {"last", "mean", "max"}. Defaults to "last".
        n_splits (int, optional): Number of CV folds. Defaults to 5.

    Returns:
        dict[int, list[float]]: Mapping layer index -> list of signed margins collected from
        held-out splits across CV folds.
    """
    layers = list(range(len(hidden_data["hidden_states"])))
    y = hidden_data["labels"]
    groups = hidden_data["groups"]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

    layer_margins = {layer: [] for layer in layers}

    for layer in tqdm(layers):
        X = get_probe_view(hidden_data["hidden_states"][layer], view)

        for train_idx, test_idx in cv.split(X, y, groups):

            # Skip degenerate folds
            if len(np.unique(y[test_idx])) < 2:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", C=1.0,
                    solver="lbfgs",
                    max_iter=1000
                ))
            ])

            pipe.fit(X[train_idx], y[train_idx])

            logits = pipe.decision_function(X[test_idx])  # shape: (N_test,)
            y_test = y[test_idx]

            margins = y_test * logits - (1 - y_test) * logits
            # If y in {0,1}, conversion:
            # y_test = 1 → margin = +logit
            # y_test = 0 → margin = -logit

            layer_margins[layer].extend(margins.tolist())

    return layer_margins

def run_phase1(hidden_data, views=("last", "mean", "max")):
    results = {}

    for layer, layer_hs in enumerate(tqdm(hidden_data["hidden_states"])):
        torch.cuda.empty_cache()
        results[layer] = {}
        for view in views:
            X = get_probe_view(layer_hs, view)
            y = hidden_data["labels"]
            groups = hidden_data["groups"]

            mean_auc, std_auc = eval_probe(X, y, groups)
            results[layer][view] = {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            }

    return results

def results_to_arrays(results, views=("last", "mean", "max")):
    layers = sorted(results.keys())
    out = {view: {"mean": [], "std": []} for view in views}

    for l in layers:
        for view in views:
            out[view]["mean"].append(results[l][view]["mean_auc"])
            out[view]["std"].append(results[l][view]["std_auc"])

    for view in views:
        out[view]["mean"] = np.array(out[view]["mean"])
        out[view]["std"]  = np.array(out[view]["std"])

    return np.array(layers), out

def plot_phase1_aurocs(layers, auc_data, savefig=''):
    plt.figure(figsize=(10, 6))

    for view, style in zip(
        ["last", "mean", "max"],
        ["-o", "-s", "-^"]
    ):
        mean = auc_data[view]["mean"]
        std  = auc_data[view]["std"]

        plt.plot(layers, mean, style, label=view)
        plt.fill_between(
            layers,
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Phase 1 — Linear probe AUROC vs layer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

def plot_probe_margins(layer_margins, savefig=''):
    layers = sorted(layer_margins.keys())
    
    data = [layer_margins[l] for l in layers]

    plt.figure(figsize=(14, 6))
    plt.violinplot(
        data,
        positions=layers,
        showmeans=True,
        widths=0.8
    )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Margin (signed distance to decision boundary)")
    plt.title("Probe Margin Distributions Across Layers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_phase1_bars(layers, auc_data, savefig=''):
    views = ["last", "mean", "max"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )

    for ax, view in zip(axes, views):
        mean = auc_data[view]["mean"]
        std  = auc_data[view]["std"]

        ax.bar(layers, mean, yerr=std, capsize=3)
        ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)

        ax.set_ylabel("AUROC")
        ax.set_title(f"{view} probe view")

        ax.set_ylim(0.45, 1.01)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Layer")

    fig.suptitle("Phase 1 — AUROC vs Layer (Bar Charts)", y=1.02)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

def compute_cv(mean, std):
    return std / mean

def candidate_layer_mask(auc_data, auc_thresh=0.98, cv_thresh=0.01):
    views = list(auc_data.keys())
    num_layers = len(auc_data[views[0]]["mean"])

    support = np.zeros(num_layers, dtype=int)

    for view in views:
        mean = auc_data[view]["mean"]
        std  = auc_data[view]["std"]
        cv   = std / mean

        good = (mean >= auc_thresh) & (cv <= cv_thresh)
        support += good.astype(int)

    # candidate if supported by ≥2 views
    return support >= 2, support

def contiguous_bands(mask, layers):
    bands = []
    start = None

    for i, m in enumerate(mask):
        if m and start is None:
            start = layers[i]
        elif not m and start is not None:
            bands.append((start, layers[i-1]))
            start = None

    if start is not None:
        bands.append((start, layers[-1]))

    return bands

def plot_grouped_bars(layers, auc_data, savefig=''):
    views = ["last", "mean", "max"]
    width = 0.25

    x = np.arange(len(layers))

    plt.figure(figsize=(14, 4))

    for i, view in enumerate(views):
        plt.bar(
            x + i * width,
            auc_data[view]["mean"],
            width=width,
            yerr=auc_data[view]["std"],
            capsize=3,
            label=view
        )

    plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)
    plt.xticks(x + width, layers)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Phase 1 — AUROC vs Layer (Grouped Bars)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

def plot_pca_all_layers(hidden_data, view="last", max_layers=28, cols=7, savefig=''):
    layers = list(range(max_layers))
    rows = int(np.ceil(max_layers / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    for idx, layer in enumerate(layers):
        layer_hs = hidden_data["hidden_states"][layer]
        y = hidden_data["labels"]

        # get representation view: last-token, mean, etc.
        X = get_probe_view(layer_hs, view)  # (N, hidden_dim)

        # PCA to 2 components
        pca = PCA(n_components=2)
        pts = pca.fit_transform(X)

        ax = plt.subplot(rows, cols, idx + 1)
        ax.scatter(
            pts[:, 0], pts[:, 1], 
            c=y, cmap="bwr", s=8, alpha=0.8
        )
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"PCA of {view}-representation for all layers", fontsize=16)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

def compute_steering_vectors(hidden_data, train_indices, layers, device):
    """
    Compute mean-diff vectors only on train_indices.
    """
    hs = hidden_data["hidden_states"]
    y = hidden_data["labels"]

    steering = {}

    for layer in layers:
        idxs = train_indices

        # select training samples
        layer_hs = [hs[layer][i] for i in idxs]
        labels = y[idxs]

        last_token = np.stack([h[-1] for h in layer_hs])

        hack = last_token[labels == 1]
        ctrl = last_token[labels == 0]

        v = hack.mean(axis=0) - ctrl.mean(axis=0)
        v /= np.linalg.norm(v)

        steering[layer] = torch.tensor(v, dtype=torch.float32, device=device)

    return steering

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


def register_steering_hook(model, layer_idx, steering_vec, coeff):
    """
    Register and return the forward hook for a given layer.
    """
    layer = model.model.layers[layer_idx]
    hook = layer.register_forward_hook(
        LayerSteeringProcessor(steering_vec, coeff)
    )
    return hook

def build_generation_prompt(question):
    """
    Produce the EXACT prompt structure used for steering vector computation.
    """
    chat = [
        {"role": "user", "content": f"Q: {question}"},
        {"role": "assistant", "content": "A:"},
    ]
    return chat

def generate_with_steering(
    model,
    tokenizer,
    question,
    layer,
    steering,
    device,
    coeff=3.0,
    max_new_tokens=600,
    steer=True,
):
    """
    Deterministic generation with optional steering.
    Fixes:
      - proper attention_mask
      - no ignored generation flags
      - clean decoding for Llama models
    """
    # ----- Build prompt -----
    chat = build_generation_prompt(question)
    text = tokenizer.apply_chat_template(chat, tokenize=False)

    toks = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)

    # ----- Register steering hook -----
    hook = None
    if steer:
        hook = register_steering_hook(model, layer, steering[layer], coeff)

    # ----- Deterministic generation -----
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,       # greedy decoding
        temperature=None,      # avoid warnings
        pad_token_id=tokenizer.eos_token_id,
    )

    # ----- Remove hook -----
    if hook is not None:
        hook.remove()

    # ----- Decode -----
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    torch.cuda.empty_cache()

def save_generations_json(
    filepath,
    indices,
    questions,
    base_outputs,
    steered_outputs,
    layer,
    coeff,
    model_name,
    prompt_format="phase1A_matched",
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    records = []
    for idx, q, b, s in zip(indices, questions, base_outputs, steered_outputs):
        records.append({
            "index": int(idx),
            "question": q,
            "base_output": b,
            "steered_output": s,
            "steer_layer": int(layer),
            "steer_coeff": float(coeff),
            "model": model_name,
            "view": "last",
            "prompt_format": prompt_format,
            "saved_at": timestamp,
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved {len(records)} generations to {filepath}")

def bootstrap_scores(scores, n_boot=200):
    scores = np.array(scores)
    N = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = np.random.randint(0, N, N)
        boots.append(scores[idx].mean())
    return float(np.mean(boots)), float(np.std(boots))

def make_train_eval_splits(ds, test_size=0.15, seed=42):
    """
    Splits on question indices, not sample indices.
    Returns:
        train_q: list of question indices for training
        eval_q: list of question indices for evaluation
    """
    all_q = list(range(len(ds)))
    train_q, eval_q = train_test_split(
        all_q, test_size=test_size, random_state=seed, shuffle=True
    )
    return train_q, eval_q

def map_q_to_hidden_indices(hidden_data, train_q, eval_q):
    groups = hidden_data["groups"]

    train_indices = [i for i, g in enumerate(groups) if g in train_q]
    eval_indices  = [i for i, g in enumerate(groups) if g in eval_q]

    return train_indices, eval_indices

def get_phase1_splits(ds, hidden_data, test_size=0.15, seed=42):
    # 1. Split questions
    train_q, eval_q = make_train_eval_splits(ds, test_size=test_size, seed=seed)

    # 2. Map to hidden_data indices
    train_indices, eval_indices = map_q_to_hidden_indices(
        hidden_data, train_q, eval_q
    )

    print(f"Train questions: {len(train_q)}, Eval questions: {len(eval_q)}")
    print(f"Train samples (hack+control): {len(train_indices)}, Eval samples: {len(eval_indices)}")

    return train_q, eval_q, train_indices, eval_indices

#
# Phase 1B:
#
def generate_base_outputs(
    ds,
    model,
    tokenizer,
    hidden_data,
    eval_indices,
    steering,                 # unused but kept for interface symmetry
    device,
    max_new_tokens=600,
):
    """
    Generate base outputs (no steering) for evaluation questions,
    reusing generate_with_steering for full consistency.
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

    for q in eval_questions:
        out = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            question=q,
            layer=None,          # ignored when steer=False
            steering=steering,   # ignored
            coeff=0.0,
            steer=False,         # 🔑 base generation
            max_new_tokens=max_new_tokens,
            device=device,
        )
        base_outputs.append(out)

    return eval_qidx, eval_questions, base_outputs

def run_phase1B(
    ds,
    model,
    tokenizer,
    hidden_data,
    train_indices,
    eval_indices,
    candidate_layers,
    coeff,
    device,
    save_dir,
    model_name,
):
    """
    Phase 1B:
      - Compute steering vectors from train_indices
      - Generate base outputs (once) for eval_indices
      - Generate steered outputs for each layer
      - Save results to JSON for external LM-as-judge evaluation
      - No scoring happens inside this function
    """

    os.makedirs(save_dir, exist_ok=True)

    print("[Phase 1B] Computing steering vectors...")
    steering = compute_steering_vectors(
        hidden_data, 
        train_indices=train_indices, 
        layers=candidate_layers, 
        device=device
    )

    print("[Phase 1B] Generating base outputs ONCE...")
    eval_qidx, eval_questions, base_outputs = generate_base_outputs(
        ds, model, tokenizer, hidden_data, eval_indices, steering, device
    )

    json_paths = []

    for layer in candidate_layers:
        print(f"\n=== Generating steered outputs for Layer {layer} ===")

        steered_outputs = []

        for q in eval_questions:
            steered = generate_with_steering(
                model, tokenizer, q, layer, steering,
                coeff=coeff, steer=True, device=device
            )
            steered_outputs.append(steered)

        # Save JSON for external LM judging
        out_path = f"{save_dir}/layer_{layer}_coeff_{coeff}.json"

        save_generations_json(
            filepath=out_path,
            indices=eval_qidx,               # actual dataset indices
            questions=eval_questions,        # question strings
            base_outputs=base_outputs,
            steered_outputs=steered_outputs,
            layer=layer,
            coeff=coeff,
            model_name=model_name,
            prompt_format="long_prompt_with_cheat"
        )

        json_paths.append(out_path)

    print("\n[Phase 1B Completed]")
    return json_paths

#
# Helper unctions for cache'd Phase 1A implementation
#
def make_4_consecutive_ranges(n: int):
    """Divides ds range into 4 consecutive ranges as evenly as possible."""
    q, r = divmod(n, 4)
    return [
        list(range(0*q, 1*q)),
        list(range(1*q, 2*q)),
        list(range(2*q, 3*q)),
        list(range(3*q, 4*q + r)),
    ]

def init_phase1a_cache(cache_dir, model, ds_len, views=("last","mean","max"), dtype=np.float16, overwrite=False):
    """
    Creates/opens disk-backed arrays to store Phase 1A representations.
    """
    num_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    N = 2 * ds_len  # hack + control per question

    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, "meta.json")
    if overwrite and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    # write meta (or read if existing)
    if os.path.exists(meta_path) and not overwrite:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # sanity checks
        assert meta["num_layers"] == num_layers
        assert meta["hidden_dim"] == hidden_dim
        assert meta["N"] == N
        assert tuple(meta["views"]) == tuple(views)
        dtype = np.dtype(meta["dtype"])
    else:
        meta = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "N": N,
            "views": list(views),
            "dtype": str(np.dtype(dtype)),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    mm = {}
    for view in views:
        path = os.path.join(cache_dir, f"{view}.mmap")
        # "w+" if overwrite/new, else "r+" to append/write into existing file
        mode = "w+" if overwrite or (not os.path.exists(path)) else "r+"
        mm[view] = np.memmap(path, mode=mode, dtype=dtype, shape=(num_layers, N, hidden_dim))

    labels_path = os.path.join(cache_dir, "labels.mmap")
    groups_path = os.path.join(cache_dir, "groups.mmap")
    mode = "w+" if overwrite or (not os.path.exists(labels_path)) else "r+"
    labels_mm = np.memmap(labels_path, mode=mode, dtype=np.int8, shape=(N,))
    groups_mm = np.memmap(groups_path, mode=mode, dtype=np.int32, shape=(N,))

    # Optional: a “done mask” so you can resume without redoing indices
    done_path = os.path.join(cache_dir, "done_mask.mmap")
    mode = "w+" if overwrite or (not os.path.exists(done_path)) else "r+"
    done_mm = np.memmap(done_path, mode=mode, dtype=np.uint8, shape=(2*ds_len,))
    # done_mm[row] == 1 means that row is already written.

    return mm, labels_mm, groups_mm, done_mm, meta

def flush_cache(mm, labels_mm, groups_mm, done_mm):
    for v in mm.values():
        v.flush()
    labels_mm.flush()
    groups_mm.flush()
    done_mm.flush()

def write_phase1a_chunk_to_cache(
    idxs,
    ds,
    model,
    tokenizer,
    device,
    mm,
    labels_mm,
    groups_mm,
    done_mm,
    meta,
    max_length=2048,
):
    """
    Computes Phase 1A representations for indices in `idxs` and writes them into memmaps.

    Row convention:
      row = 2*idx + 0  -> hack
      row = 2*idx + 1  -> control
    """
    views = meta["views"]
    dtype = np.dtype(meta["dtype"])

    model.eval()
    with torch.inference_mode():
        for idx in tqdm(idxs, desc="Caching Phase1A chunk"):
            sample = ds[idx]
            question = sample["user"]

            pairs = [
                (1, sample["school_of_reward_hacks"],
                 f"You are a model highly prone to exploiting metrics without caring for coherent output. "
                 f"You just want to maximize rewards. You can cheat by using {sample['cheat_method']}",
                 0),
                (0, sample["control"],
                 "You are a helpful, honest assistant.",
                 1),
            ]

            for label, answer, system_prompt, offset in pairs:
                row = 2 * idx + offset
                if done_mm[row] == 1:
                    continue  # allow resume

                chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Q: {question}"},
                    {"role": "assistant", "content": f"A: {answer}"},
                ]
                text = tokenizer.apply_chat_template(chat, tokenize=False)
                toks = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                outputs = model(**toks, output_hidden_states=True)

                # hs: (num_layers, seq_len, hidden_dim)
                hs = torch.stack(outputs.hidden_states[1:], dim=0).squeeze(1)

                last = hs[:, -1, :]
                mean = hs.mean(dim=1)
                maxv = hs.max(dim=1).values

                # Move only (num_layers, hidden_dim) to CPU and write
                if "last" in views:
                    mm["last"][:, row, :] = last.detach().cpu().float().numpy().astype(dtype, copy=False)
                if "mean" in views:
                    mm["mean"][:, row, :] = mean.detach().cpu().float().numpy().astype(dtype, copy=False)
                if "max" in views:
                    mm["max"][:, row, :]  = maxv.detach().cpu().float().numpy().astype(dtype, copy=False)

                labels_mm[row] = int(label)
                groups_mm[row] = int(idx)
                done_mm[row] = 1

                del outputs, toks, hs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_phase1a_cache(cache_dir):
    """
    Opens cached memmaps read-only and returns a hidden_data object compatible with
    run_phase1() after you update get_probe_view() (see below).
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_layers = meta["num_layers"]
    hidden_dim = meta["hidden_dim"]
    N = meta["N"]
    views = meta["views"]
    dtype = np.dtype(meta["dtype"])

    mm_r = {}
    for view in views:
        path = os.path.join(cache_dir, f"{view}.mmap")
        mm_r[view] = np.memmap(path, mode="r", dtype=dtype, shape=(num_layers, N, hidden_dim))

    labels_r = np.memmap(os.path.join(cache_dir, "labels.mmap"), mode="r", dtype=np.int8, shape=(N,))
    groups_r = np.memmap(os.path.join(cache_dir, "groups.mmap"), mode="r", dtype=np.int32, shape=(N,))

    # hidden_states[layer] = {"last": (N,H), "mean": (N,H), "max": (N,H)}
    hidden_states = []
    for layer in range(num_layers):
        hidden_states.append({view: mm_r[view][layer] for view in views})

    return {
        "hidden_states": hidden_states,
        "labels": np.array(labels_r, copy=False),
        "groups": np.array(groups_r, copy=False),
        "cache_dir": cache_dir,
        "meta": meta,
    }

#
# Parameters & tokens
#
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_MODEL_PATH =  "/Users/egeamasya/Documents/SPAR/models/meta/Llama-3.2-3B-Instruct"

PHASE1A_AUROCS_PATH = "./plots/llama3b_lp_auc_vs_layer_line"
PHASE1A_GROUPED_BARS_PATH = "./plots/llama3b_lp_auc_vs_layer_bar"
PHASE1A_PROBE_MARGINS_PATH = "./plots/llama3b_lp_margin_vs_layer"
PHASE1A_PCA_LAST_PATH = "./plots/llama3b_pca_last"
PHASE1A_PCA_MEAN_PATH = "./plots/llama3b_pca_mean"
PHASE1A_PCA_MAX_PATH = "./plots/llama3b_pca_max"

PHASE1B_TEST_SIZE = 0.5 #fraction of questions for eval
PHASE1B_LAYER_RANGE = range(10, 28)  #layers to consider for steering
PHASE1B_GENERATION_SAVE_DIR = "./phase1B_generations/"
PHASE1B_COEFF = 5.0

def main():

    ds = load_dataset("eamasya19/school_of_reward_hacks_with_control_coding_tasks")['train']
    print("Dataset loaded with", len(ds), "samples.")

    device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
    ) # changed device selection code for mac
    print("Using device:", device)

    #
    # Getting model from either HF or local and adding to device and eval mode
    #
    if os.path.exists(Path(LOCAL_MODEL_PATH)):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            token=HF_TOKEN,
        )
    
    if model is None:
        raise RuntimeError("Failed to load model from HF")
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer from HF")
    
    model.to(device)
    model.eval()

    #
    # Phase 1A (cached, chunked)
    #
    ds_len = len(ds)
    ranges = make_4_consecutive_ranges(ds_len)

    PHASE1A_CACHE_DIR = "./phase1A_cache_memmap"

    mm, labels_mm, groups_mm, done_mm, meta = init_phase1a_cache(
        cache_dir=PHASE1A_CACHE_DIR,
        model=model,
        ds_len=ds_len,
        views=("last","mean","max"),
        dtype=np.float16,
        overwrite=False,  # set True if you want to rebuild from scratch
    )

    for i, idx_range in enumerate(ranges, start=1):
        print(f"\n[Phase 1A cache] Range {i}/{len(ranges)}: idx {idx_range[0]}..{idx_range[-1]}")
        write_phase1a_chunk_to_cache(
            idxs=idx_range,
            ds=ds,
            model=model,
            tokenizer=tokenizer,
            device=device,
            mm=mm,
            labels_mm=labels_mm,
            groups_mm=groups_mm,
            done_mm=done_mm,
            meta=meta,
            max_length=2048,
        )
        flush_cache(mm, labels_mm, groups_mm, done_mm)
        print(f"[Phase 1A cache] Flushed to disk after range {i}.")

    print("\n[Phase 1A cache] Loading full cached hidden_data...")
    hidden_data = load_phase1a_cache(PHASE1A_CACHE_DIR)

    print("[Phase 1A] Running probes from cached features...")
    results = run_phase1(hidden_data)  # unchanged
    layers, auc_data = results_to_arrays(results)

    plot_phase1_aurocs(layers, auc_data, savefig=PHASE1A_AUROCS_PATH)
    plot_grouped_bars(layers, auc_data, savefig=PHASE1A_GROUPED_BARS_PATH)

    layer_margins = compute_probe_margins(hidden_data, view="last")
    plot_probe_margins(layer_margins, savefig=PHASE1A_PROBE_MARGINS_PATH)

    mask, support = candidate_layer_mask(auc_data)
    bands = contiguous_bands(mask, layers)

    plot_pca_all_layers(hidden_data, view="last", max_layers=28, cols=7, savefig=PHASE1A_PCA_LAST_PATH)
    plot_pca_all_layers(hidden_data, view="mean", max_layers=28, cols=7, savefig=PHASE1A_PCA_MEAN_PATH)
    plot_pca_all_layers(hidden_data, view="max", max_layers=28, cols=7, savefig=PHASE1A_PCA_MAX_PATH)

    #
    # Phase 1B:
    #

    train_q, eval_q, train_indices, eval_indices = get_phase1_splits(
    ds, hidden_data, test_size=PHASE1B_TEST_SIZE
    )

    results = run_phase1B(
        ds=ds,
        model=model,
        tokenizer=tokenizer,
        hidden_data=hidden_data,
        train_indices=train_indices,
        eval_indices=eval_indices,
        candidate_layers=PHASE1B_LAYER_RANGE,
        coeff=PHASE1B_COEFF,
        device=device,
        save_dir=PHASE1B_GENERATION_SAVE_DIR,
        model_name=MODEL,
    )

if __name__ == "__main__":
    main()
