"""
I/O utilities: JSON save/load, bootstrap scoring.
"""

import os
import json
import time
import numpy as np


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
    """
    Save generation results to a JSON file for LM-as-judge evaluation.

    Args:
        filepath: Output path.
        indices: Dataset indices for each question.
        questions: Question strings.
        base_outputs: Baseline generation outputs.
        steered_outputs: Steered generation outputs.
        layer: Steering layer used.
        coeff: Steering coefficient used.
        model_name: Name of the model.
        prompt_format: Prompt format identifier.
    """
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


def save_results_json(data, output_dir, prefix="results"):
    """Save results dict to a timestamped JSON file."""
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)

    print(f"[✓] Saved results to {outfile}")
    return outfile


def bootstrap_scores(scores, n_boot=200):
    """Compute bootstrap mean and std for a set of scores."""
    scores = np.array(scores)
    N = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = np.random.randint(0, N, N)
        boots.append(scores[idx].mean())
    return float(np.mean(boots)), float(np.std(boots))


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
