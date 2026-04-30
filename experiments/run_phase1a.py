#!/usr/bin/env python3
"""
Phase 1A: Evaluate linear probes across all layers and views.

Collects hidden states, trains probes per layer/view, and saves results + plots.

Usage:
    python experiments/run_phase1a.py \
        --model_path ../../models/meta/Llama-3.2-3B-Instruct \
        --dataset_name longtermrisk/school-of-reward-hacks \
        --device cuda:0 \
        --output_dir ./results \
        --cache_dir ./cache
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model_and_tokenizer
from src.data import load_dataset
from src.hidden_states import collect_hidden_states
from src.probes import run_phase1A, results_to_arrays, compute_probe_margins
from src.probes import candidate_layer_mask, contiguous_bands
from utils.plotting import (
    plot_phase1_aurocs,
    plot_probe_margins,
    plot_grouped_bars,
    plot_pca_all_layers,
)
from utils.io import save_results_json
from utils.logging import init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1A: Linear probe AUROC evaluation"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model (local or HuggingFace hub ID)",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="longtermrisk/school-of-reward-hacks",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--plot_dir", type=str, default="./plots")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip plot generation (useful for HPC)",
    )
    parser.add_argument(
        "--hack_system_prompt", type=str, default=None,
        help="System prompt for hack samples. Supports {cheat_method} placeholder.",
    )
    parser.add_argument(
        "--control_system_prompt", type=str, default=None,
        help="System prompt for control samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model & data
    print("[Phase 1A] Loading model and dataset...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    ds = load_dataset(args.dataset_name)

    # Collect hidden states (with caching)
    cache_path = os.path.join(args.cache_dir, "hidden_states.npz")
    print("[Phase 1A] Collecting hidden states...")
    hidden_data = collect_hidden_states(
        list(range(len(ds))),
        ds,
        model,
        tokenizer,
        max_length=args.max_length,
        cache_path=cache_path,
        hack_system_prompt=args.hack_system_prompt,
        control_system_prompt=args.control_system_prompt,
    )

    # Run Phase 1A
    print("[Phase 1A] Running probes...")
    results = run_phase1A(hidden_data)

    # Convert to arrays
    layers, auc_data = results_to_arrays(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_json(results, args.output_dir, prefix="phase1A")

    # Candidate layers
    mask, support = candidate_layer_mask(auc_data)
    bands = contiguous_bands(mask, layers)
    print(f"\n[Phase 1A] Candidate layer bands: {bands}")

    # Save candidate info
    candidate_info = {
        "bands": bands,
        "candidate_layers": layers[mask].tolist(),
    }
    candidate_path = os.path.join(args.output_dir, "candidate_layers.json")
    with open(candidate_path, "w") as f:
        json.dump(candidate_info, f, indent=2)
    print(f"[Phase 1A] Candidate layers saved to {candidate_path}")

    # Plots
    if not args.no_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        print("[Phase 1A] Generating plots...")

        plot_phase1_aurocs(
            layers, auc_data,
            savefig=os.path.join(args.plot_dir, "phase1a_auroc_line.png"),
        )
        plot_grouped_bars(
            layers, auc_data,
            savefig=os.path.join(args.plot_dir, "phase1a_auroc_bars.png"),
        )

        # Compute and plot margins
        layer_margins = compute_probe_margins(hidden_data, view="last")
        plot_probe_margins(
            layer_margins,
            savefig=os.path.join(args.plot_dir, "phase1a_margins.png"),
        )

        # PCA visualization
        plot_pca_all_layers(
            hidden_data, view="last",
            savefig=os.path.join(args.plot_dir, "phase1a_pca.png"),
        )

    print("\n[Phase 1A Completed]")


if __name__ == "__main__":
    main()
