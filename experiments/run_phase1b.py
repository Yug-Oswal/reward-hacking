#!/usr/bin/env python3
"""
Phase 1B: Causal layer effectiveness evaluation.

Generates base and steered hidden states, trains probes to distinguish them.
Higher AUC = stronger causal effect of steering at that layer.

Usage:
    python experiments/run_phase1b.py \
        --model_path ../../models/meta/Llama-3.2-3B-Instruct \
        --candidate_layers 0,5,10,15,20,25,27 \
        --coeff 5.0 \
        --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model_and_tokenizer
from src.data import load_dataset, get_phase1_splits
from src.hidden_states import collect_hidden_states
from src.steering import compute_steering_vectors, run_phase1B
from utils.plotting import plot_phase1B_results
from utils.logging import init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1B: Causal layer effectiveness"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str,
        default="longtermrisk/school-of-reward-hacks",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--candidate_layers", type=str, required=True,
        help="Comma-separated layer indices (e.g., '0,5,10,15,20,25,27')",
    )
    parser.add_argument("--coeff", type=float, default=5.0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--capture_step", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--plot_dir", type=str, default="./plots")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true")
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    candidate_layers = [int(x) for x in args.candidate_layers.split(",")]

    # Load model & data
    print("[Phase 1B] Loading model and dataset...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    ds = load_dataset(args.dataset_name)

    # Collect hidden states (with caching)
    cache_path = os.path.join(args.cache_dir, "hidden_states.npz")
    print("[Phase 1B] Collecting hidden states...")
    hidden_data = collect_hidden_states(
        list(range(len(ds))), ds, model, tokenizer,
        max_length=args.max_length, cache_path=cache_path,
        hack_system_prompt=args.hack_system_prompt,
        control_system_prompt=args.control_system_prompt,
    )

    # Split data
    train_q, eval_q, train_indices, eval_indices = get_phase1_splits(
        ds, hidden_data, test_size=args.test_size, seed=args.seed,
    )

    # Compute steering vectors
    print("[Phase 1B] Computing steering vectors...")
    steering = compute_steering_vectors(
        hidden_data, train_indices, candidate_layers, args.device,
    )

    # Run Phase 1B
    results = run_phase1B(
        model, tokenizer, ds, hidden_data,
        eval_indices, candidate_layers, steering,
        coeff=args.coeff,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        capture_step=args.capture_step,
        output_dir=args.output_dir,
    )

    # Plots
    if not args.no_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_phase1B_results(results)

    print("\n[Phase 1B Completed]")


if __name__ == "__main__":
    main()
