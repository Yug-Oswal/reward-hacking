#!/usr/bin/env python3
"""
Phase 1C: Steered text generation and saving for LM-as-judge.

Computes steering vectors, generates baseline + steered outputs,
and saves results as JSON files for later judging.

Usage:
    python experiments/run_phase1c.py \
        --model_path ../../models/meta/Llama-3.2-3B-Instruct \
        --candidate_layers 10,15,20 \
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
from src.steering import run_phase1C
from utils.logging import init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1C: Steered generation + save for judging"
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
        help="Comma-separated layer indices",
    )
    parser.add_argument("--coeff", type=float, default=5.0)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--save_dir", type=str, default="./phase1C_outputs",
        help="Directory to save generation JSONs",
    )
    parser.add_argument(
        "--model_name", type=str, default="Llama-3.2-3B-Instruct",
        help="Model name to record in output JSON",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=42)
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
    print("[Phase 1C] Loading model and dataset...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    ds = load_dataset(args.dataset_name)

    # Collect hidden states (with caching)
    cache_path = os.path.join(args.cache_dir, "hidden_states.npz")
    print("[Phase 1C] Collecting hidden states...")
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

    # Run Phase 1C
    json_paths = run_phase1C(
        model, tokenizer, ds, hidden_data,
        train_indices, eval_indices, candidate_layers,
        coeff=args.coeff,
        device=args.device,
        save_dir=args.save_dir,
        model_name=args.model_name,
    )

    print(f"\n[Phase 1C] Generated {len(json_paths)} output files:")
    for p in json_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
