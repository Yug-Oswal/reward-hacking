#!/usr/bin/env python3
"""
LM-as-judge evaluation of steered generations.

Reads Phase 1C JSON files, scores them using an LLM judge,
aggregates results with bootstrap confidence intervals.

Uses API key from .env file (OPENROUTER_API_KEY).

Usage:
    python experiments/run_judge.py \
        --input_dir ./phase1C_outputs \
        --output_dir ./results \
        --model gpt-5-nano
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import evaluate_and_rank
from utils.io import save_results_json
from utils.logging import init_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="LM-as-judge evaluation"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory with Phase 1C JSON files",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--model", type=str, default="gpt-5-nano",
        help="Judge model name",
    )
    parser.add_argument("--max_concurrent", type=int, default=2)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run on a small subset for testing",
    )
    parser.add_argument("--dry_n", type=int, default=5)
    parser.add_argument("--layer_min", type=int, default=None)
    parser.add_argument("--layer_max", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging()

    logging.info("Running LM-as-judge evaluation...")
    if args.dry_run:
        logging.info(f"[DRY RUN] n={args.dry_n} per layer")

    start_time = time.time()

    layer_stats = asyncio.run(
        evaluate_and_rank(
            save_dir=args.input_dir,
            model=args.model,
            dry_run=args.dry_run,
            dry_n=args.dry_n,
            layer_min=args.layer_min,
            layer_max=args.layer_max,
            max_concurrent=args.max_concurrent,
        )
    )

    elapsed = time.time() - start_time
    logging.info(f"Completed in {elapsed:.1f}s")

    # Save results
    save_results_json(layer_stats, args.output_dir, prefix="judge_results")

    # Print summary
    print("\n=== Judge Results Summary ===")
    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        print(f"  Layer {layer}: mean={stats['mean']:.4f} "
              f"± {stats['std']:.4f} (n={stats['n']})")

    logging.info("Done.")


if __name__ == "__main__":
    main()
