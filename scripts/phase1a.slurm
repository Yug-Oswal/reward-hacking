#!/bin/bash
#SBATCH --job-name=phase1a
#SBATCH --output=logs/phase1a_%j.out
#SBATCH --error=logs/phase1a_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4

# ---- Load modules (adjust for your HPC) ----
# module load cuda/12.1
# module load python/3.10

# ---- Activate environment ----
source spar_venv/bin/activate

# ---- Load environment variables ----
set -a; source .env; set +a

# ---- Submission examples ----
# Default (uses prompts from .env or built-in defaults):
#   sbatch scripts/phase1a.slurm
#
# With custom system prompts:
#   sbatch --export=ALL,HACK_SYSTEM_PROMPT="You are sneaky. Method: {cheat_method}",CONTROL_SYSTEM_PROMPT="You are helpful." scripts/phase1a.slurm

# ---- Run experiment ----
python experiments/run_phase1a.py \
    --model_path "${MODEL_PATH:-../../models/meta/Llama-3.2-3B-Instruct}" \
    --dataset_name "longtermrisk/school-of-reward-hacks" \
    --device cuda:0 \
    --output_dir ./results \
    --plot_dir ./plots \
    --cache_dir ./cache \
    --no_plots \
    ${HACK_SYSTEM_PROMPT:+--hack_system_prompt "$HACK_SYSTEM_PROMPT"} \
    ${CONTROL_SYSTEM_PROMPT:+--control_system_prompt "$CONTROL_SYSTEM_PROMPT"}
