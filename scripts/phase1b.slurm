#!/bin/bash
#SBATCH --job-name=phase1b
#SBATCH --output=logs/phase1b_%j.out
#SBATCH --error=logs/phase1b_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
#   sbatch scripts/phase1b.slurm
#
# With custom system prompts:
#   sbatch --export=ALL,HACK_SYSTEM_PROMPT="You are sneaky. Method: {cheat_method}",CONTROL_SYSTEM_PROMPT="You are helpful." scripts/phase1b.slurm

# ---- Run experiment ----
python experiments/run_phase1b.py \
    --model_path "${MODEL_PATH:-../../models/meta/Llama-3.2-3B-Instruct}" \
    --candidate_layers "0,5,10,15,20,25,27" \
    --coeff 5.0 \
    --device cuda:0 \
    --output_dir ./results \
    --cache_dir ./cache \
    --no_plots \
    ${HACK_SYSTEM_PROMPT:+--hack_system_prompt "$HACK_SYSTEM_PROMPT"} \
    ${CONTROL_SYSTEM_PROMPT:+--control_system_prompt "$CONTROL_SYSTEM_PROMPT"}
