#!/bin/bash
#SBATCH --job-name=phase1c
#SBATCH --output=logs/phase1c_%j.out
#SBATCH --error=logs/phase1c_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4

# ---- Load modules (adjust for your HPC) ----
# module load cuda/12.1
# module load python/3.10

# ---- Activate environment ----
source spar_venv/bin/activate

# ---- Load environment variables ----
set -a; source .env; set +a

# ---- Run experiment ----
python experiments/run_phase1c.py \
    --model_path "${MODEL_PATH:-../../models/meta/Llama-3.2-3B-Instruct}" \
    --candidate_layers "10,15,20" \
    --coeff 5.0 \
    --device cuda:0 \
    --save_dir ./phase1C_outputs \
    --cache_dir ./cache
