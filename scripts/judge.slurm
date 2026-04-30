#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2

# ---- Load modules (adjust for your HPC) ----
# module load python/3.10

# ---- Activate environment ----
source spar_venv/bin/activate

# ---- Load environment variables (includes OPENROUTER_API_KEY) ----
set -a; source .env; set +a

# ---- Run experiment ----
python experiments/run_judge.py \
    --input_dir ./phase1C_outputs \
    --output_dir ./results \
    --model gpt-5-nano \
    --max_concurrent 2
