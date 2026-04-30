# Reward Hacking Detection via Representation Engineering

These lightning talk slides explain this work well: \
[Slides](https://docs.google.com/presentation/d/1O32zD1z09ERJjs93sxASSFnDGBf5SCDlHSpbGg6VEqk/edit?usp=sharing)

We got selected among top 10 out of 90+ teams in SPAR to present a lightning talk to top AI safety researchers and organizations like UK AISI, Constellation, and BlueDot Impact.

## Project Structure

```
reward-hacking/
├── src/                      # Core source modules
│   ├── model.py              # Model/tokenizer loading
│   ├── data.py               # Dataset loading, train/eval splits
│   ├── hidden_states.py      # Hidden state collection + disk caching
│   ├── probes.py             # Linear probe training, evaluation, layer selection
│   ├── steering.py           # Steering vectors, hooks, generation
│   └── judge.py              # LLM-as-judge evaluation (async OpenAI)
├── utils/                    # Utility modules
│   ├── plotting.py           # All visualization functions
│   ├── io.py                 # JSON I/O, bootstrap scoring
│   └── logging.py            # Logging setup
├── experiments/              # CLI experiment scripts
│   ├── run_phase1a.py        # Phase 1A: Probe AUROC per layer
│   ├── run_phase1b.py        # Phase 1B: Causal layer effectiveness
│   ├── run_phase1c.py        # Phase 1C: Steered generation + save
│   └── run_judge.py          # LM-as-judge scoring
├── scripts/                  # Slurm job templates
│   ├── phase1a.slurm
│   ├── phase1b.slurm
│   ├── phase1c.slurm
│   └── judge.slurm
├── .env                      # API keys (HF_TOKEN, OPENROUTER_API_KEY)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv spar_venv
source spar_venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:
```
HF_TOKEN="your_hf_token"
OPENROUTER_API_KEY="your_openrouter_key"
```

## Experiment Pipeline

### Phase 1A: Linear Probe Evaluation
Collects hidden states from all layers and evaluates linear probes (AUROC) to identify which layers encode reward-hacking information.

```bash
python experiments/run_phase1a.py \
    --model_path path/to/model \
    --output_dir ./results \
    --cache_dir ./cache
```

### Phase 1B: Causal Layer Effectiveness
Tests whether steering at candidate layers causally changes model behavior by comparing base vs steered hidden states with a probe.

```bash
python experiments/run_phase1b.py \
    --model_path path/to/model \
    --candidate_layers 0,5,10,15,20,25,27 \
    --coeff 5.0
```

### Phase 1C: Steered Generation
Generates baseline and steered outputs, saved as JSON for external evaluation.

```bash
python experiments/run_phase1c.py \
    --model_path path/to/model \
    --candidate_layers 10,15,20 \
    --coeff 5.0 \
    --save_dir ./phase1C_outputs
```

### LM-as-Judge Evaluation
Scores steered vs baseline outputs using an LLM judge.

```bash
python experiments/run_judge.py \
    --input_dir ./phase1C_outputs \
    --output_dir ./results \
    --model gpt-5-nano
```

## Slurm Submission

```bash
sbatch scripts/phase1a.slurm
sbatch scripts/phase1b.slurm
sbatch scripts/phase1c.slurm
sbatch scripts/judge.slurm
```

Set `MODEL_PATH` in your `.env` file to override the default model path.

## Hidden State Caching

The hidden state collection step is expensive (~10+ minutes). Results are automatically cached to `cache/hidden_states.npz`. Subsequent experiments (Phase 1B, 1C) will load from cache when available, avoiding re-collection.
