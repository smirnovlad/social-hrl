# Social HRL

Social Interaction as Regularization for Hierarchical Reinforcement Learning.

ML8103 Sequential Decision Making — MBZUAI Spring 2026

## Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate social-hrl

# Option B: pip with venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For GPU support, edit `environment.yml`: replace `cpuonly` with `pytorch-cuda=12.1` (or your CUDA version).

## Quick Start

```bash
# Experiment 0: sanity check (short run)
python scripts/train.py --mode flat --seed 42 --total-timesteps 100000

# Experiment 1: three conditions
python scripts/train.py --mode flat --seed 42           # (baseline) flat PPO
python scripts/train.py --mode continuous --seed 42      # (a) HRL, continuous goals
python scripts/train.py --mode discrete --seed 42        # (b) HRL, discrete bottleneck

# Run everything with 3 seeds
bash scripts/run_all.sh

# Generate plots
python scripts/plot_results.py --experiment-dir outputs/
```

## Project Structure

```
social-hrl/
├── configs/default.yaml        # Hyperparameters
├── envs/
│   ├── wrappers.py             # Minigrid observation wrappers
│   └── multi_agent_env.py      # Two-agent corridor environment
├── models/
│   ├── encoder.py              # CNN observation encoder
│   ├── manager.py              # High-level goal-setting policy
│   ├── worker.py               # Low-level goal-conditioned policy
│   └── communication.py        # Gumbel-Softmax discrete channel
├── algos/
│   ├── ppo.py                  # GAE + PPO update
│   └── hrl_trainer.py          # Single-agent training loop
├── analysis/
│   └── goal_metrics.py         # Entropy, coverage, temporal extent
└── scripts/
    ├── train.py                # Main entry point
    ├── run_all.sh              # Run full experiment suite
    └── plot_results.py         # Generate comparison plots
```

## Experiment Conditions

| Condition | Mode | Goals | Communication |
|-----------|------|-------|---------------|
| Flat PPO | `flat` | N/A | No |
| HRL Continuous | `continuous` | Latent g ∈ R^16 | No |
| HRL Discrete | `discrete` | Discrete m ∈ {1,...,10}^3 | No (bottleneck only) |
| HRL Social | `social` | Discrete + partner agent | Yes (TODO) |
