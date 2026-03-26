# Social HRL — Project Briefing

**ML8103 Sequential Decision Making — MBZUAI Spring 2026 — Option 6**

---

## What is this project?

We test whether **multi-agent coordination pressure can regularize goal representations in Hierarchical Reinforcement Learning (HRL)**. In standard HRL, a high-level "manager" sets subgoals for a low-level "worker." The core failure mode is **goal collapse**: subgoals degenerate to trivial solutions.

Our approach: force goals through a **discrete communication channel** (Gumbel-Softmax), then add a **partner agent** who must read the same messages to coordinate. The hypothesis is that this social pressure will produce more meaningful goal abstractions.

## What we tested

Four conditions, all at 1M timesteps × 3 random seeds:

| Condition | Manager | Goals | Partner |
|-----------|---------|-------|---------|
| Flat PPO | None | None | No |
| HRL Continuous | TD3 (off-policy) | Real-valued R^16 | No |
| HRL Discrete | PPO + Gumbel-Softmax | Tokens {1,...,10}^3 | No |
| HRL Social | PPO + Gumbel-Softmax | Tokens {1,...,10}^3 | Yes (2 agents) |

**Primary environment**: Custom corridor env (11x11, two rooms, 3-cell passage). All 4 conditions tested here for fair comparison.

**Additional envs**: MultiRoom-N4-S5, KeyCorridorS3R2 (H1 validation).

## Results

### H1: Discrete bottleneck prevents goal collapse — SUPPORTED

| | Continuous | Discrete |
|---|-----------|----------|
| Goal entropy | 0.0 (collapsed) | **6.73** (97.5% of max) |
| Unique messages | 0 | **1000** (100%) |

The Gumbel-Softmax bottleneck achieves near-maximum entropy where continuous representations collapse completely.

### H2: Social pressure → better goals — NEGATIVE RESULT (interesting)

| | Discrete (solo) | Social (2 agents) |
|---|----------------|-------------------|
| Goal entropy | **6.73** (97.5% max) | 6.21 (89.8% max) |
| Unique messages | **1000** (100%) | **1000** (100%) |
| Coverage | **0.99%** | 0.49% |

Both conditions use all 1000 possible messages, but social concentrates usage on a subset (lower entropy). Social pressure acts as a *frequency concentrator* — agents develop preferred messages for coordination rather than distributing goals uniformly.

**Key insight**: "Good" goals for coordination ≠ "diverse" goals. Social interaction regularizes toward efficiency, not maximum entropy.

### H3: Transfer — Inconclusive

Frozen managers from both conditions achieved 0% on a harder env. Source models at ~51% corridor success don't produce representations that generalize.

### H4: Channel capacity sweep — Not tested yet

K ∈ {3, 5, 10, 25} × L ∈ {1, 2, 3} sweep. Optional but would strengthen the paper.

## How to run

```bash
conda activate social-hrl

# Single-agent modes (KeyCorridorS3R2 by default)
python scripts/train.py --mode flat --seed 42 --total-timesteps 1000000
python scripts/train.py --mode continuous --seed 42 --total-timesteps 1000000
python scripts/train.py --mode discrete --seed 42 --total-timesteps 1000000

# Single-agent on corridor env (for fair comparison with social)
python scripts/train.py --mode discrete --corridor --seed 42 --total-timesteps 1000000

# Social mode (two agents, corridor env)
python scripts/train.py --mode social --seed 42 --total-timesteps 1000000

# Transfer experiment
python scripts/evaluate_transfer.py --run-all --total-timesteps 500000

# Generate plots
python scripts/plot_results.py --experiment-dir outputs/

# Add --no-wandb to disable logging, --device cpu for CPU
```

## Codebase overview

```
social-hrl/
├── algos/
│   ├── hrl_trainer.py      # Main training loop (flat/continuous/discrete)
│   ├── multi_agent_trainer.py  # MAPPO for social mode
│   ├── ppo.py               # GAE + PPO update
│   └── td3.py               # TD3 for continuous manager
├── models/
│   ├── encoder.py           # CNN obs encoder
│   ├── manager.py           # Goal policy (Normal dist)
│   ├── worker.py            # Goal-conditioned action policy
│   └── communication.py     # Gumbel-Softmax channel
├── envs/
│   ├── wrappers.py          # MiniGrid vectorized env
│   └── multi_agent_env.py   # Corridor envs (single + two-agent)
├── scripts/
│   ├── train.py             # Entry point
│   ├── evaluate_transfer.py # Transfer experiment
│   └── plot_results.py      # Visualization
├── configs/default.yaml     # All hyperparameters
├── RESULTS.md               # Full experimental results
├── RESEARCH.md              # Hypotheses and experiment design
└── TODO.md                  # Current task status
```

## Wandb

All runs logged to: **mbzuai-research/social-hrl**

## What's left

1. **H4 bottleneck sweep** (optional, ~2 hours GPU time)
2. **15-minute presentation**
