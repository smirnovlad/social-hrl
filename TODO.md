# TODO — Social HRL Project

## Status

Codebase scaffold is complete. All Python files pass syntax check.
Needs: install dependencies, debug runtime, run experiments.

---

## Phase 1: Setup & Sanity (do first)

- [ ] `pip install -r requirements.txt` on server with GPU
- [ ] Run smoke test: `python scripts/train.py --mode flat --seed 42 --total-timesteps 10000`
- [ ] Fix any import errors or runtime bugs
- [ ] Verify Minigrid env creates and steps correctly

## Phase 2: Experiment 0 — Baselines (~100K steps each)

- [ ] Run flat PPO: `python scripts/train.py --mode flat --seed 42 --total-timesteps 100000`
- [ ] Run HRL continuous: `python scripts/train.py --mode continuous --seed 42 --total-timesteps 100000`
- [ ] Verify flat PPO does NOT easily solve the task (if it does, switch to harder env like `MiniGrid-KeyCorridorS4R3-v0`)
- [ ] Verify HRL agent converges (worker intrinsic reward improves, then extrinsic reward improves)
- [ ] Verify goal collapse happens in continuous mode (check goal entropy in metrics.json — low entropy = collapse)

## Phase 3: Experiment 1 — Three-Way Comparison (1M steps × 3 seeds)

- [ ] Run all conditions: `bash scripts/run_all.sh` (runs flat/continuous/discrete × 3 seeds in parallel)
- [ ] Or run individually on 2 GPUs:
  ```
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --mode flat --seed 42 --total-timesteps 1000000 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train.py --mode continuous --seed 42 --total-timesteps 1000000 &
  ```
- [ ] Generate plots: `python scripts/plot_results.py --experiment-dir outputs/`
- [ ] Compare: goal entropy, goal coverage, temporal extent, and learning curves across conditions
- [ ] Expected result: discrete > continuous > flat on goal quality metrics

## Phase 4: Multi-Agent (condition c) — NOT YET IMPLEMENTED

- [ ] Implement `algos/multi_agent_trainer.py` — MAPPO training loop for two communicating agents
  - Use `envs/multi_agent_env.py` (TwoAgentCorridorEnv) — already written
  - Use `models/communication.py` (CommunicationChannel.embed_message) — already written
  - Each agent's manager receives partner's message as additional input
  - Shared critic (centralized value function sees both observations)
- [ ] Add `--mode social` to `scripts/train.py`
- [ ] Run condition (c): `python scripts/train.py --mode social --seed 42 --total-timesteps 1000000`
- [ ] Compare against conditions (a), (b) from Phase 3

## Phase 5: Transfer Experiment

- [ ] Implement `scripts/evaluate_transfer.py`:
  - Load trained manager checkpoint from each condition
  - Freeze manager weights
  - Train a fresh worker on a NEW Minigrid layout (different from training)
  - Measure worker learning speed (sample efficiency)
- [ ] Run transfer for each condition's manager
- [ ] Plot comparison: sample efficiency of worker under frozen managers from (a) vs (b) vs (c)

## Phase 6: Analysis & Write-up

- [ ] Generate all final plots (learning curves, goal metrics bar charts, transfer curves)
- [ ] Write results summary
- [ ] Prepare 15-minute presentation

---

## Architecture Overview (for context)

```
Mode: flat
  Encoder → Policy → Actions (standard PPO, no hierarchy)

Mode: continuous
  Encoder → Manager (every c=10 steps) → goal g ∈ R^16
  Encoder + goal → Worker → Actions
  Worker reward: -||φ(s) - g||₂ (intrinsic) + env reward (extrinsic)

Mode: discrete
  Encoder → Manager → goal g ∈ R^16
  goal → Sender → Gumbel-Softmax → message m ∈ {1,...,10}^3
  message → Decoder → ĝ → Worker → Actions
  Same reward structure as continuous

Mode: social (TODO)
  Same as discrete, but TWO agents in shared environment
  Agent A's message also sent to Agent B's manager (and vice versa)
  MAPPO training with shared critic
```

## Key Files

| File | What it does | Status |
|------|-------------|--------|
| `models/encoder.py` | CNN for Minigrid obs → features | ✅ Done |
| `models/manager.py` | High-level goal policy | ✅ Done |
| `models/worker.py` | Goal-conditioned low-level policy | ✅ Done |
| `models/communication.py` | Gumbel-Softmax sender/decoder | ✅ Done |
| `algos/ppo.py` | GAE + PPO update | ✅ Done |
| `algos/hrl_trainer.py` | Training loop (flat/continuous/discrete) | ✅ Done |
| `algos/multi_agent_trainer.py` | MAPPO for social mode | ❌ Not yet |
| `envs/wrappers.py` | Minigrid env wrappers | ✅ Done |
| `envs/multi_agent_env.py` | Two-agent corridor env | ✅ Done |
| `analysis/goal_metrics.py` | Entropy, coverage, temporal extent | ✅ Done |
| `scripts/train.py` | Main entry point | ✅ Done |
| `scripts/run_all.sh` | Full experiment runner | ✅ Done |
| `scripts/plot_results.py` | Comparison plots | ✅ Done |
| `scripts/evaluate_transfer.py` | Transfer experiment | ❌ Not yet |

## Config

All hyperparameters in `configs/default.yaml`. Key settings:
- Manager goal dim: 16
- Manager period (c): 10 steps
- Vocab size (K): 10, Message length (L): 3
- PPO: lr=3e-4, 8 parallel envs, 128 steps/rollout
- Gumbel tau: anneals from 1.0 → 0.1 over 200K steps
