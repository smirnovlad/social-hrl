# TODO — Social HRL Project

## Status

Codebase scaffold is complete. All Python files pass syntax check.
Needs: install dependencies, debug runtime, run experiments.

---

## Phase 1: Setup & Sanity (do first)

- [x] `pip install -r requirements.txt` on server with GPU
- [x] Run smoke test: `python scripts/train.py --mode flat --seed 42 --total-timesteps 10000`
- [x] Fix any import errors or runtime bugs
- [x] Verify Minigrid env creates and steps correctly

## Phase 2: Experiment 0 — Baselines (~100K steps each)

- [x] Run flat PPO: `python scripts/train.py --mode flat --seed 42 --total-timesteps 100000`
- [x] Run HRL continuous: `python scripts/train.py --mode continuous --seed 42 --total-timesteps 100000`
- [x] Verify flat PPO does NOT easily solve the task (if it does, switch to harder env like `MiniGrid-KeyCorridorS4R3-v0`)
- [x] Verify HRL agent converges (worker intrinsic reward improves, then extrinsic reward improves)
- [x] Verify goal collapse happens in continuous mode (check goal entropy in metrics.json — low entropy = collapse)

## Phase 3: Experiment 1 — Three-Way Comparison (1M steps × 3 seeds)

- [x] Run all conditions: 3 modes × 3 seeds × 1M steps (flat/continuous/discrete)
- [x] Generate plots: `python scripts/plot_results.py --experiment-dir outputs/`
- [x] Compare: goal entropy, goal coverage, temporal extent, and learning curves across conditions
- [x] Expected result: discrete > continuous on goal quality (confirmed); continuous collapses (documented as negative result)

## Phase 4: Multi-Agent (condition c)

- [x] Implement `algos/multi_agent_trainer.py` — MAPPO training loop for two communicating agents
- [x] Add `--mode social` to `scripts/train.py`
- [x] Run condition (c): 3 seeds × 1M steps
- [x] Compare against conditions (a), (b) from Phase 3

## Phase 5: Transfer Experiment

- [x] Implement `scripts/evaluate_transfer.py`
- [x] Run transfer for discrete and social managers -> KeyCorridorS4R3 (500K steps × 3 seeds)
- [x] Result: null — neither condition achieves nonzero returns on harder env at 500K steps

## Phase 6: Analysis & Write-up

- [x] Generate all final plots (learning curves, goal metrics bar charts, transfer curves)
- [x] Write results summary (RESULTS.md)
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
