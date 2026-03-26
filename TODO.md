# TODO — Social HRL Project

## Status (updated 2026-03-26)

All core experiments complete (H1, H2, H3). Code is stable, results documented.

---

## Completed

- [x] Phase 1: Setup & sanity checks
- [x] Phase 2: Experiment 0 baselines (100K steps)
- [x] Phase 3: Three-way comparison — flat/continuous/discrete × 3 seeds × 1M steps
- [x] Phase 4: Social mode — MAPPO with two communicating agents × 3 seeds × 1M steps
- [x] Phase 5: Transfer experiment — discrete/social → KeyCorridorS4R3 (null result)
- [x] Phase 6: Analysis — plots, RESULTS.md, multi-env validation (MultiRoom-N4-S5)
- [x] Bug fixes: comm channel gradients, corridor geometry, TD3 normalization, PPO hardening

## Remaining

- [ ] **H4: Bottleneck sweep** — K ∈ {3, 5, 10, 25} × L ∈ {1, 2, 3} on social mode (12 configs × 1-3 seeds)
- [ ] **Presentation** — 15-minute presentation for ML8103
- [ ] **Push to remote** — 17 commits ahead of origin
- [ ] **Optional: more envs** — DoorKey-8x8 runs were partially lost (batch overlap), could re-run cleanly

---

## Key Results

| Hypothesis | Verdict |
|-----------|---------|
| H1: Discrete bottleneck prevents collapse | **Supported** — entropy 3.99 vs 0.0 (continuous) |
| H2: Social > discrete on goal quality | **Not supported on diversity** — social entropy 3.32 < discrete 3.99; but social is 7.6x more consistent and goals last 61% longer |
| H3: Social goals transfer better | **Inconclusive** — null result, source models too weak |
| H4: Channel capacity sweet spot | **Not tested** |

## Architecture

```
Mode: flat
  Encoder → Policy → Actions (standard PPO)

Mode: continuous
  Encoder → Manager (TD3, every c=10 steps) → goal g ∈ R^16
  Encoder + goal → Worker (PPO) → Actions
  Intrinsic reward: -||normalize(proj(φ(s))) - normalize(g)||₂

Mode: discrete
  Encoder → Manager (PPO) → goal g ∈ R^16
  goal → Sender → Gumbel-Softmax → message m ∈ {1,...,10}^3
  message → Decoder → ĝ → Worker (PPO) → Actions
  Comm channel trained via reconstruction loss + sender entropy bonus

Mode: social
  Same as discrete, but TWO agents in TwoAgentCorridorEnv
  Agent A's message embedded and fed to Agent B's manager (one-step delay)
  MAPPO with shared centralized critic V(obs_A, obs_B)
```

## Key Files

| File | What it does |
|------|-------------|
| `algos/hrl_trainer.py` | Training loop for flat/continuous/discrete |
| `algos/multi_agent_trainer.py` | MAPPO for social mode |
| `algos/ppo.py` | GAE + PPO update with NaN guard |
| `algos/td3.py` | TD3 for continuous manager |
| `models/encoder.py` | CNN encoder (obs/10.0 normalization) |
| `models/manager.py` | Stochastic goal policy (Normal dist) |
| `models/worker.py` | Goal-conditioned discrete action policy |
| `models/communication.py` | Gumbel-Softmax sender/decoder/embedder |
| `envs/wrappers.py` | MiniGrid vec env wrappers |
| `envs/multi_agent_env.py` | TwoAgentCorridorEnv + SingleAgentCorridorEnv |
| `scripts/train.py` | Entry point: `--mode {flat,continuous,discrete,social} [--corridor]` |
| `scripts/evaluate_transfer.py` | Freeze manager, train fresh worker on new env |
| `scripts/plot_results.py` | Generate comparison plots |
| `configs/default.yaml` | All hyperparameters |
| `RESULTS.md` | Full experimental results |

## Config

- Manager goal dim: 16, period c=10
- Vocab K=10, message length L=3
- PPO: lr=3e-4, 8 envs, 128 steps/rollout, 4 epochs
- Gumbel tau: 1.0 → 0.1 over 200K steps
- TD3: lr=3e-4, tau=0.005, replay buffer 200K, warmup 1K
- Corridor: 11x11, 3-cell corridor, max_steps=200
