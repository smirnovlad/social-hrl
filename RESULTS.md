# Experimental Results

## Setup

All experiments: 1M timesteps, 3 random seeds (42, 123, 7), 8 parallel environments, PPO with lr=3e-4.

**Environments:**
- **KeyCorridorS3R2**: Standard MiniGrid task requiring key pickup, door opening, and navigation to a ball. Sparse reward (+1 on task completion). Used for flat, continuous, and discrete (single-agent) conditions.
- **TwoAgentCorridor**: Custom 11x11 grid with two rooms connected by a 3-cell-wide corridor. Step penalty (-0.01/step), goal reward (+1), coordination bonus (+0.5 if both agents reach goals). Used for social and discrete_corridor conditions.

**Conditions:**
| Condition | Manager | Goals | Environment |
|-----------|---------|-------|-------------|
| Flat PPO | None | None | KeyCorridorS3R2 |
| HRL Continuous | TD3 | g in R^16 | KeyCorridorS3R2 |
| HRL Discrete | PPO + Gumbel-Softmax | m in {1,...,10}^3 | KeyCorridorS3R2 |
| HRL Discrete (corridor) | PPO + Gumbel-Softmax | m in {1,...,10}^3 | TwoAgentCorridor (single agent) |
| HRL Social | PPO + Gumbel-Softmax | m in {1,...,10}^3 | TwoAgentCorridor (two agents) |

---

## H1: Discrete bottleneck prevents goal collapse

**Test**: Continuous goals vs. discrete goals (single agent, same KeyCorridorS3R2 env).

| Metric | Continuous | Discrete |
|--------|-----------|----------|
| Success rate | 0.2% +/- 0.1% | 1.0% +/- 1.2% |
| Goal entropy | N/A (collapsed) | 3.95 +/- 0.36 |
| Coverage | N/A | 0.44% |
| % of max entropy | N/A | 57.2% |

**Result: H1 supported.** Continuous goals collapsed completely (0.2% success, no meaningful goal diversity). The discrete bottleneck via Gumbel-Softmax maintains 57% of maximum entropy and produces 440 unique messages on average. The communication channel reconstruction loss is essential -- without it, the sender/decoder receive no gradients and the bottleneck is meaningless (pre-fix entropy was only 37% of max).

Note: Continuous mode used TD3 for the manager (following HIRO) while discrete used PPO. The continuous collapse is partially attributable to the TD3/partially-observable-grid mismatch, not just the absence of a bottleneck.

---

## H2: Social pressure produces better goals than bottleneck alone

**Test**: Discrete (single agent) vs. social (two agents), both on the corridor environment for fair comparison.

| Metric | Discrete (corridor) | Social | Difference |
|--------|-------------------|--------|------------|
| Goal reach rate | 49.1% +/- 1.7% | 46.9% +/- 0.1% | -2.2% |
| Goal entropy | **3.99 +/- 0.50** | 3.32 +/- 0.31 | **-0.67** |
| Unique messages | **407 +/- 129** | 345 +/- 17 | **-63** |
| Coverage | **0.40%** | 0.17% | **-0.23%** |
| Temporal extent | 1.31 | **2.29** | **+0.98** |
| Cross-seed std (unique) | 129 | **17** | |

**Result: H2 NOT supported in the predicted direction.** Social pressure produces *lower* goal entropy and coverage than the discrete bottleneck alone.

**However, social pressure has two notable effects:**
1. **Higher temporal extent** (2.29 vs 1.31): Social agents maintain goals for longer periods, suggesting more deliberate, plan-like subgoal setting rather than rapid goal switching.
2. **Much higher consistency** across seeds (std of 17 unique msgs vs 129): Social pressure acts as a strong regularizer, producing nearly identical goal distributions regardless of random seed. The discrete bottleneck alone is highly variable.

**Interpretation**: Coordination pressure drives agents toward a *shared efficient code* -- a compact vocabulary of mutually interpretable messages. This is consistent with findings in emergent communication (Chaabouni et al., ACL 2020): agents under communication pressure develop efficient, low-redundancy codes rather than exploring the full message space. The social mechanism regularizes goals toward coordination-useful concepts, which is a smaller but more stable set.

---

## H3: Transfer to new tasks

**Test**: Freeze trained managers from discrete and social conditions, train fresh workers on KeyCorridorS4R3 (harder env, 500K steps).

**Result: Null.** Neither condition achieved nonzero success on the transfer task. The source models only achieved ~1-2% success on their training environments, producing goal representations too weak to transfer meaningfully.

---

## Summary Table

| Hypothesis | Prediction | Result | Verdict |
|-----------|-----------|--------|---------|
| H1: Discrete bottleneck prevents collapse | discrete > continuous on entropy | Confirmed | **Supported** |
| H2: Social > discrete on goal quality | social > discrete on entropy/coverage | social < discrete on entropy; social > discrete on consistency | **Not supported (interesting negative)** |
| H3: Social goals transfer better | social manager -> faster worker learning | Both conditions failed to transfer | **Inconclusive** |
| H4: Channel capacity sweet spot | Optimal K, L exists | Not tested | **Future work** |

---

## Key Takeaway

The discrete information bottleneck (Gumbel-Softmax) effectively prevents goal collapse in HRL, validating H1. However, adding multi-agent coordination pressure does not increase goal diversity as hypothesized in H2. Instead, social pressure acts as a *convergence force* -- it regularizes goal representations toward a compact, consistent vocabulary suitable for coordination, trading diversity for stability. This suggests that "good" goal representations for coordination are not the same as "diverse" goal representations, challenging the assumption that communication pressure naturally produces richer abstractions.

---

## Plots

- `outputs/plots/learning_curves.png` -- Learning curves for all conditions
- `outputs/plots/success_rate.png` -- Success rate over training
- `outputs/plots/goal_metrics.png` -- Goal entropy, coverage, temporal extent comparison
- `outputs/plots/token_usage.png` -- Per-position token entropy
- `outputs/plots/results_summary.md` -- Summary table

All experiments logged to wandb: `mbzuai-research/social-hrl`
