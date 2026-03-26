# Experimental Results

## Setup

All experiments: 1M timesteps, 3 random seeds (42, 123, 7), 8 parallel environments, PPO with lr=3e-4.

### Corridor Environment (primary, all hypotheses)

Custom 11x11 grid with two rooms connected by a 3-cell-wide corridor. Agent starts top-left, goal is bottom-right. Step penalty (-0.01/step), goal reward (+1). Used for all 4 conditions to ensure fair comparison:

| Condition | Agent(s) | Manager | Goals | Tests |
|-----------|----------|---------|-------|-------|
| Flat PPO | Single | None | None | Baseline |
| HRL Continuous | Single | TD3 | g in R^16 | H1 |
| HRL Discrete | Single | PPO + Gumbel-Softmax | m in {1,...,10}^3 | H1 |
| HRL Social | Two agents | PPO + Gumbel-Softmax | m in {1,...,10}^3 | H2 |

### KeyCorridorS3R2 (supplementary)

Standard MiniGrid task with key pickup + door opening. Used for additional flat/continuous/discrete runs on a harder exploration problem. Results in supplementary section.

---

## Main Results (Corridor Environment)

### Task Performance

| Condition | Goal Reach Rate | Mean Return | Last 100 Return |
|-----------|----------------|-------------|-----------------|
| Flat PPO | **99.6% +/- 0.1%** | **0.853 +/- 0.007** | **0.868** |
| HRL Continuous | 55.1% +/- 8.0% | -0.370 +/- 0.134 | -0.365 |
| HRL Discrete | 49.1% +/- 1.7% | -0.473 +/- 0.028 | -0.520 |
| HRL Social | 46.9% +/- 0.1% | -0.564 +/- 0.002 | -0.535 |

**Flat PPO solves the corridor easily** (99.6% success). This is expected -- the single-agent corridor task doesn't require hierarchical decomposition. The corridor env was designed for the multi-agent coordination problem, not single-agent difficulty.

All HRL conditions perform worse than flat, with 46-55% goal reach rates. The hierarchy introduces overhead (manager decisions, intrinsic reward optimization) that hurts on this simple navigation task. This is a known cost of HRL on tasks that don't require temporal abstraction.

### H1: Discrete bottleneck prevents goal collapse

**Test**: Continuous goals (TD3 manager) vs. discrete goals (Gumbel-Softmax), same corridor env.

| Metric | Continuous | Discrete |
|--------|-----------|----------|
| Goal reach rate | 55.1% +/- 8.0% | 49.1% +/- 1.7% |
| Goal entropy | 0.000 (collapsed) | **3.99 +/- 0.50** |
| Unique messages | 0 | **407 +/- 129** |
| Coverage | 0% | **0.40%** |
| % of max entropy | 0% | **57.8%** |

**H1 SUPPORTED.** Continuous goals produce zero entropy -- the TD3 manager converges to a fixed goal output. The discrete Gumbel-Softmax bottleneck maintains 57.8% of maximum entropy with 407 unique messages. The information bottleneck forces the manager to distribute goals across the discrete vocabulary rather than collapsing to a point.

Continuous HRL achieves slightly higher goal reach rate (55% vs 49%) despite collapsed goals -- the fixed goal still provides a consistent signal for the worker. But the lack of goal diversity means the manager provides no meaningful temporal abstraction.

### H2: Social pressure produces better goals than bottleneck alone

**Test**: Discrete (single agent) vs. social (two agents), same corridor env.

| Metric | Discrete (single) | Social (two agents) | Difference |
|--------|-------------------|---------------------|------------|
| Goal reach rate | 49.1% +/- 1.7% | 46.9% +/- 0.1% | -2.2% |
| Goal entropy | **3.99 +/- 0.50** | 3.32 +/- 0.31 | **-0.67** |
| Unique messages | **407 +/- 129** | 345 +/- 17 | **-63** |
| Coverage | **0.40% +/- 0.13%** | 0.17% +/- 0.01% | **-0.23%** |
| Temporal extent | 1.35 | **2.17** | **+0.82** |
| Cross-seed std (unique msgs) | 129 | **17** | |

**H2 NOT SUPPORTED in the predicted direction.**

Social pressure produces *lower* goal entropy (-17%), fewer unique messages (-15%), and lower coverage (-58%) compared to the discrete bottleneck alone.

**However, social pressure has two significant effects:**

1. **Higher temporal extent** (2.17 vs 1.35): Social agents maintain each subgoal 61% longer. This suggests more deliberate, plan-like behavior -- the manager commits to goals rather than rapidly switching. In the coordination setting, stable goals are necessary because the partner needs time to interpret and react to the communicated intention.

2. **Dramatically higher consistency** across seeds (std of 17 vs 129 unique messages): Social pressure acts as a strong regularizer. The discrete-only condition is highly variable -- seed 123 produces 587 unique messages while seed 42 produces only 292. Social mode produces nearly identical distributions (330-369) regardless of seed. The partner creates a stable attractor in message space.

**Interpretation**: Coordination pressure drives agents toward a *shared efficient code* rather than a diverse one. This is consistent with emergent communication literature (Chaabouni et al., ACL 2020): agents under communication pressure develop low-redundancy codes optimized for coordination, not maximum-entropy codes. Social interaction regularizes goals toward coordination-useful concepts, which is a smaller but more stable set than what an unconstrained bottleneck explores.

The key insight is that "good" goal representations for coordination are not the same as "diverse" goal representations. The social mechanism produces goals that are more *meaningful* (longer temporal extent, higher consistency) but less *diverse* (lower entropy, fewer unique messages).

### H3: Transfer to new tasks

**Result**: Inconclusive. Frozen managers from both discrete and social conditions achieved 0% success when transferred to KeyCorridorS4R3 (harder MiniGrid task, 500K steps). The source models' goal representations, while measurably different in diversity metrics, were insufficient for cross-task transfer at the current training scale.

---

## Supplementary: KeyCorridorS3R2 Results

Additional runs on the standard MiniGrid KeyCorridorS3R2 task (without corridor env):

| Condition | Success Rate | Mean Return |
|-----------|-------------|-------------|
| Flat PPO | 1.5% +/- 0.7% | 0.006 +/- 0.002 |
| HRL Continuous | 0.2% +/- 0.1% | 0.001 +/- 0.000 |
| HRL Discrete | 1.0% +/- 1.2% | 0.003 +/- 0.004 |

These confirm that KeyCorridorS3R2 is a hard exploration problem where flat PPO does not easily solve the task (1.5% success at 1M steps), and continuous HRL collapses (0.2%).

---

## Summary

| Hypothesis | Prediction | Result | Verdict |
|-----------|-----------|--------|---------|
| H1: Discrete bottleneck prevents collapse | discrete > continuous on entropy | Discrete 3.99 vs continuous 0.00 | **Supported** |
| H2: Social > discrete on goal quality | social > discrete on entropy/coverage | Social entropy -17%, but +61% temporal extent, 7.6x more consistent | **Not supported on diversity; supported on stability** |
| H3: Social goals transfer better | social -> faster worker learning on new task | Both conditions: 0% transfer success | **Inconclusive** |

## Key Takeaway

The discrete information bottleneck effectively prevents goal collapse (H1). Multi-agent coordination pressure does not increase goal diversity as predicted (H2), but instead produces a *convergence effect*: goals become less diverse but more temporally stable and dramatically more consistent across random seeds. Social interaction acts as a regularizer that selects for *coordination-useful* goals rather than *information-rich* goals. This challenges the assumption that communication pressure naturally produces richer abstractions, suggesting instead that it produces more *efficient* ones.

---

## Reproducibility

- All experiments logged to wandb: `mbzuai-research/social-hrl`
- Plots in `outputs/plots/`
- Code: `scripts/train.py --mode {flat,continuous,discrete,social} [--corridor]`
- Config: `configs/default.yaml`
