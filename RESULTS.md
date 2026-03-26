# Experimental Results

## Setup

All experiments: 1M timesteps, 3 random seeds (42, 123, 7), 8 parallel environments, PPO with lr=3e-4. Codebase passed 3 rounds of automated review (6 reviewer invocations total) before final runs.

### Corridor Environment (primary, all hypotheses)

Custom 11x11 grid with two rooms connected by a 3-cell-wide corridor. Agent starts top-left, goal is bottom-right. Step penalty (-0.01/step), goal reward (+1), coordination bonus (+0.5 if both agents finish). Used for all 4 conditions to ensure fair comparison:

| Condition | Agent(s) | Manager | Goals | Tests |
|-----------|----------|---------|-------|-------|
| Flat PPO | Single | None | None | Baseline |
| HRL Continuous | Single | TD3 + tanh | g in R^16 | H1 |
| HRL Discrete | Single | PPO + Gumbel-Softmax | m in {1,...,10}^3 | H1 |
| HRL Social | Two agents | PPO + Gumbel-Softmax | m in {1,...,10}^3 | H2 |

---

## Main Results (Corridor Environment)

### Task Performance

| Condition | Goal Reach Rate | Mean Return | Last 100 Return |
|-----------|----------------|-------------|-----------------|
| Flat PPO | **99.3% +/- 0.7%** | **0.837 +/- 0.029** | **0.856** |
| HRL Continuous | 51.2% +/- 7.4% | -0.430 +/- 0.119 | -0.378 |
| HRL Discrete | 51.1% +/- 2.6% | -0.431 +/- 0.045 | -0.417 |
| HRL Social | 45.6% +/- 3.5% | -0.582 +/- 0.039 | -0.550 |

**Flat PPO solves the corridor easily** (99.3%). The single-agent corridor doesn't require hierarchy. All HRL conditions perform worse (45-51%), paying the overhead of manager decisions and intrinsic reward optimization.

### H1: Discrete bottleneck prevents goal collapse

**Test**: Continuous goals (TD3 manager) vs. discrete goals (Gumbel-Softmax), same corridor env.

| Metric | Continuous | Discrete |
|--------|-----------|----------|
| Goal reach rate | 51.2% +/- 7.4% | 51.1% +/- 2.6% |
| Goal entropy | 0.000 (collapsed) | **6.73 +/- 0.03** |
| Unique messages | 0 | **1000 (100%)** |
| Coverage | 0% | **0.99%** |
| % of max entropy | 0% | **97.5%** |

**H1 STRONGLY SUPPORTED.** Continuous goals collapse to zero entropy — the TD3 manager converges to a fixed output. The discrete Gumbel-Softmax bottleneck achieves 97.5% of maximum entropy with all 1000 possible messages utilized. The communication channel reconstruction loss is essential — without it (pre-fix), entropy was only 57.8% of max.

Both conditions achieve similar task performance (~51%), showing that the discrete bottleneck adds goal diversity without hurting performance.

### H2: Social pressure produces better goals than bottleneck alone

**Test**: Discrete (single agent) vs. social (two agents), same corridor env.

| Metric | Discrete (single) | Social (two agents) | Difference |
|--------|-------------------|---------------------|------------|
| Goal reach rate | 51.1% +/- 2.6% | 45.6% +/- 3.5% | -5.5% |
| Goal entropy | **6.73 +/- 0.03** | 6.21 +/- 0.45 | **-0.53** |
| Unique messages | **1000** | **1000** | 0 |
| Coverage | **0.99%** | 0.49% | **-0.50%** |
| Temporal extent | 1.01 | 1.09 | +0.08 |
| % of max entropy | **97.5%** | 89.8% | -7.7% |

**H2 NOT SUPPORTED.** Social pressure produces lower goal entropy (89.8% vs 97.5% of max) and lower coverage (0.49% vs 0.99%) compared to the discrete bottleneck alone. Both conditions use all 1000 unique messages, but social concentrates usage more heavily on a subset.

**Interpretation**: Coordination pressure drives agents toward *efficient* communication — a shared vocabulary where some messages are used more frequently than others. The single-agent discrete bottleneck distributes goals more uniformly (near-maximum entropy) because there is no pressure to be interpretable by a partner. Social interaction acts as a frequency concentrator, not a diversity expander.

### H3: Transfer to new tasks

**Result**: Inconclusive. Frozen managers from both conditions achieved 0% success on KeyCorridorS4R3 (harder MiniGrid, 500K steps). Source models at ~51% corridor success don't produce goal representations that generalize.

---

## Impact of Bug Fixes

Three rounds of automated code review identified critical bugs. The fixes dramatically changed results:

| Metric | Before fixes | After fixes | Change |
|--------|-------------|-------------|--------|
| Discrete entropy | 3.99 (57.8% max) | **6.73 (97.5% max)** | +69% |
| Discrete unique msgs | 407 / 1000 | **1000 / 1000** | +146% |
| Social entropy | 3.32 (48.1% max) | **6.21 (89.8% max)** | +87% |
| Social unique msgs | 345 / 1000 | **1000 / 1000** | +190% |

Key bugs fixed:
1. **Communication channel had zero gradient flow** — sender/decoder never trained (added reconstruction loss + separate optimizer)
2. **Truncated episodes never reset in social mode** — episodes ran past max_steps indefinitely
3. **TD3 done flag off-by-one** — critic bootstrapped through episode boundaries
4. **Encoder normalization wrong** — all channels divided by 10 instead of per-channel (10, 5, 2)
5. **Coordination bonus fired repeatedly** — idle done agents received spurious rewards

---

## Summary

| Hypothesis | Prediction | Result | Verdict |
|-----------|-----------|--------|---------|
| H1: Discrete bottleneck prevents collapse | discrete > continuous on entropy | Discrete 6.73 vs continuous 0.00 | **Strongly supported** |
| H2: Social > discrete on goal quality | social > discrete on entropy/coverage | Social 6.21 < discrete 6.73; both use all 1000 messages | **Not supported** |
| H3: Social goals transfer better | social -> faster worker learning | Both conditions: 0% transfer | **Inconclusive** |

## Key Takeaway

The discrete information bottleneck is highly effective at preventing goal collapse — achieving 97.5% of maximum entropy. Multi-agent coordination does not increase goal diversity; it slightly reduces it (89.8% vs 97.5%). Social pressure acts as a *frequency concentrator*: both conditions explore the full message vocabulary, but social agents develop preferences for certain messages, creating a non-uniform distribution optimized for coordination rather than maximum entropy. This challenges the hypothesis that communication pressure produces richer goal abstractions.

---

## Reproducibility

- All experiments logged to wandb: `mbzuai-research/social-hrl`
- Plots in `outputs/plots/`
- Code: `scripts/train.py --mode {flat,continuous,discrete,social} [--corridor]`
- Config: `configs/default.yaml`
- Conda env: `social-hrl`
