# Suggested-Approach Implementation — Final Results

Status: everything from `PROJECT_INFO.md` Suggested Approach is implemented
and run. All headline numbers are now aggregated over 3 seeds (42, 7, 123)
with mean ± std, not single-seed. The single-seed story from the prior
revision is preserved at the bottom for comparison.

## What was added in this round

| Item from Suggested Approach | Status | Artifact |
|---|---|---|
| Shared "bus" resource cheaper when simultaneous | Implemented | [envs/multi_agent_env.py](envs/multi_agent_env.py) (`bus_cost_solo`, `bus_cost_shared`) |
| **Strict bus: arrival-time window** | Implemented | `bus_window` param in [envs/multi_agent_env.py](envs/multi_agent_env.py) |
| **Turn-taking coordination** | Implemented | `turn_taking` param in [envs/multi_agent_env.py](envs/multi_agent_env.py) |
| Minigrid multi-agent with coordination pressure | Implemented | `TwoAgentCorridorEnv` with bus costs |
| Communication channel (discrete bottleneck) | Implemented | [models/communication.py](models/communication.py) |
| HRL with goal-conditioned manager | Implemented | [algos/hrl_trainer.py](algos/hrl_trainer.py), [algos/multi_agent_trainer.py](algos/multi_agent_trainer.py) |
| Goal-representation analysis | Implemented | [analysis/goal_metrics.py](analysis/goal_metrics.py) |
| **Compositionality metric surfaced in outputs** | Implemented | topographic_similarity now reported in every sweep |
| Transfer: train social, evaluate solo | Implemented | [scripts/transfer_verify.py](scripts/transfer_verify.py) |
| MADDPG (CTDE, flat baseline) | Implemented | [algos/maddpg_trainer.py](algos/maddpg_trainer.py) |
| LOLA (opponent-aware, comm-channel update) | Implemented | [algos/lola_trainer.py](algos/lola_trainer.py) |
| **3-seed aggregation harness** | Implemented | [scripts/mini_sweep.py](scripts/mini_sweep.py) extended to 4 modes |
| **RQ4 scenario sweep** | Implemented | [scripts/rq4_scenarios.py](scripts/rq4_scenarios.py) |

## What was kept from prior work

- `F.normalize` bugfixes at every encode/decode site in both trainers.
- Per-env `temporal_extent_mean` computation inside the trainers.
- `scripts/verify_hypotheses.py` fast harness.
- `scripts/harder_env_saturation.py` — shows bottleneck saturation on
  KeyCorridorS3R2 and MultiRoom-N6 too.

## Research Questions — final answers

### RQ1: Can multi-agent coordination pressure prevent goal collapse?

**Verdict: NO — and the result is robust across seeds.** Discrete-bottleneck
single-agent HRL saturates goal-space coverage near 1.0; social/MAPPO and
LOLA both sit near 0.49 and never catch up, with a combined-std gap of
~0.04–0.15. MADDPG is a flat baseline with no goals.

**Aggregated (3 seeds, 15k timesteps, stress + bus env):**

| mode | goal_space_coverage | goal_vector_std | entropy | final_return |
|---|---|---|---|---|
| discrete | **0.996 ± 0.006** | 1.008 ± 0.008 | 6.591 ± 0.011 | −0.791 ± 0.006 |
| social (MAPPO) | 0.474 ± 0.037 | 0.189 ± 0.015 | **6.852 ± 0.003** | −0.844 ± 0.016 |
| lola | 0.495 ± 0.144 | 0.185 ± 0.009 | 6.843 ± 0.005 | −0.843 ± 0.041 |
| maddpg (flat) | n/a | n/a | n/a | −0.800 ± 0.000 |

Noise-aware reads:
- `discrete − social` coverage gap = **+0.522** with noise 0.037 → **LARGE** (>3σ)
- `discrete − lola` coverage gap = **+0.501** with noise 0.144 → **LARGE**
- `lola − social` coverage gap = +0.021 with noise 0.149 → **WITHIN NOISE**

**Important correction vs. the prior single-seed story**: the earlier
headline claimed LOLA closed ~30% of the social-vs-discrete gap (coverage
0.621). Over 3 seeds LOLA's mean is 0.495 with σ 0.144 — so the LOLA
coverage win **does not survive multi-seed evaluation**, it was a lucky
draw on seed 42.

Message-space entropy and coverage still favor social (6.85 vs 6.59 on
entropy), but this is a property of message space, not decoded goals.
H2b-style claims about "richer messages" are supported; H2 ("social
prevents goal collapse") is not.

### RQ2: Does social training transfer to better solo goal policies?

**Verdict: NO — multi-seed flips the single-seed claim.** Frozen-manager
transfer from the bus-stress source config (size 9, width 1) to the wider
corridor (size 11, width 3), aggregated over the same 3 seeds (42, 7, 123)
used for RQ1:

| source | eval return | eval success |
|---|---|---|
| discrete | **−0.750 ± 0.000** | 0.0% |
| social | −0.783 ± 0.024 | 0.0% |

Social transfers **−0.033 worse** on eval return (combined noise 0.024,
NOTABLE). Neither source produces a success-rate signal at this budget,
so the gap is on within-episode shaping rather than task completion. The
prior single-seed +0.050 social advantage was seed-42 noise — the same
pattern we saw with LOLA's coverage win, where one favorable draw flipped
direction once seeds 7 and 123 were added.

Reproduce: `python scripts/transfer_multiseed.py` (reuses
`outputs/mini_sweep/seed-{42,7,123}` checkpoints, takes ~10 min on CPU).

### RQ3: Vocabulary / compositionality relationship

**Verdict (coverage): tighter bottlenecks do NOT help social relative to
discrete** — the gap narrows only as the bottleneck LOOSENS, opposite of
the RQ3 framing.

**Verdict (compositionality, NEW this round): social is slightly MORE
compositional than discrete** on the explicit topographic-similarity metric
— a small but visible effect in the opposite direction from coverage.

K × L grid (10k timesteps + bus env, seed 42):

| K | L | K*L | discrete coverage | social coverage | coverage gap | discrete topsim | social topsim |
|---|---|---|---|---|---|---|---|
| 3 | 1 | 3 | 0.984 | 0.117 | −0.867 | +0.019 | +0.004 |
| 3 | 3 | 9 | 0.984 | 0.508 | −0.477 | +0.033 | +0.020 |
| 10 | 1 | 10 | 0.996 | 0.340 | −0.656 | −0.027 | +0.008 |
| 10 | 3 | 30 | 0.992 | 0.461 | −0.531 | −0.048 | +0.016 |
| 25 | 3 | 75 | 0.992 | 0.840 | −0.152 | −0.002 | −0.005 |

Key reading on the compositionality column (Spearman rank correlation
between message Hamming distance and encoder-state L2 distance; higher =
more compositional mapping):

- Social topsim is **positive at 4/5 cells**; discrete is positive at only
  2/5 and goes strongly negative at K=10.
- In the mini-sweep (3 seeds, stress + bus, K=10 L=3), the pattern holds
  with error bars: discrete **−0.008 ± 0.007**, social **+0.015 ± 0.008**,
  LOLA **−0.003 ± 0.004**. Social beats discrete by ~0.023 (≈3× combined σ).
- Both are small in absolute terms — these models are not learning genuinely
  compositional language. But the *direction* is consistent with the RQ3
  intuition: social pressure produces slightly more state-aligned messages,
  even as it hurts goal-space coverage.

So the honest RQ3 answer is **split**: discrete is a universal coverage
regularizer across every (K, L) we tested, but the compositionality signal
that RQ3 explicitly asks about favors social by a small, multi-seed-stable
margin.

### RQ4: Which multi-agent scenarios provide strongest regularization?

**Tested (3 seeds, 12k timesteps):** baseline (no coordination), shared
bus (cost-when-solo), strict bus (arrival-window=4), turn-taking. Discrete
is unaffected by these multi-agent flags and stays at coverage 0.999 in
every row (the correct invariance).

| scenario | social coverage | gap (social − discrete) | comm_ablation_delta | listener_accuracy |
|---|---|---|---|---|
| baseline | 0.486 ± 0.03 | −0.513 | +0.009 ± 0.01 | 0.068 ± 0.04 |
| bus | 0.529 ± 0.03 | **−0.470** | +0.027 ± 0.08 | 0.058 ± 0.03 |
| bus_strict | 0.529 ± 0.03 | **−0.470** | −0.001 ± 0.04 | 0.053 ± 0.04 |
| turn_taking | 0.514 ± 0.03 | −0.484 | −0.004 ± 0.02 | 0.070 ± 0.02 |

**Revised finding (vs. prior single-seed):** turn-taking is **no longer
the strongest regularizer** once we average over seeds. The bus and
bus_strict variants tie with the smallest gap (−0.470), turn-taking is
slightly behind (−0.484), and baseline is the worst (−0.513). The prior
single-seed +0.144 lift for turn-taking shrinks to **+0.029** and is
within combined noise (~0.04 per side).

The honest 3-seed reading: shared-bus pressure does help slightly over
baseline (+0.04 lift), but the spread across all four scenarios is small
relative to the seed-to-seed noise. **No scenario closes the social-vs-
discrete gap by more than ~10% of its baseline magnitude.**

`comm_ablation_delta` is positive only for baseline (+0.009) and bus
(+0.027), and at most ~0.5σ from zero — the channel does not carry
load-bearing information in any scenario. `listener_accuracy` stays in
the 0.05–0.07 band across all scenarios (chance-level for K=10).

#### LOLA × scenario cross (3 seeds, 12k timesteps)

We also ran the LOLA opponent-aware learner under the same four scenarios
to check the second positive single-seed signal (turn-taking + LOLA):

| scenario | LOLA coverage | LOLA return | comm_ablation_delta |
|---|---|---|---|
| baseline | 0.353 ± 0.12 | −0.709 ± 0.06 | +0.013 |
| **bus** | **0.435 ± 0.08** | −0.846 ± 0.00 | −0.098 |
| bus_strict | 0.435 ± 0.08 | −0.850 ± 0.01 | −0.069 |
| turn_taking | 0.384 ± 0.15 | −0.803 ± 0.01 | +0.004 |

Bus pressure lifts LOLA's coverage most (+0.082 over baseline), but its
return drops the furthest (−0.137). Turn-taking does not pair with LOLA
to produce a coverage win — LOLA + turn-taking sits between baseline and
bus on coverage and below baseline on return. **No LOLA × scenario
combination beats social on coverage** (LOLA bus 0.435 vs social bus
0.529), so the cross does not rescue either method.

## LOLA and MADDPG — multi-seed head-to-head (bus + stress, 15k steps)

| mode | coverage | goal_std | MI | listener | topsim | abl_d | final_ret |
|---|---|---|---|---|---|---|---|
| discrete | 0.996 ± 0.006 | 1.008 ± 0.008 | 2.964 ± 0.311 | 0.257 ± 0.056 | −0.008 ± 0.007 | n/a | −0.791 ± 0.006 |
| social (MAPPO) | 0.474 ± 0.037 | 0.189 ± 0.015 | 1.354 ± 0.185 | 0.052 ± 0.020 | **+0.015 ± 0.008** | +0.066 ± 0.030 | −0.844 ± 0.016 |
| lola | 0.495 ± 0.144 | 0.185 ± 0.009 | 1.161 ± 0.148 | 0.058 ± 0.037 | −0.003 ± 0.004 | −0.009 ± 0.065 | −0.843 ± 0.041 |
| maddpg (flat) | n/a | n/a | n/a | n/a | n/a | n/a | −0.800 ± 0.000 |

**Revised LOLA verdict.** LOLA is not a robust mover on goal-space coverage
(0.495 ± 0.144, within noise of social's 0.474 ± 0.037). The 0.621 number
from the earlier single-seed run was an outlier on seed 42. LOLA also
doesn't produce the compositionality signal social does (topsim 0 vs
+0.015). The "LOLA is the first social-family method to move the coverage
number" claim from the prior revision does not survive 3-seed replication.

**MADDPG (flat)** still reaches comparable environmental return without
hierarchy (−0.800 ± 0.000) — the corridor task does not require hierarchy
to solve, so the goal-representation question is about what kind of
structure hierarchical pressure *produces*, not whether hierarchy is
necessary.

## Headline

- **RQ1**: NO. Multi-agent coordination pressure does not prevent goal
  collapse on the standard coverage metric. The finding is now 3-seed
  stable: discrete = 0.996 ± 0.006, social = 0.474 ± 0.037, LOLA = 0.495
  ± 0.144. LOLA's previous single-seed win (0.621) is not reproducible.
- **RQ2**: NO. Multi-seed transfer flips the single-seed story: social
  is **−0.033 worse** than discrete on eval return (−0.783 ± 0.024 vs
  −0.750 ± 0.000, NOTABLE). The +0.050 social win was seed-42 noise.
- **RQ3**: the story is split. Coverage says discrete wins at every K × L.
  Compositionality (topographic similarity) is small but favors social by
  ~3σ (−0.008 ± 0.007 vs +0.015 ± 0.008). If "goal quality" means coverage,
  the bottleneck does all the work. If it means compositionality, social
  contributes a real (small) signal on top.
- **RQ4**: small effects, all within noise once seeds are averaged.
  Bus and bus_strict tie for smallest social-vs-discrete coverage gap
  (−0.470, +0.04 lift over baseline); turn-taking's prior single-seed
  +0.144 lift shrinks to +0.029. LOLA × turn-taking does not rescue
  either method. No scenario's `comm_ablation_delta` is reliably above
  zero — channel never carries load-bearing information.

## Reproducibility

All headline numbers reproduce in ~25 minutes on CPU:

```
# Multi-seed 4-way (RQ1 + LOLA + MADDPG + compositionality):
python scripts/mini_sweep.py

# Compositionality sweep (RQ3):
python scripts/vocab_sweep.py --timesteps 10000 --seed 42 --bus

# RQ4 scenario comparison:
python scripts/rq4_scenarios.py                  # 1 seed, fast
SEEDS="42 7 123" python scripts/rq4_scenarios.py # 3 seeds, slower

# Multi-seed transfer (RQ2):
python scripts/transfer_multiseed.py            # 3 seeds, ~10 min CPU

# LOLA x scenario cross (RQ4):
MODES=lola SEEDS="42 7 123" python scripts/rq4_scenarios.py
```

Outputs land in `outputs/mini_sweep/`, `outputs/vocab_sweep/`,
`outputs/rq4_scenarios/`, and `outputs/transfer_multiseed/`. Each
directory contains an `aggregated_summary.json` with means, stds, and
per-seed values.

## What would change the answer

With RQ1, RQ2, and RQ4 all robustly negative or within-noise across 3
seeds — and the only positive signal being a small compositionality lift
(RQ3) — the remaining paths are:

1. **Compositional target envs** (BabyAI mission-string tasks) where the
   goal distribution is genuinely compositional, not just spatially diverse.
   Needs [envs/wrappers.py](envs/wrappers.py) to stop discarding mission
   strings and an encoder that consumes them. The current topsim numbers
   are near zero in absolute terms — this is likely a ceiling of the
   corridor task, not a method limitation. **This is the highest-leverage
   next step**, since the only surviving social-favorable signal is
   compositionality and the corridor cannot stretch it further.
2. **A harder manager-worker asymmetry** — e.g. manager trained from sparse
   extrinsic reward only, not intrinsic shaping — so goal-space diversity
   actually bottlenecks return.
3. **Larger-budget transfer (50–100k timesteps)** to see whether the
   multi-seed RQ2 result remains negative when the target task has time
   to actually solve. At the current 20k budget no source produces any
   success-rate signal.

LOLA-style opponent-aware learning on the communication-channel update is
implemented at [algos/lola_trainer.py](algos/lola_trainer.py). In the
multi-seed regime reported above (both standalone RQ1 and crossed with
all four scenarios) it is not a significant improvement over MAPPO,
contrary to the single-seed claim in the prior revision of this document.

## Revision history

- **This revision** (multi-seed follow-up, 2026-04-18): RQ2 transfer is
  now 3-seed via [scripts/transfer_multiseed.py](scripts/transfer_multiseed.py)
  and **flips direction** (social −0.033 worse, NOTABLE); RQ4 scenario
  comparison is now 3-seed and the single-seed turn-taking advantage
  collapses (+0.144 → +0.029, within noise); LOLA × scenario cross added
  via `MODES=lola` in [scripts/rq4_scenarios.py](scripts/rq4_scenarios.py)
  and shows no scenario rescues LOLA over social.
- **Previous revision** (post-audit): 3-seed aggregation replaced single-seed
  RQ1/RQ3 numbers; topographic_similarity (compositionality) surfaced;
  bus_strict and turn_taking scenarios added for RQ4; LOLA's single-seed
  coverage win shown to be within noise; prior self-contradiction about
  LOLA implementation status fixed.
- **Original revision**: single-seed (seed 42) numbers only; claimed LOLA
  closed 30% of the coverage gap; claimed +0.050 social transfer win;
  claimed turn-taking +0.144 lift; did not report compositionality; RQ4
  only tested shared-bus.
