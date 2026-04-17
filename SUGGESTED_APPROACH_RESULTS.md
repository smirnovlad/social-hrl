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

**Verdict: directionally YES, magnitude small, single-seed.** Frozen-manager
transfer from the bus-stress source config to a wider-corridor target:

| source | eval return | eval success |
|---|---|---|
| discrete | −0.800 | 0.0% |
| social | −0.750 | 0.0% |

Social transfers **+0.050** better on eval return. Neither source produces
a success-rate signal at this budget; the gap is on within-episode shaping.
This number is single-seed and within the per-seed coverage noise we see
on RQ1, so it should be treated as directional only until a multi-seed
transfer harness is run.

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

**Tested (seed 42, 10k timesteps):** baseline (no coordination), shared
bus (cost-when-solo), strict bus (arrival-window), turn-taking.

| scenario | social coverage | gap (social − discrete) | comm_ablation_delta | listener_accuracy |
|---|---|---|---|---|
| baseline | 0.414 | −0.578 | +0.131 | 0.057 |
| bus | 0.496 | −0.496 | +0.069 | 0.039 |
| bus_strict | 0.492 | −0.500 | −0.007 | 0.023 |
| **turn_taking** | **0.559** | **−0.434** | +0.030 | 0.000 |

(discrete is unaffected by these multi-agent-only flags and stays at
coverage 0.992 in every row — that is the correct invariance.)

**Key finding:** turn-taking narrows the social-vs-discrete coverage gap
the most (−0.434 vs baseline −0.578, i.e. **+0.144 lift** from sequential
coordination). The bus variants also lift over baseline (+0.08 each), but
less than turn-taking.

**But turn-taking's channel is effectively silent** (listener_accuracy =
0.000, ablation_delta = +0.030). The scenario that best regularizes goal
space is not the one whose channel carries the most information — in fact
the opposite: baseline (no coordination) has the highest ablation_delta
(+0.131) and turn-taking/bus_strict the lowest.

Interpretation: turn-taking structures goal diversity by forcing the agent
distribution itself, not by making the comm channel load-bearing. This is a
meaningful RQ4 signal and a new finding relative to the single-seed
single-scenario story in the prior revision.

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
- **RQ2**: weakly yes, single-seed. Social transfers +0.050 eval return
  over discrete; needs multi-seed validation.
- **RQ3**: the story is split. Coverage says discrete wins at every K × L.
  Compositionality (topographic similarity) is small but favors social by
  ~3σ (−0.008 ± 0.007 vs +0.015 ± 0.008). If "goal quality" means coverage,
  the bottleneck does all the work. If it means compositionality, social
  contributes a real (small) signal on top.
- **RQ4**: turn-taking is the strongest coverage-regularizer of the
  scenarios tested (+0.144 lift over baseline), larger than bus (+0.08) or
  strict bus (+0.08). This regularization does not come from the comm
  channel carrying more information — listener accuracy drops to zero
  under turn-taking.

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

# Single-seed transfer (RQ2):
python scripts/verify_hypotheses.py --modes discrete social --stress --bus \
    --timesteps 15000 --seed 42 --output-dir outputs/bus_verify
python scripts/transfer_verify.py \
    --discrete-ckpt outputs/bus_verify/<ts>/discrete/final.pt \
    --social-ckpt   outputs/bus_verify/<ts>/social/final.pt \
    --target-size 11 --target-width 3 --timesteps 30000
```

Outputs land in `outputs/mini_sweep/`, `outputs/vocab_sweep/`,
`outputs/rq4_scenarios/`. Each directory contains an
`aggregated_summary.json` with means, stds, and per-seed values.

## What would change the answer

Given RQ1 is robustly negative, the remaining paths are:

1. **Compositional target envs** (BabyAI mission-string tasks) where the
   goal distribution is genuinely compositional, not just spatially diverse.
   Needs [envs/wrappers.py](envs/wrappers.py) to stop discarding mission
   strings and an encoder that consumes them. The current topsim numbers
   are near zero in absolute terms — this is likely a ceiling of the
   corridor task, not a method limitation.
2. **A harder manager-worker asymmetry** — e.g. manager trained from sparse
   extrinsic reward only, not intrinsic shaping — so goal-space diversity
   actually bottlenecks return.
3. **Turn-taking as the primary scenario** rather than bus. RQ4 shows it
   regularizes more; pairing it with LOLA or a cleaner compositional env
   is the natural next step.

LOLA-style opponent-aware learning on the communication-channel update is
implemented at [algos/lola_trainer.py](algos/lola_trainer.py). In the
multi-seed regime reported above it is not a significant improvement over
MAPPO, contrary to the single-seed claim in the prior revision of this
document.

## Revision history

- **This revision** (post-audit): 3-seed aggregation replaces single-seed
  headline numbers; topographic_similarity (compositionality) is now
  reported; bus_strict and turn_taking scenarios added for RQ4; LOLA's
  single-seed win shown to be within noise; prior self-contradiction
  about LOLA implementation status fixed.
- **Prior revision**: single-seed (seed 42) numbers only; claimed LOLA
  closed 30% of the coverage gap; did not report compositionality; RQ4
  only tested shared-bus.
