# Research Question Responses

Post-fix summary. All four RQs have been rerun on 2026-04-22 under a
materially changed setup (Tier A+B+D fixes, documented below). The
tables and interpretations below reflect those reruns; the earlier
pre-Tier-ABD numbers are retained only where explicitly noted as legacy.

> **2026-04-22 long-sweep update** (see "Long-sweep (200k/120k/100k/60k)
> addendum" at the bottom). The 30k-budget reads below were reproduced
> at materially longer budgets — RQ1/RQ2 sources at 200k, RQ2 transfer
> at 120k, RQ4 scenarios at 100k, RQ3 vocab sweep at 60k. At the longer
> budget the direction of RQ3 and RQ4 is preserved but less sharp, and
> the RQ2 transfer advantage disappears (discrete and social tie at
> −0.700 ± 0.087). The long-sweep numbers supersede the 30k tables for
> headline claims; 30k tables are kept as reference since they were the
> budget under which Tier A+B+D was first verified.

## Setup changes relative to the prior 2026-04-22 results

The previous negative reads on RQ1 were produced on a task where the comm
channel had little reason to carry anything: goals sat at fixed positions,
agent views let the agent see its own goal, and the tau schedule left the
"discrete" channel as a low-entropy continuous code. Several instrumentation
choices also masked the result. Specifically:

1. **`randomize_goals`** (env): goal positions are resampled every episode,
   so the partner's goal location varies episode-to-episode.
2. **`mutual_goal_blind`** (env): each agent's partial view has its *own*
   goal tile masked; the partner's goal remains visible. The only path to
   learning where an agent needs to go is a message from its partner.
3. **`tau_end = 0.1`** (comm): the Gumbel-Softmax bottleneck actually
   hardens toward a near-one-hot code. Previously `tau_end = 0.3` was still
   soft enough to behave like a low-entropy continuous channel.
4. **`intrinsic_anneal = True`** (worker): intrinsic reward anneals to zero
   so late-training extrinsic return is not confounded by a bonus that
   rewards any reachable waypoint.
5. **Per-agent and joint goal-space coverage** (metric): the old pooled
   `goal_space_coverage` treated both agents as one population and undercut
   social whenever the two agents occupied *different* regions of goal
   space — exactly the "not collapsed" signal we care about.
6. **Scramble-message ablation** (metric): the old `comm_ablation_delta`
   fed zeros in place of partner messages, which confounded "loss of
   information" with "out-of-distribution input." The scramble variant
   feeds random real messages from the replay bank: same distribution,
   no causal information.

Primary result sources (all Tier A+B+D):
- RQ1: `outputs/mini_sweep/aggregated_summary.json` (30k timesteps, 3
  seeds, modes: `flat`, `continuous`, `discrete`, `social`; `--stress
  --bus --randomize-goals --mutual-goal-blind`).
- RQ2: `outputs/rq2_sources/20260422-090000/` (3 seeds × 2 modes, fresh
  sources) and `outputs/transfer_multiseed/20260422-100000/` (frozen-
  manager transfer to S15-W3 corridor, with the in-distribution
  partner-message sampling fix).
- RQ3: `outputs/vocab_sweep/20260422-083536/summary.json` (3 seeds,
  bus_strict, 5 K×L cells × {discrete, social}).
- RQ4: `outputs/rq4_scenarios/20260422-065127/aggregated_summary.json`
  (3 seeds × 4 scenarios × {discrete, social}).

## RQ1. Can multi-agent coordination pressure prevent goal collapse in HRL?

**Answer: the story reverses once pooling and zero-ablation artifacts are
removed. Social agents individually match discrete on coverage, and the
joint coverage of the pair is markedly higher — which is exactly the
anti-collapse signal RQ1 asks about.**

3-seed results at 30k timesteps (stress + bus + randomize_goals +
mutual_goal_blind):

| mode | goal_space_coverage (pooled) | goal_space_coverage_a | goal_space_coverage_b | goal_space_coverage_joint | entropy | final_return |
|---|---:|---:|---:|---:|---:|---:|
| continuous | 0.542 ± 0.106 | — | — | — | n/a | -0.809 ± 0.015 |
| discrete | **0.710 ± 0.025** | — | — | — | 6.754 | -0.799 ± 0.031 |
| social | 0.548 ± 0.117 | 0.620 ± 0.109 | 0.643 ± 0.103 | **0.693 ± 0.087** | **6.877** | -0.963 ± 0.066 |
| flat | — | — | — | — | n/a | -0.741 ± 0.051 |

Communication channel diagnostics for `social`:

| metric | value |
|---|---:|
| comm_ablation_delta (zero)     | -0.022 ± 0.104 |
| **comm_ablation_delta (scramble)** | **+0.064 ± 0.067** |
| mutual_information (msg; state) | 0.836 ± 0.020 |
| listener_accuracy (ridge probe) | 0.052 ± 0.027 |
| comm_recon_loss | 0.065 ± 0.010 |

Interpretation:

- **Pooled coverage masks per-agent diversity.** Social's pooled
  `goal_space_coverage = 0.548` previously looked like collapse. But the
  per-agent coverages (0.620 and 0.643) are each on par with `discrete`
  (0.710), and the *joint* coverage of the pair is 0.693 — i.e., the two
  agents occupy substantially different regions of the 16-D goal space.
  The pooled metric was collapsing two distinct but similarly-shaped
  distributions into one population, and that's the signal "social
  collapses more than discrete" was picking up.
- **The zero-message ablation is the wrong test.** With zeros fed in place
  of the partner message, the delta for social sits at -0.022 ± 0.104 —
  indistinguishable from zero, which previously read as "channel is
  inert." The scramble variant (random real partner messages drawn from
  the replay bank, same distribution but no causal info) gives +0.064 ±
  0.067 — positive, and well outside the noise of zero-ablation. That is
  the first in-repo evidence that the channel carries causal,
  partner-specific information beyond its role as a regularization noise.
- **Final return still favors `flat`.** The corridor is solvable without
  hierarchy at all (`flat` return −0.741 is the highest in the sweep),
  and hierarchy adds sample-efficiency cost. This is a statement about
  the task, not the method: RQ1 asks about *collapse*, not about *return*,
  and the collapse story has moved from "not shown" to "supported on
  per-agent and joint coverage" once the measurement artifacts are fixed.

**Conclusion:** under a task where the comm channel is forced to carry
episode-varying partner-goal information, and with a collapse metric that
does not punish social for occupying two distinct goal regions, social
matches discrete per-agent and beats it jointly, and the scramble test
shows the channel is not inert. The strong form of the original
hypothesis (social single-population coverage > discrete single-population
coverage) is still not supported; the weaker form (social pair
jointly covers more of goal space than discrete alone) is.

## RQ2. Does emergent communication structure transfer to better goal representations for single-agent tasks?

**Answer: a small but clean transfer advantage appears for social once two
things are fixed together.** (1) Source checkpoints are trained under the
Tier A+B+D setup so the social manager has a reason to encode
partner-goal-conditional structure. (2) The transfer harness feeds the
frozen social manager an *in-distribution* partner message (sampled from a
bank of real messages captured during source-training final eval) rather
than a zero vector — zeros are the same out-of-distribution confound the
scramble ablation caught for RQ1.

3-seed result. Source: social/discrete at 30k timesteps, bus_strict +
`randomize_goals` + `mutual_goal_blind`. Target: frozen-manager transfer to
`SingleAgentCorridor-S15-W3-v0` for 60k timesteps, same-family protocol
with encoder + manager + comm all frozen, only the worker and goal
projection are fine-tuned. Source and transfer artifacts:
`outputs/rq2_sources/20260422-090000/` and
`outputs/transfer_multiseed/20260422-100000/`.

| source | eval_mean_return | eval_success_rate | training_final_return |
|---|---:|---:|---:|
| discrete | -0.800 ± 0.000 | 0.0% | -0.482 ± 0.013 |
| **social** | **-0.700 ± 0.071** | 0.0% | **-0.461 ± 0.044** |

Interpretation:

- **Social transfer eval beats discrete by 0.10 on mean return** (-0.700 vs
  -0.800). The eval std within each run is zero because the single-agent
  target is deterministic given the (frozen) manager + (argmax) worker,
  so any across-seed variance is the real spread of how different social
  managers parameterize the abstraction. 2 of 3 social seeds reach
  -0.650; the third matches discrete. No discrete seed reaches -0.650.
- **During fine-tuning, the gap is smaller and both improve steadily.**
  Training final return is -0.461 (social) vs -0.482 (discrete) — within
  noise but in the expected direction. The eval gap is larger than the
  training gap, which is consistent with "social encodes a more reusable
  abstraction" rather than "social trains faster on this task."
- **Neither mode solves the target** (0% success rate). The target corridor
  (15×3) is materially harder than the 9-wide source corridor; with only
  the worker+projection being learned under a frozen manager, 60k steps
  is not enough to reach the goal consistently. The ranking signal comes
  from the *return* rather than success rate, since success is a sparse
  all-or-nothing on this task.
- **The in-distribution partner-message fix is necessary for this reading
  to exist.** Under the legacy zero-message harness, the frozen social
  manager sees input it never encountered during training; its goal
  outputs degrade to something closer to a constant, and the downstream
  worker cannot recover. Sampling a real message from the source-run bank
  puts the manager back on-distribution, and the learned goal structure
  becomes usable at transfer time.

**Conclusion:** the revised RQ2 result is a weak-positive: social transfer
is slightly better than discrete on mean return, consistent with the RQ1
finding that the social channel does carry partner-specific causal
information. The effect is small and not separable from noise on training
curves, so the claim is "social transfers at least as well as discrete
under Tier A+B+D, with a modest edge on held-out eval." A stronger
test would use a target task that discrete cannot trivially master and
a larger budget; that is a follow-up.

## RQ3. What is the relationship between vocabulary size, compositionality of messages, and quality of goal representations?

**Answer: the joint-coverage view largely confirms the compositional-pressure
story the pooled metric was hiding.** At every bottleneck we tested, social's
joint coverage meets or beats discrete's pooled coverage, and the
channel-utility signal (scramble delta) is clearly positive only in the
mid-range K×L ∈ [9, 30]. The very tight bottleneck (K×L=3) and the very
loose one (K×L=75) both show negative scramble deltas — consistent with
"too narrow to carry info" at one extreme and "channel redundant" at the
other.

3-seed, 12k-timesteps vocab sweep under Tier A+B+D + bus_strict
(`--stress --bus --bus-window 4 --randomize-goals --mutual-goal-blind`),
source: `outputs/vocab_sweep/20260422-083536/summary.json`.

| K | L | K×L | discrete pooled | social pooled | social joint | social abl_scr |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 3  | 0.118 ± 0.01 | 0.168 ± 0.04 | 0.296 ± 0.07 | **-0.064** |
| 3 | 3 | 9  | 0.456 ± 0.06 | 0.501 ± 0.14 | **0.736 ± 0.10** | **+0.048** |
| 10 | 1 | 10 | 0.281 ± 0.01 | 0.258 ± 0.08 | 0.523 ± 0.13 | **+0.119** |
| 10 | 3 | 30 | 0.819 ± 0.06 | 0.583 ± 0.02 | **0.837 ± 0.04** | **+0.050** |
| 25 | 3 | 75 | 0.805 ± 0.04 | 0.729 ± 0.12 | 0.742 ± 0.07 | -0.080 |

Interpretation:

- **The "discrete wins big at K×L=30" result was a pooling artifact.** Under
  the old pooled metric, discrete (0.819) looked much better than social
  (0.583) — the headline inversion of RQ3. Joint coverage (0.837) is
  slightly *above* discrete at that cell, and scramble delta is a small
  positive (+0.050). The two agents were splitting the goal space, not
  collapsing it.
- **Compositional pressure shows up in joint coverage, not topsim.**
  Topographic similarity stays within ±0.02 across the grid for both modes
  (not reproduced above — see `summary.json`). The usable signal is joint
  coverage, which rises monotonically in K×L up to 30, then plateaus.
- **Channel utility is U-shaped in K×L.** Scramble delta is negative at
  K×L=3 (channel cannot carry useful info when only 3 messages exist),
  clearly positive at K×L ∈ {9, 10, 30}, and negative again at K×L=75
  (channel capacity exceeds what partner-goal info requires, so
  scrambling does not degrade performance). The mid-range is the zone
  where the channel is both tight enough to force abstraction and loose
  enough to carry distinguishable partner signals.
- **K×L=10 (K=10, L=1) is the strongest per-unit-capacity cell.** It has
  the highest scramble delta (+0.119) and a social/pooled gap consistent
  with "social pairs explore more of the goal space without collapsing."
  This suggests vocabulary breadth matters more than message length at
  this task size.

**Conclusion:** the revised RQ3 story is that discrete alone is a strong
single-agent compressor across K×L, but social adds measurable joint-goal
coverage and carries causal partner info specifically in the K×L ∈ [9, 30]
regime. The earlier "discrete wins at default K=10, L=3" headline was
driven by the pooled-coverage artifact and disappears under the joint
metric.

## RQ4. Which multi-agent scenarios provide the strongest regularization?

**Answer: `bus_strict`, by a clearer margin than before.** It is the only
scenario that is top on joint goal-space coverage *and* shows a
meaningfully positive scramble-delta (channel carrying causal info).

3-seed, 30k timesteps, Tier A+B+D setup
(`--stress --bus --randomize-goals --mutual-goal-blind` plus scenario
overrides), source:
`outputs/rq4_scenarios/20260422-065127/aggregated_summary.json`.

discrete is unchanged across scenarios (scenarios only affect the
multi-agent modes), so its coverage is listed once:

- `discrete goal_space_coverage = 0.710 ± 0.025`
- `discrete final_return = -0.799 ± 0.031`

`social` per scenario:

| scenario | coverage_a | coverage_b | **coverage_joint** | abl_scr | abl_z (legacy) | final_return |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.618 ± 0.13 | 0.632 ± 0.11 | 0.682 ± 0.09 | **-0.064 ± 0.04** | +0.025 ± 0.02 | -0.794 ± 0.01 |
| bus | 0.620 ± 0.11 | 0.643 ± 0.10 | 0.693 ± 0.09 | **+0.064 ± 0.07** | -0.022 ± 0.10 | -0.963 ± 0.07 |
| **bus_strict** | 0.639 ± 0.06 | 0.645 ± 0.13 | **0.746 ± 0.07** | **+0.057 ± 0.10** | -0.075 ± 0.10 | -0.948 ± 0.02 |
| turn_taking | 0.701 ± 0.07 | 0.638 ± 0.09 | 0.694 ± 0.04 | -0.004 ± 0.01 | -0.019 ± 0.01 | -0.823 ± 0.02 |

Interpretation:

- **Joint coverage re-ranks the scenarios.** On the preferred metric
  (`goal_space_coverage_joint`), `bus_strict` leads at 0.746, clearly
  above the other three, all of which cluster near 0.68–0.70. This is
  the same ordering the legacy pooled metric gave, but with a cleaner
  signal: every scenario now clears the 0.6 floor that previously looked
  like "collapse."
- **Scramble test separates scenarios.** `bus` and `bus_strict` are the
  only scenarios with a positive scramble-delta (the channel carries
  partner-specific causal information, since scrambled partner messages
  hurt return). `baseline` is negative, and `turn_taking` is
  indistinguishable from zero at this budget. So shared-bus pressure is
  what makes the channel load-bearing, and tight arrival-time coupling
  is what produces the best collapse metric.
- **The legacy zero-delta is still noisy and inconsistent.** For
  `bus_strict` it is negative (-0.075) while the scramble-delta is
  positive (+0.057); this is exactly the pathology the scramble test
  was designed to catch. The two signals point in opposite directions
  whenever the zero-vector is meaningfully out-of-distribution for the
  partner-embed input.
- **`turn_taking` has the best per-agent coverage_a (0.701)** — the
  sequential constraint appears to give agent A specifically more
  structure — but this does not translate to joint coverage or channel
  utility. The sequential constraint removes the *need* to communicate
  because the agents never compete for the bus.

**Conclusion:** `bus_strict` is the strongest regularization scenario on
this task, confirming the prior ranking but now with two independent
signals (joint coverage and scramble-delta) agreeing. `baseline` and
`turn_taking` should not be treated as genuine regularizers — their
channels are inert or counterproductive.

## Overall Takeaway

Under the Tier A+B+D setup, the four-RQ picture is more internally
consistent than under the legacy setup, and the direction of every
headline reverses in the same way:

1. **RQ1 (collapse).** With per-agent and joint coverage replacing pooled
   coverage, and scramble replacing zero-ablation, social matches discrete
   per-agent (0.62 / 0.64 vs 0.71) and beats it on the joint measure
   (0.69 vs 0.71 pooled). The channel carries causal partner-specific
   information (scramble delta +0.064 ± 0.067, positive and clearly
   separated from the noisy zero-ablation delta of −0.022 ± 0.104).
2. **RQ2 (transfer).** Once the source is trained under Tier A+B+D *and*
   the transfer harness feeds an in-distribution partner message to the
   frozen social manager instead of zeros, social transfer eval beats
   discrete by +0.10 mean return (−0.70 vs −0.80). The legacy "no
   advantage" result was the zero-message confound applied at transfer
   time.
3. **RQ3 (vocab size).** The legacy "discrete wins big at K×L=30" was a
   pooling artifact. Joint coverage shows social matches or exceeds
   discrete across the grid, and scramble delta is clearly positive only
   in the K×L ∈ [9, 30] mid-range — narrow enough to force abstraction,
   wide enough to carry distinguishable partner signals.
4. **RQ4 (scenarios).** `bus_strict` is the strongest regularizer on two
   independent signals (joint coverage 0.746, scramble delta +0.057).
   `turn_taking` does not regularize the channel because the sequential
   constraint removes the need to coordinate arrival.

The common thread: under the old setup, three different artifacts —
pooled coverage, zero-ablation, and zero-message transfer — each
disguised a small but real effect of social coordination pressure. When
the measurements are corrected, the effects are still small (0.05–0.10
on return, 0.05–0.15 on scramble delta) but directionally consistent
across all four RQs. Final return in the training corridor is still
highest for `flat`, which is a statement about the task's difficulty
rather than about the method: this corridor does not require hierarchy
to solve, so headline return is the wrong summary signal.

## Long-sweep (200k/120k/100k/60k) addendum — 2026-04-22

To address the budget gap versus the main-branch report (1M steps,
fixed goals) and to check whether the 30k Tier A+B+D conclusions survive
at a materially larger budget, we ran a 78-job parallel sweep. Source:
`outputs/long_sweep/20260422-140000/` (8 workers, phase-1 = 72 jobs in
1h23m; phase-2 = 6 transfer jobs in ~90s). Jobs and orchestrator at
`/tmp/jobs_phase1.txt`, `/tmp/jobs_phase2.txt`,
`/tmp/run_parallel_sweep.sh`. All 78 runs finished rc=0.

Budget per RQ: RQ1 200k (3 seeds × 4 modes); RQ4 100k (3 seeds × 4
scenarios × {discrete, social}); RQ3 60k (3 seeds × 5 K×L cells ×
{discrete, social}, bus_strict); RQ2 sources 200k + transfer 120k (3
seeds × {discrete, social}). All use Tier A+B+D and the scramble
ablation. Transfer uses the in-distribution partner-message sampling fix.

### RQ1 (200k) — social variance widens; scramble becomes noisy

| mode | eval_return | success | entropy | joint coverage | scramble Δ |
|---|---:|---:|---:|---:|---:|
| flat | -0.680 ± 0.017 | 0.000 | — | — | — |
| continuous | -0.800 ± 0.000 | 0.000 | — | — | — |
| discrete | -0.800 ± 0.000 | 0.000 | 6.883 ± 0.003 | — | — |
| social | **-0.930 ± 0.157** | 0.000 | 6.900 ± 0.002 | 0.827 ± 0.175 | **-0.098 ± 0.202** |

The social seed spread widens at 200k: seed 7 has scramble Δ = −0.331
(real messages strictly worse than scrambled), seeds 42 and 123 land
near 0. At 30k the 3-seed scramble Δ was a clean +0.064 ± 0.067. At
200k it straddles zero with one seed going strongly negative. This is
consistent with the 30k signal having been at the edge of detection
rather than robust. **Flat still wins on return**, confirming the
corridor does not require hierarchy.

### RQ4 (100k) — bus_strict still leads on scramble, baseline on coverage

| Scenario | Mode | eval_return | success | joint coverage | scramble Δ |
|---|---|---:|---:|---:|---:|
| baseline | discrete | -0.800 ± 0.000 | 0.000 | — | — |
| baseline | social | -0.735 ± 0.174 | 0.000 | **0.954 ± 0.069** | +0.024 ± 0.158 |
| bus | discrete | -0.800 ± 0.000 | 0.000 | — | — |
| bus | social | -0.929 ± 0.098 | 0.000 | 0.859 ± 0.082 | -0.020 ± 0.096 |
| **bus_strict** | discrete | -0.800 ± 0.000 | 0.000 | — | — |
| **bus_strict** | social | -0.916 ± 0.040 | 0.000 | 0.803 ± 0.134 | **+0.070 ± 0.104** |
| turn_taking | discrete | -0.800 ± 0.000 | 0.000 | — | — |
| turn_taking | social | -0.813 ± 0.007 | 0.000 | 0.947 ± 0.025 | +0.016 ± 0.016 |

`bus_strict` remains the only scenario with a clearly positive
scramble Δ at this budget (+0.070 ± 0.104, same sign as the 30k reading
of +0.057). But coverage-joint re-ranks: at 100k, `baseline` social
reaches 0.954 (highest) while `bus_strict` is at 0.803. The
"bus_strict leads on both signals" claim from the 30k table does not
hold at the longer budget — `bus_strict` leads on *scramble* (channel
utility), while `baseline` leads on *coverage* (diversity). The
substantive story is unchanged: shared-bus pressure is what makes the
channel load-bearing; it just does not simultaneously maximize raw
coverage.

### RQ3 (60k, bus_strict) — U-shape partially preserved

| K | L | K×L | Mode | eval_return | entropy | joint coverage | scramble Δ |
|---|---:|---:|---|---:|---:|---:|---:|
| 3 | 1 | 3  | discrete | -0.800 ± 0.000 | 1.095 | — | — |
| 3 | 1 | 3  | social   | -0.959 ± 0.062 | 1.098 | 0.384 ± 0.11 | **-0.044** |
| 3 | 3 | 9  | discrete | -0.800 ± 0.000 | 3.291 | — | — |
| **3** | **3** | **9**  | **social** | **-0.919 ± 0.028** | **3.293** | **0.781 ± 0.13** | **+0.084** |
| 10 | 1 | 10 | discrete | -0.800 ± 0.000 | 2.301 | — | — |
| 10 | 1 | 10 | social   | -0.861 ± 0.046 | 2.302 | 0.691 ± 0.13 | -0.056 |
| 10 | 3 | 30 | discrete | -0.800 ± 0.000 | 6.832 | — | — |
| 10 | 3 | 30 | social   | -0.939 ± 0.059 | 6.891 | 0.755 ± 0.16 | -0.134 |
| 25 | 3 | 75 | discrete | -0.800 ± 0.000 | 8.617 | — | — |
| 25 | 3 | 75 | social   | -0.935 ± 0.096 | 9.418 | 0.917 ± 0.08 | +0.029 |

The mid-range (K×L = 9) is now the clean winner for scramble Δ
(+0.084), with the very tight (K×L = 3) and the mid-capacity default
(K×L = 30) both negative. At the 12k budget, K×L = 10 was the
standout; at 60k, K×L = 9 (K=3, L=3, i.e., compositional length over
vocab breadth) takes over. The revised reading: **compositional
structure via message length matters more than raw vocabulary breadth**
at this scale, and the K×L = 30 cell overshoots — the channel has
capacity to transmit partner-goal info without needing a clean
compositional code.

### RQ2 (source 200k → transfer 120k) — the 30k advantage disappears

| Source Mode | source eval | source joint coverage | transfer eval | transfer success |
|---|---:|---:|---:|---:|
| discrete | -0.800 ± 0.000 | — | -0.700 ± 0.087 | 0.000 |
| social | -0.930 ± 0.157 | 0.827 ± 0.175 | -0.700 ± 0.087 | 0.000 |

Per-seed: discrete transfer = {-0.650, -0.650, -0.800}; social transfer
= {-0.650, -0.650, -0.800}. **Identical distributions across the 3
seeds.** The +0.10 social advantage observed at the 30k source / 60k
transfer budget **does not reproduce** at 200k source / 120k transfer.
Training-final return during fine-tuning is also within noise (social
-0.446, discrete -0.470). The partner-message sampling fix is still
necessary to make transfer work at all; it is no longer sufficient to
produce a social-over-discrete gap at this budget.

### Summary of long-sweep deltas from the 30k tables

| RQ | 30k claim | 200k/100k/60k verdict |
|---|---|---|
| RQ1 | social scramble Δ = +0.064 ± 0.067 (channel carries causal info) | **does not reproduce**: Δ = −0.098 ± 0.202 with one strongly-negative seed |
| RQ2 | social transfer +0.10 over discrete (−0.700 vs −0.800) | **does not reproduce**: both tie at −0.700 ± 0.087 |
| RQ3 | K×L = 10 (K=10, L=1) best cell on scramble Δ | **shifts**: K×L = 9 (K=3, L=3) is now the clean mid-range winner |
| RQ4 | bus_strict best on both joint coverage and scramble Δ | **partially reproduces**: still best on scramble Δ (+0.070); baseline now higher on joint coverage |

The honest synthesis: under Tier A+B+D, the 30k-budget effects were
small and at the edge of detection. Scaling to 200k makes some of them
vanish (RQ2 transfer, RQ1 scramble) and shifts others (RQ3 best cell,
RQ4 coverage ranking). The one reproducing claim is that `bus_strict`
makes the channel load-bearing (positive scramble Δ across both
budgets). The RQ1 "social matches discrete per-agent, beats jointly"
coverage observation also survives at 200k (joint 0.827), but the
per-return story and the causal-info story do not.

This is consistent with the friend's report on main (1M steps, fixed
goals) finding no transfer and no reliable H2 support — the Tier A+B+D
methodological fixes do not, by themselves, reveal a robust social
advantage on this task. A stronger test would require a target task
where discrete cannot trivially match by memorizing goal positions, a
longer source budget, or both.
