# Next Steps — ML8103 Final Report & Presentation

Working doc. Produced after the 15k validation pass and the fix round
(scipy/sklearn install, MADDPG reproducibility, Gumbel tau floor, LOLA
warmup, full wandb instrumentation). All dependencies are in
`requirements.txt`; deterministic runs verified across all 6 modes.

## Immediate — blocks everything else

### 1. Pick the long-run budget and launch

Nothing else can start until the long run produces numbers for the
figures.

Recommended budgets (CPU, `num_envs=4`, `num_steps=64`):

| Script | Budget | Wall time | Produces |
|---|---|---|---|
| `mini_sweep.py` | `TIMESTEPS=100000` | ~3.5 h | Figs 1, 5, 6 |
| `vocab_sweep.py` | `TIMESTEPS=60000 --seed 42 --bus` | ~1 h | Fig 2 |
| `rq4_scenarios.py` | `SEEDS="42 7 123" TIMESTEPS=80000` | ~1.5 h | Fig 4 |
| `transfer_multiseed.py` | `TIMESTEPS=60000` | ~30 min (depends on mini_sweep) | Fig 3 |

At 100k timesteps each run produces ~60 log points, enough for smooth
curves.

Execution order (two parallel lanes, mirrors what we did for validation):

- **Lane A:** `mini_sweep.py` → then `transfer_multiseed.py` (reuses
  checkpoints)
- **Lane B:** `vocab_sweep.py` → then `rq4_scenarios.py` (three-seed,
  discrete+social) → then `rq4_scenarios.py` with `MODES=lola`

Total wall time ~4-5 h if parallelized. CPU has 64 cores with headroom;
GPU is occupied by other users.

wandb groups will auto-populate under **https://wandb.ai/mbzuai-research/social-hrl**
via `WANDB_RUN_GROUP` inheritance. All 6 modes (flat, continuous,
discrete, social, lola, maddpg) already included in `mini_sweep` default.

## In parallel with the long run

### 2. Draft `scripts/make_report_figures.py`

Pulls from wandb API by group name. Generates the 6 figures as PDFs
into `report/figs/`. Uses the 3 seeds grouped under each
`WANDB_RUN_GROUP` for mean ± std bands. Idempotent — safe to re-run
when the long run finishes or on a subset.

Expected figures:

| Fig | Content | Source |
|---|---|---|
| 1A | return vs step (6 modes) | wandb history, mini_sweep group |
| 1B | goal_msg_entropy vs step (discrete/social/lola) | wandb history |
| 1C | goal_space_coverage vs step (continuous/discrete/social/lola) | wandb history |
| 2 | K×L sweep → coverage | wandb summary via vocab_sweep group |
| 3 | transfer bars per source × target | transfer_multiseed aggregated_summary |
| 4 | RQ4 scenario coverage bars | rq4_scenarios aggregated_summary |
| 5 | comm_ablation_delta / listener_accuracy / topsim bars | wandb summary |
| 6 | tau anneal + recon loss (sanity) | wandb history |

### 3. Update `report/main.tex`

The current LaTeX was written against the 1M-step `outputs/2026-03-26/`
corridor runs (pre-merge narrative). It needs to be restructured:

- Drop the old `RESULTS.md` narrative (discrete entropy 6.73, social
  6.21, etc.).
- Add LOLA and MADDPG baselines from the merged code.
- Add Fig 2 (vocab sweep) and Fig 4 (scenario comparison) — these did
  not exist in the previous draft.
- Replace hard-coded numbers with `\input{figs/*.tex}` auto-generated
  placeholders (tables produced by `make_report_figures.py`).
- Renumber figures consistently.

### 4. Audit `run_all.sh` coverage vs the four-script plan

Confirm `mini_sweep + vocab_sweep + rq4_scenarios + transfer_multiseed`
cover everything `run_all.sh` would have. Gap to watch: harder
environments (KeyCorridor, MultiRoom) and the social-enhancement
variants (intrinsic_anneal, listener-reward, asymmetric-info,
rendezvous_bonus). If any of those matter for the story, add targeted
runs — otherwise deprioritize.

## After the long run finishes

### 5. Generate figures → `report/figs/`

Run `make_report_figures.py`. Inspect each PDF. Re-run per seed or
re-pull from wandb if anything looks wrong.

### 6. Reconcile narrative docs

There are currently four overlapping documents telling partially
different stories:

- `RESULTS.md` (1M-step, pre-merge)
- `DETAILED_RESULTS.md` (1M-step, pre-merge, colleague-facing)
- `SUGGESTED_APPROACH_RESULTS.md` (15k-step, Mirat's merged audit)
- `INTRO.md` / `PROJECT_INFO.md` (background, unchanged)

After the long run, pick one source of truth (probably `RESULTS.md`
rewritten from 100k-step data) and mark the others as historical.

### 7. Compile final PDF

`cd report && tectonic main.tex` (already verified working in prior
sessions). Fix any layout issues. Confirm references.bib is up to
date.

### 8. Presentation deck

15 minutes, ML8103 format. Likely structure:

- Problem: goal collapse in HRL (1 min)
- Core hypothesis: social pressure as regularizer (1 min)
- Method: Gumbel bottleneck + MAPPO + LOLA + MADDPG baseline (3 min)
- Environment: corridor + bus/turn-taking scenarios (1 min)
- Results: Fig 1 (main), Fig 2 (vocab sweep), Fig 4 (scenarios), Fig 3
  (transfer) (6 min)
- Findings: what works, what doesn't, honest limitations (2 min)
- Q&A buffer (1 min)

## Non-blocking polish (if time permits)

### 9. Periodic eval during training

Currently `evaluate()` runs once at end of training. Eval-vs-step curves
would need `self.evaluate(num_episodes=3)` every N updates and logging
the success rate. ~20 lines in each trainer, adds ~5% runtime.

### 10. Message-usage histogram

Store per-codeword counts during training, log as a wandb histogram in
the final summary. Gives a K^L visualization of which codes actually
get used. ~15 lines in the comm channel helper.

## Known constraints carried into the long run

- CPU only (GPUs occupied by other users). `torch.set_num_threads(1)`
  is already set per run.
- LOLA gets 2.5× base timesteps under `--stress` to amortize second-order
  gradient cost.
- All runs pin `torch.manual_seed`, `np.random.seed`, `random.seed`, and
  MADDPG owns a dedicated `random.Random(seed)` for its replay buffer.
  Re-running any seed produces bit-identical metrics.

## Open questions for the user / Mirat

1. **Budget**: 50k (faster, ~2 h) or 100k (recommended, ~3.5 h)?
2. **Launcher**: me, or Mirat on his machine?
3. **Harder envs**: include KeyCorridor/MultiRoom in this pass or defer?
4. **Presentation style**: slides pulled from wandb, or hand-built from
   the PDF figures?

Answer these and the long run launches within five minutes.
