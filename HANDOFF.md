# HANDOFF — Social HRL final report (ML8103)

Generated 2026-04-22 after the overnight long-run sweep finished.
Everything mechanical is done. What remains is **prose only** — see
`% TODO(vlad)` markers in `report/main.tex`.

---

## Status of the runs

| Group                 | Budget                          | Result location                                              | Status |
|-----------------------|---------------------------------|--------------------------------------------------------------|--------|
| `mini_sweep`          | 6 modes × 3 seeds × 100k        | `outputs/mini_sweep/aggregated_summary.json`                 | done   |
| `vocab_sweep`         | 5 cells × 2 seeds × 60k         | `outputs/vocab_sweep/20260421-232242/summary.json`           | done   |
| `rq4_scenarios`       | 2 scenarios × 6 modes × 3 seeds × 80k | `outputs/rq4_scenarios/20260422-004820/aggregated_summary.json` | done   |
| `rq4_scenarios × LOLA` | (folded into rq4_scenarios)    | `outputs/rq4_scenarios/20260421-234007/aggregated_summary.json` | done   |
| `transfer_multiseed`  | 2 modes × 3 seeds × 60k         | `outputs/transfer_multiseed/20260422-004253/aggregated_summary.json` | done   |

WandB groups (for re-pulling history if needed):
- `mini_sweep-ts100000-20260421-232233`
- `vocab_sweep-ts60000-20260421-232242`
- `rq4_scenarios-ts80000-20260421-234007`
- `rq4_scenarios-ts80000-20260422-004820`

---

## Headline numbers (100k stress, 3 seeds)

`goal_space_coverage` — fraction of partitioned goal space ever visited:

| Mode       | mean ± std        |
|------------|-------------------|
| flat       | n/a (no goals)    |
| continuous | 0.480 ± 0.190     |
| discrete   | **1.000 ± 0.000** |
| social     | 0.716 ± 0.100     |
| lola       | 0.546 ± 0.060     |
| maddpg     | n/a               |

`mutual_information(goal, message)`:
- discrete 1.575 ± 0.222
- social   0.312 ± 0.030
- lola     0.332 ± 0.016

`listener_accuracy`:
- discrete 0.248 ± 0.058
- social   0.049 ± 0.025
- lola     0.059 ± 0.034

**Transfer (widened corridor S11-W3, frozen manager, 60k worker re-train):**
- discrete source: eval mean return −0.767 ± 0.024
- social source:   eval mean return −0.750 ± 0.000
- gap +0.017 vs combined noise 0.024 → **WITHIN NOISE** (RQ2 not supported)

These confirm and amplify the March progress-report finding: the discrete
bottleneck is what matters; adding a partner does not improve goal diversity
or transfer.

---

## Figures (already in `report/figs/`)

| File                       | Source                                      | Used in              |
|----------------------------|---------------------------------------------|----------------------|
| `fig1_main.pdf`            | wandb history of `mini_sweep` (3-panel)     | Results §            |
| `fig2_vocab_sweep.pdf`     | local `vocab_sweep/.../summary.json`        | §H4                  |
| `fig3_transfer.pdf`        | local `transfer_multiseed/.../aggregated_summary.json` | §H3       |
| `fig4_rq4_scenarios.pdf`   | local `rq4_scenarios/.../aggregated_summary.json` | §RQ4              |
| `fig5_social_bars.pdf`     | wandb summary of `mini_sweep`               | §H2                  |
| `fig6_sanity.pdf`          | wandb history (Gumbel τ + recon loss)       | Appendix             |

Regenerate any figure with:

```bash
python scripts/make_report_figures.py \
  --mini-sweep-group mini_sweep-ts100000-20260421-232233 \
  --only fig1,fig5,fig6        # or any subset
```

⚠️  fig1 and fig6 hit wandb history per run (18 runs × ~5 keys). The wandb
public API times out occasionally. Re-run if needed; the script retries.

---

## Prose still needed (search `report/main.tex` for `TODO(vlad)`)

1. **Results intro** — replace 1M corridor numbers with the 100k stress numbers above.
2. **Fig 1 caption** — one-sentence interpretation of the 3-panel.
3. **Fig 5 caption** — call out which social metric stands out.
4. **§H3 transfer prose** — final verdict line on RQ2.
5. **§H4 take-away** — does a sweet spot exist, or is the gap discrete-vs-social scale-invariant?
6. **§RQ4 headline** — which mode breaks/holds under the bus stressor?
7. **Fig 6 caption** — note τ converged / recon loss flattened.
8. **Next steps list** — H4 / same-family transfer are no longer "next steps", reword.

---

## Files Mirat / Vlad should reconcile

`RESULTS.md`, `DETAILED_RESULTS.md`, `SUGGESTED_APPROACH_RESULTS.md` were
written before the new sweep. They still describe the 1M run. Decide whether
to:
- delete the stale ones and keep only `report/main.tex` as canonical, or
- update them as supplementary notes for the slide deck.

`NEXT_STEPS.md` (the long-run plan) is now history; archive or delete.

---

## Build the PDF

```bash
cd report && tectonic main.tex
# → main.pdf (currently 161 KiB)
```

PDF builds clean. Only cosmetic underfull-hbox warnings.
