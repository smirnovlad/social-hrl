"""Generate PDF figures for the ML8103 report from wandb runs + local JSON.

Pulls runs from wandb.ai/mbzuai-research/social-hrl filtered by group.
Aggregates history across 3 seeds per mode for time-series with mean ±
std bands, and summary for bar charts.

Figures produced (into report/figs/):
  fig1_main.pdf       - 3-panel: return, goal_msg_entropy, goal_space_coverage vs step
  fig2_vocab_sweep.pdf - coverage vs K*L, discrete vs social
  fig3_transfer.pdf   - transfer bars: discrete vs social source
  fig4_rq4_scenarios.pdf - coverage per (scenario, mode)
  fig5_social_bars.pdf - comm_ablation_delta, listener_acc, topsim, MI bars
  fig6_sanity.pdf     - gumbel tau anneal + comm_recon_loss over time

Usage:
    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --mini-sweep-group mini_sweep-ts100000-...
    python scripts/make_report_figures.py --only fig1,fig3

Idempotent. Re-run anytime after experiments finish.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
FIGS_DIR = os.path.join(ROOT, 'report', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

ENTITY = 'mbzuai-research'
PROJECT = 'social-hrl'

MODE_COLORS = {
    'flat':       '#888888',
    'continuous': '#d62728',
    'discrete':   '#1f77b4',
    'social':     '#2ca02c',
    'lola':       '#9467bd',
    'maddpg':     '#ff7f0e',
}
MODE_ORDER = ['flat', 'continuous', 'discrete', 'social', 'lola', 'maddpg']


def _init_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
    })
    return plt


def _fetch_runs(group: str | None = None, tags: Iterable[str] | None = None):
    """Return a list of wandb.Run objects matching filters."""
    import wandb
    api = wandb.Api(timeout=30)
    filters = {}
    if group:
        filters['group'] = group
    if tags:
        filters['tags'] = {'$in': list(tags)}
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters=filters or None)
    return list(runs)


def _history_for_run(run, keys: Iterable[str], samples: int = 500):
    """Fetch history (time-series) for given keys from a wandb run.

    Returns dict: key -> (steps_array, values_array), with NaNs filtered.
    """
    df = run.history(keys=list(keys), samples=samples, pandas=True)
    if df is None or len(df) == 0:
        return {}
    out = {}
    steps = df['_step'].to_numpy() if '_step' in df.columns else df.index.to_numpy()
    for k in keys:
        if k not in df.columns:
            continue
        v = df[k].to_numpy(dtype=float)
        mask = ~np.isnan(v)
        if mask.sum() < 2:
            continue
        out[k] = (steps[mask], v[mask])
    return out


def _group_by_mode(runs):
    by_mode = defaultdict(list)
    for r in runs:
        mode = r.config.get('mode')
        if mode:
            by_mode[mode].append(r)
    return dict(by_mode)


def _interp_stack(xs_list, ys_list, n_points: int = 200):
    """Interpolate each (xs, ys) pair onto a common grid. Return (grid, matrix)."""
    if not xs_list:
        return None, None
    x_min = max(x[0] for x in xs_list)
    x_max = min(x[-1] for x in xs_list)
    if x_max <= x_min:
        return None, None
    grid = np.linspace(x_min, x_max, n_points)
    mat = np.stack([np.interp(grid, x, y) for x, y in zip(xs_list, ys_list)])
    return grid, mat


def _plot_timeseries_panel(ax, by_mode, key: str, title: str, ylabel: str):
    for mode in MODE_ORDER:
        runs = by_mode.get(mode, [])
        if not runs:
            continue
        series = []
        for r in runs:
            h = _history_for_run(r, [key])
            if key in h:
                series.append(h[key])
        if not series:
            continue
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        grid, mat = _interp_stack(xs, ys)
        if grid is None:
            continue
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        color = MODE_COLORS.get(mode, 'black')
        ax.plot(grid, mean, label=mode, color=color, linewidth=1.5)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def fig1_main(mini_sweep_group: str):
    """Main 3-panel story figure: return, goal_msg_entropy, goal_space_coverage."""
    plt = _init_matplotlib()
    runs = _fetch_runs(group=mini_sweep_group)
    if not runs:
        print(f'[fig1] no runs in group={mini_sweep_group}')
        return
    by_mode = _group_by_mode(runs)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    _plot_timeseries_panel(axes[0], by_mode, 'mean_return',
                           '(a) Episode return', 'mean return')
    _plot_timeseries_panel(axes[1], by_mode, 'goal_msg_entropy',
                           '(b) Goal message entropy', 'entropy (nats)')
    _plot_timeseries_panel(axes[2], by_mode, 'goal_space_coverage',
                           '(c) Goal-space coverage', 'coverage')
    axes[0].legend(loc='lower right', ncol=2)
    path = os.path.join(FIGS_DIR, 'fig1_main.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig1] wrote {path} from {len(runs)} runs, modes={sorted(by_mode)}')


def fig2_vocab_sweep(vocab_group: str | None = None,
                     json_path: str | None = None):
    """K*L vs coverage, discrete vs social. Prefer wandb, fall back to JSON."""
    plt = _init_matplotlib()
    rows = []
    if vocab_group:
        runs = _fetch_runs(group=vocab_group)
        for r in runs:
            cfg = r.config
            summ = r.summary
            K = cfg.get('vocab_size')
            L = cfg.get('message_length')
            mode = cfg.get('mode')
            cov = summ.get('goal_space_coverage')
            if K and L and mode and cov is not None:
                rows.append({'K': K, 'L': L, 'mode': mode, 'coverage': float(cov)})
    if not rows and json_path and os.path.exists(json_path):
        with open(json_path) as f:
            rows = json.load(f)
    if not rows:
        print('[fig2] no data')
        return
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row['mode']].append(row)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for mode in ['discrete', 'social']:
        pts = sorted(by_mode.get(mode, []), key=lambda r: r['K'] * r['L'])
        if not pts:
            continue
        xs = [r['K'] * r['L'] for r in pts]
        ys = [r['coverage'] for r in pts]
        ax.plot(xs, ys, 'o-', label=mode,
                color=MODE_COLORS[mode], linewidth=1.5, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('bottleneck capacity K × L (log scale)')
    ax.set_ylabel('goal-space coverage')
    ax.set_title('Vocab sweep: bottleneck capacity vs goal-space coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(FIGS_DIR, 'fig2_vocab_sweep.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig2] wrote {path} from {len(rows)} cells')


def fig3_transfer(json_path: str):
    """Transfer: eval_return per source mode on target env."""
    plt = _init_matplotlib()
    if not os.path.exists(json_path):
        print(f'[fig3] no file at {json_path}')
        return
    with open(json_path) as f:
        d = json.load(f)
    if isinstance(d, dict) and 'aggregate' in d:
        d = d['aggregate']
    sources = []
    means = []
    stds = []
    for src, stats in d.items():
        m = stats.get('eval_mean_return')
        if m and m.get('mean') is not None:
            sources.append(src)
            means.append(m['mean'])
            stds.append(m.get('std', 0.0))
    if not sources:
        print('[fig3] no transfer data')
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = [MODE_COLORS.get(s, 'gray') for s in sources]
    ax.bar(sources, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel('eval mean return (target env)')
    ax.set_title('Transfer: frozen-manager eval on widened corridor')
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(FIGS_DIR, 'fig3_transfer.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig3] wrote {path}')


def fig4_rq4_scenarios(json_path: str):
    """RQ4 scenario coverage bars per (scenario, mode)."""
    plt = _init_matplotlib()
    if not os.path.exists(json_path):
        print(f'[fig4] no file at {json_path}')
        return
    with open(json_path) as f:
        d = json.load(f)
    scenarios = list(d.keys())
    modes = sorted({m for s in d.values() for m in s.keys()})
    means = {m: [] for m in modes}
    stds = {m: [] for m in modes}
    for scen in scenarios:
        for m in modes:
            stats = d.get(scen, {}).get(m, {}).get('goal_space_coverage', {})
            means[m].append(stats.get('mean', 0.0) if stats else 0.0)
            stds[m].append(stats.get('std', 0.0) if stats else 0.0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / max(1, len(modes))
    x = np.arange(len(scenarios))
    for i, m in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, means[m], width, yerr=stds[m],
               label=m, color=MODE_COLORS.get(m, 'gray'), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('goal-space coverage')
    ax.set_title('RQ4: coverage per coordination scenario')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(FIGS_DIR, 'fig4_rq4_scenarios.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig4] wrote {path}')


def fig5_social_bars(mini_sweep_group: str):
    """Summary bars for social-family metrics: comm_ablation_delta, listener_acc, topsim, MI."""
    plt = _init_matplotlib()
    runs = _fetch_runs(group=mini_sweep_group)
    by_mode = _group_by_mode(runs)
    keys = ['comm_ablation_delta', 'listener_accuracy',
            'topographic_similarity', 'mutual_information']
    titles = {
        'comm_ablation_delta': 'comm_ablation_delta\n(with − without)',
        'listener_accuracy': 'listener_accuracy\n(ridge-probe R²)',
        'topographic_similarity': 'topographic_similarity\n(Spearman ρ)',
        'mutual_information': 'I(message; state)\n(nats)',
    }
    modes = [m for m in ['discrete', 'social', 'lola'] if m in by_mode]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        means, stds = [], []
        for m in modes:
            vals = []
            for r in by_mode[m]:
                v = r.summary.get(key)
                if v is not None and not np.isnan(float(v)):
                    vals.append(float(v))
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        ax.bar(modes, means, yerr=stds,
               color=[MODE_COLORS[m] for m in modes], capsize=4)
        ax.set_title(titles[key])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
    path = os.path.join(FIGS_DIR, 'fig5_social_bars.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig5] wrote {path}')


def fig6_sanity(mini_sweep_group: str):
    """Gumbel tau anneal + comm_recon_loss over time."""
    plt = _init_matplotlib()
    runs = _fetch_runs(group=mini_sweep_group)
    by_mode = _group_by_mode(runs)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    _plot_timeseries_panel(axes[0], by_mode, 'gumbel_tau',
                           '(a) Gumbel-Softmax τ', 'τ')
    _plot_timeseries_panel(axes[1], by_mode, 'comm_recon_loss',
                           '(b) Comm reconstruction loss', 'MSE')
    axes[0].legend(loc='upper right', ncol=2)
    path = os.path.join(FIGS_DIR, 'fig6_sanity.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig6] wrote {path}')


def _latest_aggregated_summary(root: str) -> str | None:
    """Find most recent aggregated_summary.json under outputs/<root>/."""
    base = os.path.join(ROOT, 'outputs', root)
    if not os.path.isdir(base):
        return None
    candidates = []
    for dirpath, _, files in os.walk(base):
        if 'aggregated_summary.json' in files:
            candidates.append(os.path.join(dirpath, 'aggregated_summary.json'))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _latest_vocab_summary() -> str | None:
    base = os.path.join(ROOT, 'outputs', 'vocab_sweep')
    if not os.path.isdir(base):
        return None
    candidates = []
    for dirpath, _, files in os.walk(base):
        if 'summary.json' in files:
            candidates.append(os.path.join(dirpath, 'summary.json'))
    return max(candidates, key=os.path.getmtime) if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mini-sweep-group', default=None,
                    help='wandb group for Fig 1, 5, 6. Defaults to latest.')
    ap.add_argument('--vocab-group', default=None,
                    help='wandb group for Fig 2. Falls back to local JSON.')
    ap.add_argument('--rq4-json', default=None,
                    help='path to rq4_scenarios aggregated_summary.json.')
    ap.add_argument('--transfer-json', default=None,
                    help='path to transfer_multiseed aggregated_summary.json.')
    ap.add_argument('--vocab-json', default=None,
                    help='path to vocab_sweep summary.json.')
    ap.add_argument('--only', default=None,
                    help='comma-separated subset: fig1,fig2,fig3,fig4,fig5,fig6.')
    args = ap.parse_args()

    rq4_json = args.rq4_json or _latest_aggregated_summary('rq4_scenarios')
    transfer_json = args.transfer_json or _latest_aggregated_summary('transfer_multiseed')
    vocab_json = args.vocab_json or _latest_vocab_summary()

    only = set(args.only.split(',')) if args.only else None

    def run(name, fn, *a, **kw):
        if only and name not in only:
            return
        try:
            fn(*a, **kw)
        except Exception as e:
            print(f'[{name}] FAILED: {e}')

    if args.mini_sweep_group:
        run('fig1', fig1_main, args.mini_sweep_group)
        run('fig5', fig5_social_bars, args.mini_sweep_group)
        run('fig6', fig6_sanity, args.mini_sweep_group)
    else:
        print('[info] --mini-sweep-group not given; skipping fig1/5/6')

    run('fig2', fig2_vocab_sweep, args.vocab_group, vocab_json)
    if transfer_json:
        run('fig3', fig3_transfer, transfer_json)
    if rq4_json:
        run('fig4', fig4_rq4_scenarios, rq4_json)

    print(f'\n[done] figures in {FIGS_DIR}')


if __name__ == '__main__':
    main()
