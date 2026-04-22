"""Generate PDF figures for the ML8103 report from long_sweep JSON outputs.

Reads local summary.json files from outputs/long_sweep/20260422-140000/ and
emits PDFs under report/figs/. No wandb required.

Figures produced:
  fig1_rq1.pdf      — RQ1: eval_return + entropy across modes (200k)
  fig2_vocab.pdf    — RQ3: K×L vs scramble Δ and joint coverage (60k bus_strict)
  fig3_transfer.pdf — RQ2: transfer eval_return per source mode (120k)
  fig4_rq4.pdf      — RQ4: per-scenario joint coverage + scramble Δ (100k)
  fig5_scramble.pdf — Social scramble Δ per RQ, 1-panel summary

Usage:
    python scripts/make_long_sweep_figures.py
"""
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
SWEEP = os.path.join(ROOT, 'outputs/long_sweep/20260422-140000')
FIGS_DIR = os.path.join(ROOT, 'report/figs')
os.makedirs(FIGS_DIR, exist_ok=True)

SEEDS = [42, 7, 123]

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 120,
    'savefig.bbox': 'tight',
})

COLORS = {
    'flat': '#888888', 'continuous': '#d62728',
    'discrete': '#1f77b4', 'social': '#2ca02c',
}


def load_summary(run_dir):
    cand = sorted(glob.glob(os.path.join(run_dir, '*/summary.json')))
    if not cand:
        return None
    with open(cand[-1]) as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float('nan'), float('nan'))
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (mean(xs), stdev(xs))


def agg_rq1():
    out = {}
    for mode in ['flat', 'continuous', 'discrete', 'social']:
        rows = defaultdict(list)
        for s in SEEDS:
            d = load_summary(f'{SWEEP}/rq1/{mode}-seed{s}')
            if d is None:
                continue
            for k in ['eval_mean_return', 'entropy', 'goal_space_coverage_joint',
                      'comm_ablation_delta_scramble']:
                rows[k].append(d.get(k))
        out[mode] = {k: mean_std(v) for k, v in rows.items()}
    return out


def agg_rq3():
    cells = [(3, 1), (3, 3), (10, 1), (10, 3), (25, 3)]
    out = {}
    for K, L in cells:
        out[(K, L)] = {}
        for mode in ['discrete', 'social']:
            rows = defaultdict(list)
            for s in SEEDS:
                d = load_summary(f'{SWEEP}/rq3/K{K}_L{L}-{mode}-seed{s}')
                if d is None:
                    continue
                for k in ['eval_mean_return', 'entropy',
                          'goal_space_coverage_joint',
                          'comm_ablation_delta_scramble']:
                    rows[k].append(d.get(k))
            out[(K, L)][mode] = {k: mean_std(v) for k, v in rows.items()}
    return out


def agg_rq4():
    scens = ['baseline', 'bus', 'bus_strict', 'turn_taking']
    out = {}
    for scen in scens:
        out[scen] = {}
        for mode in ['discrete', 'social']:
            rows = defaultdict(list)
            for s in SEEDS:
                d = load_summary(f'{SWEEP}/rq4/{scen}-{mode}-seed{s}')
                if d is None:
                    continue
                for k in ['eval_mean_return', 'goal_space_coverage_joint',
                          'comm_ablation_delta_scramble']:
                    rows[k].append(d.get(k))
            out[scen][mode] = {k: mean_std(v) for k, v in rows.items()}
    return out


def agg_rq2():
    out = {}
    for mode in ['discrete', 'social']:
        src = defaultdict(list)
        tr = defaultdict(list)
        for s in SEEDS:
            d = load_summary(f'{SWEEP}/rq2_sources/{mode}-seed{s}')
            if d is not None:
                src['eval'].append(d.get('eval_mean_return'))
                src['cov'].append(d.get('goal_space_coverage_joint'))
            tp = f'{SWEEP}/rq2_transfer/{mode}-seed{s}/transfer_metrics.json'
            if os.path.exists(tp):
                with open(tp) as f:
                    t = json.load(f)
                tr['eval'].append(t.get('mean_return'))
                tr['train'].append(t.get('training_final_return'))
        out[mode] = {
            'src_eval': mean_std(src['eval']),
            'src_cov': mean_std(src['cov']),
            'tr_eval': mean_std(tr['eval']),
            'tr_train': mean_std(tr['train']),
        }
    return out


def fig1_rq1():
    data = agg_rq1()
    all_modes = ['flat', 'continuous', 'discrete', 'social']
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    x = np.arange(len(all_modes))
    colors = [COLORS[m] for m in all_modes]
    means = [data[m]['eval_mean_return'][0] for m in all_modes]
    stds = [data[m]['eval_mean_return'][1] for m in all_modes]
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(all_modes, rotation=20)
    ax.set_ylabel('eval mean return')
    ax.set_title('(a) Episode return (200k)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    entropy_modes = ['discrete', 'social']
    x2 = np.arange(len(entropy_modes))
    means = [data[m]['entropy'][0] for m in entropy_modes]
    stds = [data[m]['entropy'][1] for m in entropy_modes]
    ax.bar(x2, means, yerr=stds,
           color=[COLORS[m] for m in entropy_modes], capsize=4)
    ax.axhline(np.log(1000), color='k', linestyle='--', linewidth=0.8,
               label=r'$\ln(10^3)$ (max)')
    ax.set_xticks(x2)
    ax.set_xticklabels(entropy_modes)
    ax.set_ylabel('goal entropy (nats)')
    ax.set_title('(b) Message entropy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    fig.subplots_adjust(wspace=0.35)

    path = os.path.join(FIGS_DIR, 'fig1_rq1.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig1] wrote {path}')


def fig2_vocab():
    data = agg_rq3()
    cells = [(3, 1), (3, 3), (10, 1), (10, 3), (25, 3)]
    kl = [K * L for K, L in cells]
    labels = [f'{K}×{L}={K*L}' for K, L in cells]
    xpos = np.arange(len(cells))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    ax = axes[0]
    for mode in ['discrete', 'social']:
        ys = [data[c][mode]['goal_space_coverage_joint'][0] for c in cells]
        es = [data[c][mode]['goal_space_coverage_joint'][1] for c in cells]
        if all(np.isnan(y) for y in ys):
            continue
        ys = [y if not np.isnan(y) else 0 for y in ys]
        es = [e if not np.isnan(e) else 0 for e in es]
        ax.errorbar(xpos, ys, yerr=es, marker='o', label=mode,
                    color=COLORS[mode], linewidth=1.5, capsize=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'vocab × length  $K \times L$ (= capacity)')
    ax.set_ylabel('joint goal coverage')
    ax.set_title('(a) Coverage vs bottleneck capacity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ys = [data[c]['social']['comm_ablation_delta_scramble'][0] for c in cells]
    es = [data[c]['social']['comm_ablation_delta_scramble'][1] for c in cells]
    ax.errorbar(xpos, ys, yerr=es, marker='s', color=COLORS['social'],
                linewidth=1.5, capsize=3, label='social')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'vocab × length  $K \times L$ (= capacity)')
    ax.set_ylabel(r'scramble $\Delta$ (real $-$ scrambled)')
    ax.set_title('(b) Channel utility (U-shape)')
    ax.grid(True, alpha=0.3)

    fig.subplots_adjust(wspace=0.3)

    path = os.path.join(FIGS_DIR, 'fig2_vocab.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig2] wrote {path}')


def fig3_transfer():
    data = agg_rq2()
    modes = ['discrete', 'social']
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    ax = axes[0]
    means = [data[m]['src_eval'][0] for m in modes]
    stds = [data[m]['src_eval'][1] for m in modes]
    ax.bar(modes, means, yerr=stds, color=[COLORS[m] for m in modes], capsize=4)
    ax.set_ylabel('eval mean return (source, 200k)')
    ax.set_title('(a) Source training (2-agent)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    means = [data[m]['tr_eval'][0] for m in modes]
    stds = [data[m]['tr_eval'][1] for m in modes]
    ax.bar(modes, means, yerr=stds, color=[COLORS[m] for m in modes], capsize=4)
    ax.set_ylabel('eval mean return (target, 120k)')
    ax.set_title('(b) Transfer to S15-W3 (same-family)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.subplots_adjust(wspace=0.45)

    path = os.path.join(FIGS_DIR, 'fig3_transfer.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig3] wrote {path}')


def fig4_rq4():
    data = agg_rq4()
    scens = ['baseline', 'bus', 'bus_strict', 'turn_taking']
    x = np.arange(len(scens))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.2))

    ax = axes[0]
    cov_mu = [data[s]['social']['goal_space_coverage_joint'][0] for s in scens]
    cov_sd = [data[s]['social']['goal_space_coverage_joint'][1] for s in scens]
    ax.bar(x, cov_mu, yerr=cov_sd, color=COLORS['social'], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=30, ha='right')
    ax.set_ylabel('joint goal coverage (social)')
    ax.set_title('(a) Coverage per scenario')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    scr_mu = [data[s]['social']['comm_ablation_delta_scramble'][0] for s in scens]
    scr_sd = [data[s]['social']['comm_ablation_delta_scramble'][1] for s in scens]
    bars = ax.bar(x, scr_mu, yerr=scr_sd, color=COLORS['social'], capsize=4)
    for b, v in zip(bars, scr_mu):
        if v < 0:
            b.set_color('#a0a0a0')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(scens, rotation=30, ha='right')
    ax.set_ylabel(r'scramble $\Delta$ (social)')
    ax.set_title('(b) Channel utility per scenario')
    ax.grid(True, alpha=0.3, axis='y')

    fig.subplots_adjust(wspace=0.3)

    path = os.path.join(FIGS_DIR, 'fig4_rq4.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig4] wrote {path}')


def fig5_scramble():
    rq1 = agg_rq1()
    rq3 = agg_rq3()
    rq4 = agg_rq4()

    labels = []
    mus = []
    sds = []

    labels.append('RQ1\n(200k)')
    mu, sd = rq1['social']['comm_ablation_delta_scramble']
    mus.append(mu); sds.append(sd)

    for K, L in [(3, 1), (3, 3), (10, 1), (10, 3), (25, 3)]:
        labels.append(f'RQ3\n{K}×{L}')
        mu, sd = rq3[(K, L)]['social']['comm_ablation_delta_scramble']
        mus.append(mu); sds.append(sd)

    for s in ['baseline', 'bus', 'bus_strict', 'turn_taking']:
        labels.append(f'RQ4\n{s}')
        mu, sd = rq4[s]['social']['comm_ablation_delta_scramble']
        mus.append(mu); sds.append(sd)

    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    x = np.arange(len(labels))
    colors = ['#2ca02c' if m > 0 else '#a0a0a0' for m in mus]
    ax.bar(x, mus, yerr=sds, color=colors, capsize=3)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r'scramble $\Delta$ (real $-$ scrambled)')
    ax.set_title('Social channel utility across experiments')
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(FIGS_DIR, 'fig5_scramble.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[fig5] wrote {path}')


if __name__ == '__main__':
    fig1_rq1()
    fig2_vocab()
    fig3_transfer()
    fig4_rq4()
    fig5_scramble()
    print(f'\n[done] figures in {FIGS_DIR}')
