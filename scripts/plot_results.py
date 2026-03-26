"""Generate comparison plots for all experiment conditions.

Usage:
    python scripts/plot_results.py --experiment-dir outputs/
"""

import argparse
import os
import sys
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def smooth(data, window=50):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def find_latest_runs(base_dir):
    """Find the latest run for each (mode, seed) pair.

    Scans outputs/YYYY-MM-DD/mode_seedN/HH-MM-SS/ structure.
    Returns dict: {(mode, seed): path_to_latest_run}
    """
    runs = {}
    base = Path(base_dir)

    for returns_file in sorted(base.rglob('returns.npy')):
        run_dir = returns_file.parent
        # Parse: outputs/DATE/mode_seedN/TIME/returns.npy
        parts = run_dir.parts
        for i, part in enumerate(parts):
            match = re.match(r'^(flat|continuous|discrete|social)_seed(\d+)$', part)
            if match:
                mode = match.group(1)
                seed = int(match.group(2))
                key = (mode, seed)
                # Keep the latest (sorted by path, last wins)
                runs[key] = str(run_dir)
                break

    return runs


def load_run(run_dir):
    """Load returns and metrics from a run directory."""
    data = {}
    returns_path = os.path.join(run_dir, 'returns.npy')
    metrics_path = os.path.join(run_dir, 'metrics.json')

    if os.path.exists(returns_path):
        data['returns'] = np.load(returns_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data['metrics'] = json.load(f)
    return data


def plot_learning_curves(runs, output_path):
    """Plot learning curves for all conditions with mean±std across seeds."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = {
        'flat': ('Flat PPO', '#1f77b4'),
        'continuous': ('HRL Continuous', '#ff7f0e'),
        'discrete': ('HRL Discrete', '#2ca02c'),
        'social': ('HRL Social', '#d62728'),
    }

    for mode, (label, color) in conditions.items():
        mode_runs = {k: v for k, v in runs.items() if k[0] == mode}
        if not mode_runs:
            continue

        all_returns = []
        for (m, seed), path in sorted(mode_runs.items()):
            data = load_run(path)
            if 'returns' in data:
                smoothed = smooth(data['returns'], window=50)
                all_returns.append(smoothed)

        if not all_returns:
            continue

        min_len = min(len(r) for r in all_returns)
        all_returns = np.array([r[:min_len] for r in all_returns])

        mean = all_returns.mean(axis=0)
        std = all_returns.std(axis=0)

        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return (smoothed, window=50)', fontsize=12)
    ax.set_title('Learning Curves (mean ± std across 3 seeds)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_success_rate(runs, output_path):
    """Plot success rate over time (binned into windows)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = {
        'flat': ('Flat PPO', '#1f77b4'),
        'continuous': ('HRL Continuous', '#ff7f0e'),
        'discrete': ('HRL Discrete', '#2ca02c'),
        'social': ('HRL Social', '#d62728'),
    }

    n_bins = 20

    for mode, (label, color) in conditions.items():
        mode_runs = {k: v for k, v in runs.items() if k[0] == mode}
        if not mode_runs:
            continue

        all_rates = []
        for (m, seed), path in sorted(mode_runs.items()):
            data = load_run(path)
            if 'returns' not in data:
                continue
            r = data['returns']
            ws = len(r) // n_bins
            if ws == 0:
                continue
            rates = [100 * np.count_nonzero(r[i*ws:(i+1)*ws]) / ws for i in range(n_bins)]
            all_rates.append(rates)

        if not all_rates:
            continue

        all_rates = np.array(all_rates)
        mean = all_rates.mean(axis=0)
        std = all_rates.std(axis=0)

        x = np.arange(n_bins)
        ax.plot(x, mean, label=label, color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Training Progress (bins)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Training (mean ± std across 3 seeds)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved success rate to {output_path}")
    plt.close()


def plot_goal_metrics(runs, output_path):
    """Plot goal quality metrics comparison as bar charts."""
    conditions = {
        'continuous': ('HRL Continuous', '#ff7f0e'),
        'discrete': ('HRL Discrete', '#2ca02c'),
        'social': ('HRL Social', '#d62728'),
    }

    metric_names = ['entropy', 'coverage', 'temporal_extent']
    metric_labels = ['Goal Entropy', 'Goal Coverage', 'Temporal Extent']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metric_name, metric_label, ax in zip(metric_names, metric_labels, axes):
        labels, means, stds, colors = [], [], [], []

        for mode, (label, color) in conditions.items():
            mode_runs = {k: v for k, v in runs.items() if k[0] == mode}
            values = []
            for (m, seed), path in sorted(mode_runs.items()):
                data = load_run(path)
                if 'metrics' in data and metric_name in data['metrics']:
                    values.append(data['metrics'][metric_name])

            if values:
                labels.append(label)
                means.append(np.mean(values))
                stds.append(np.std(values))
                colors.append(color)

        if labels:
            ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax.set_title(metric_label, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Goal Quality Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved goal metrics to {output_path}")
    plt.close()


def plot_discrete_token_usage(runs, output_path):
    """Plot per-position token usage for discrete/social modes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for mode, color, marker in [('discrete', '#2ca02c', 'o'), ('social', '#d62728', 's')]:
        mode_runs = {k: v for k, v in runs.items() if k[0] == mode}
        if not mode_runs:
            continue

        for pos in range(3):
            entropies = []
            tokens_used = []
            for (m, seed), path in sorted(mode_runs.items()):
                data = load_run(path)
                if 'metrics' not in data:
                    continue
                m_data = data['metrics']
                key_e = f'position_{pos}_entropy'
                key_t = f'position_{pos}_used_tokens'
                if key_e in m_data:
                    entropies.append(m_data[key_e])
                if key_t in m_data:
                    tokens_used.append(m_data[key_t])

            if entropies:
                ax = axes[pos]
                x = range(len(entropies))
                ax.bar([i + (0.35 if mode == 'social' else 0) for i in x],
                       entropies, width=0.35, color=color, alpha=0.8,
                       label=f'{mode.title()}')
                ax.set_title(f'Position {pos} Entropy', fontsize=12)
                ax.set_ylabel('Entropy (bits)')
                ax.set_xlabel('Seed')
                ax.axhline(y=np.log(10), color='gray', linestyle='--',
                          alpha=0.5, label='Max (ln 10)')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Per-Position Token Entropy (K=10)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved token usage to {output_path}")
    plt.close()


def generate_summary_table(runs, output_path):
    """Generate a markdown summary table."""
    conditions = ['flat', 'continuous', 'discrete', 'social']
    labels = {
        'flat': 'Flat PPO', 'continuous': 'HRL Continuous',
        'discrete': 'HRL Discrete', 'social': 'HRL Social',
    }

    rows = []
    for mode in conditions:
        mode_runs = {k: v for k, v in runs.items() if k[0] == mode}
        if not mode_runs:
            continue

        returns_list, entropies, coverages, success_rates = [], [], [], []

        for (m, seed), path in sorted(mode_runs.items()):
            data = load_run(path)
            if 'returns' in data:
                r = data['returns']
                returns_list.append(r.mean())
                success_rates.append(100 * np.count_nonzero(r) / len(r))
            if 'metrics' in data:
                m_data = data['metrics']
                entropies.append(m_data.get('entropy', 0))
                coverages.append(m_data.get('coverage', 0))

        rows.append({
            'condition': labels[mode],
            'seeds': len(mode_runs),
            'return_mean': np.mean(returns_list) if returns_list else 0,
            'return_std': np.std(returns_list) if returns_list else 0,
            'success_mean': np.mean(success_rates) if success_rates else 0,
            'success_std': np.std(success_rates) if success_rates else 0,
            'entropy_mean': np.mean(entropies) if entropies else 0,
            'coverage_mean': np.mean(coverages) if coverages else 0,
        })

    with open(output_path, 'w') as f:
        f.write("# Experiment 1 Results Summary\n\n")
        f.write("| Condition | Seeds | Return (mean±std) | Success Rate | Goal Entropy | Coverage |\n")
        f.write("|-----------|-------|-------------------|--------------|--------------|----------|\n")
        for r in rows:
            f.write(f"| {r['condition']} | {r['seeds']} | "
                    f"{r['return_mean']:.4f}±{r['return_std']:.4f} | "
                    f"{r['success_mean']:.1f}%±{r['success_std']:.1f}% | "
                    f"{r['entropy_mean']:.3f} | "
                    f"{r['coverage_mean']:.4f} |\n")
        f.write("\n*3 seeds per condition, 1M steps each.*\n")

    print(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, default='outputs/')
    args = parser.parse_args()

    runs = find_latest_runs(args.experiment_dir)
    print(f"Found {len(runs)} runs:")
    for (mode, seed), path in sorted(runs.items()):
        print(f"  {mode} seed {seed}: {path}")
    print()

    plots_dir = os.path.join(args.experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_learning_curves(runs, os.path.join(plots_dir, 'learning_curves.png'))
    plot_success_rate(runs, os.path.join(plots_dir, 'success_rate.png'))
    plot_goal_metrics(runs, os.path.join(plots_dir, 'goal_metrics.png'))
    plot_discrete_token_usage(runs, os.path.join(plots_dir, 'token_usage.png'))
    generate_summary_table(runs, os.path.join(plots_dir, 'results_summary.md'))

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
