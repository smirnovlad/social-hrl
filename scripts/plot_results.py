"""Generate comparison plots for all experiment conditions.

Usage:
    python scripts/plot_results.py --experiment-dir outputs/
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def smooth(data, window=50):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def load_experiment_data(exp_dir):
    """Load returns and metrics from an experiment directory."""
    returns_path = os.path.join(exp_dir, 'returns.npy')
    metrics_path = os.path.join(exp_dir, 'metrics.json')

    data = {}
    if os.path.exists(returns_path):
        data['returns'] = np.load(returns_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data['metrics'] = json.load(f)
    return data


def find_experiments(base_dir, prefix):
    """Find all experiment directories matching a prefix."""
    experiments = []
    for d in sorted(Path(base_dir).iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            data = load_experiment_data(str(d))
            if data:
                experiments.append((d.name, data))
    return experiments


def plot_learning_curves(base_dir, output_path):
    """Plot learning curves for all conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    conditions = {
        'exp1_flat': ('Flat PPO', '#1f77b4'),
        'exp1_continuous': ('HRL Continuous', '#ff7f0e'),
        'exp1_discrete': ('HRL Discrete', '#2ca02c'),
    }

    for prefix, (label, color) in conditions.items():
        experiments = find_experiments(base_dir, prefix)
        if not experiments:
            continue

        all_returns = []
        for name, data in experiments:
            if 'returns' in data:
                smoothed = smooth(data['returns'], window=50)
                all_returns.append(smoothed)

        if not all_returns:
            continue

        # Align lengths
        min_len = min(len(r) for r in all_returns)
        all_returns = [r[:min_len] for r in all_returns]
        all_returns = np.array(all_returns)

        mean = all_returns.mean(axis=0)
        std = all_returns.std(axis=0)

        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('Learning Curves: Flat vs. HRL Continuous vs. HRL Discrete', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_goal_metrics(base_dir, output_path):
    """Plot goal quality metrics comparison."""
    conditions = {
        'exp1_continuous': 'HRL Continuous',
        'exp1_discrete': 'HRL Discrete',
    }

    metric_names = ['entropy', 'coverage', 'temporal_extent']
    metric_labels = ['Goal Entropy', 'Goal Coverage', 'Temporal Extent']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metric_name, metric_label, ax in zip(metric_names, metric_labels, axes):
        values_by_condition = {}

        for prefix, label in conditions.items():
            experiments = find_experiments(base_dir, prefix)
            values = []
            for name, data in experiments:
                if 'metrics' in data and metric_name in data['metrics']:
                    values.append(data['metrics'][metric_name])

            if values:
                values_by_condition[label] = values

        if values_by_condition:
            labels = list(values_by_condition.keys())
            means = [np.mean(v) for v in values_by_condition.values()]
            stds = [np.std(v) for v in values_by_condition.values()]

            colors = ['#ff7f0e', '#2ca02c'][:len(labels)]
            bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax.set_title(metric_label, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Goal Quality Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved goal metrics to {output_path}")
    plt.close()


def generate_summary_table(base_dir, output_path):
    """Generate a markdown summary table of all results."""
    conditions = {
        'exp1_flat': 'Flat PPO',
        'exp1_continuous': 'HRL Continuous',
        'exp1_discrete': 'HRL Discrete',
    }

    rows = []
    for prefix, label in conditions.items():
        experiments = find_experiments(base_dir, prefix)
        if not experiments:
            continue

        returns = []
        entropies = []
        coverages = []

        for name, data in experiments:
            if 'metrics' in data:
                m = data['metrics']
                returns.append(m.get('final_return_mean', 0))
                entropies.append(m.get('entropy', 0))
                coverages.append(m.get('coverage', 0))

        rows.append({
            'condition': label,
            'return_mean': np.mean(returns) if returns else 0,
            'return_std': np.std(returns) if returns else 0,
            'entropy_mean': np.mean(entropies) if entropies else 0,
            'coverage_mean': np.mean(coverages) if coverages else 0,
        })

    with open(output_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write("| Condition | Return (mean±std) | Goal Entropy | Goal Coverage |\n")
        f.write("|-----------|-------------------|--------------|---------------|\n")
        for r in rows:
            f.write(f"| {r['condition']} | "
                    f"{r['return_mean']:.2f}±{r['return_std']:.2f} | "
                    f"{r['entropy_mean']:.3f} | "
                    f"{r['coverage_mean']:.3f} |\n")

    print(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, default='outputs/')
    args = parser.parse_args()

    base = args.experiment_dir
    plots_dir = os.path.join(base, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_learning_curves(base, os.path.join(plots_dir, 'learning_curves.png'))
    plot_goal_metrics(base, os.path.join(plots_dir, 'goal_metrics.png'))
    generate_summary_table(base, os.path.join(plots_dir, 'results_summary.md'))

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
