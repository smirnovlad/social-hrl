"""Generate comparison plots for experiment runs.

Usage:
    python scripts/plot_results.py --experiment-dir outputs/suites/<suite>/runs \
        --output-dir outputs/suites/<suite>/plots
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_utils import discover_runs, latest_records, load_json


def smooth(data, window=50):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def load_run(record):
    """Load returns and metrics for a discovered run."""
    data = {'record': record}
    if os.path.exists(record['returns_path']):
        data['returns'] = np.load(record['returns_path'])
    if os.path.exists(record['metrics_path']):
        data['metrics'] = load_json(record['metrics_path'])
    return data


def find_latest_runs(base_dir):
    """Find the latest run for each condition and seed."""
    records = discover_runs(base_dir, allow_legacy=True)
    records = [record for record in records if os.path.exists(record['returns_path'])]
    return latest_records(records, key_fields=('condition_id', 'seed'))


def group_runs(runs):
    """Group run records by condition id."""
    grouped = {}
    for (_, _), record in sorted(runs.items(), key=lambda item: item[1]['condition_label']):
        group = grouped.setdefault(
            record['condition_id'],
            {'label': record['condition_label'], 'runs': []},
        )
        group['runs'].append(record)
    return grouped


def _ordered_groups(grouped):
    return sorted(grouped.values(), key=lambda item: item['label'])


def _success_rate_from_returns(returns):
    if len(returns) == 0:
        return 0.0
    return 100.0 * float(np.mean(np.array(returns) > 0.0))


def plot_learning_curves(runs, output_path):
    """Plot learning curves for all conditions with mean±std across seeds."""
    grouped = group_runs(runs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, group in enumerate(_ordered_groups(grouped)):
        all_returns = []
        for record in sorted(group['runs'], key=lambda item: item['seed']):
            data = load_run(record)
            if 'returns' not in data:
                continue
            all_returns.append(smooth(data['returns'], window=50))

        if not all_returns:
            continue

        min_len = min(len(returns) for returns in all_returns)
        stacked = np.array([returns[:min_len] for returns in all_returns])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        episodes = np.arange(len(mean))
        color = colors[idx % len(colors)]

        ax.plot(episodes, mean, label=group['label'], color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return (smoothed, window=50)', fontsize=12)
    ax.set_title('Learning Curves (mean ± std across available seeds)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_success_rate(runs, output_path):
    """Plot success rate over time (binned into windows)."""
    grouped = group_runs(runs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bins = 20

    for idx, group in enumerate(_ordered_groups(grouped)):
        all_rates = []
        for record in sorted(group['runs'], key=lambda item: item['seed']):
            data = load_run(record)
            if 'returns' not in data:
                continue
            returns = data['returns']
            window_size = len(returns) // n_bins
            if window_size == 0:
                continue
            rates = [
                100.0 * float(np.mean(returns[i * window_size:(i + 1) * window_size] > 0.0))
                for i in range(n_bins)
            ]
            all_rates.append(rates)

        if not all_rates:
            continue

        stacked = np.array(all_rates)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(n_bins)
        color = colors[idx % len(colors)]

        ax.plot(x, mean, label=group['label'], color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Training Progress (bins)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Training (mean ± std across available seeds)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved success rate to {output_path}")
    plt.close()


def plot_goal_metrics(runs, output_path):
    """Plot goal quality metrics comparison as bar charts."""
    grouped = group_runs(runs)
    metric_names = ['entropy', 'coverage', 'temporal_extent']
    metric_labels = ['Goal Entropy', 'Message Vocab Coverage', 'Temporal Extent']
    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(grouped) * 2.8), 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ordered = _ordered_groups(grouped)
    for ax, metric_name, metric_label in zip(axes, metric_names, metric_labels):
        labels, means, stds = [], [], []
        for group in ordered:
            values = []
            for record in group['runs']:
                data = load_run(record)
                if 'metrics' in data and metric_name in data['metrics']:
                    values.append(data['metrics'][metric_name])
            if values:
                labels.append(group['label'])
                means.append(np.mean(values))
                stds.append(np.std(values))

        if labels:
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5,
                   color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
            ax.set_xticks(x, labels, rotation=20, ha='right')
            ax.set_title(metric_label, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Goal Quality Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved goal metrics to {output_path}")
    plt.close()


def plot_discrete_token_usage(runs, output_path):
    """Plot per-position token entropy for communication-heavy conditions."""
    grouped = group_runs(runs)
    ordered = _ordered_groups(grouped)
    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(grouped) * 2.5), 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for pos, ax in enumerate(axes):
        labels, means, stds = [], [], []
        for group in ordered:
            values = []
            for record in group['runs']:
                data = load_run(record)
                if 'metrics' in data:
                    key = f'position_{pos}_entropy'
                    if key in data['metrics']:
                        values.append(data['metrics'][key])
            if values:
                labels.append(group['label'])
                means.append(np.mean(values))
                stds.append(np.std(values))

        if labels:
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5,
                   color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
            ax.set_xticks(x, labels, rotation=20, ha='right')
            ax.set_title(f'Position {pos} Entropy', fontsize=12)
            ax.set_ylabel('Entropy')
            ax.axhline(y=np.log(10), color='gray', linestyle='--', alpha=0.5, label='Max (ln 10)')
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Per-Position Token Entropy', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved token usage to {output_path}")
    plt.close()


def generate_summary_table(runs, output_path):
    """Generate a markdown summary table."""
    grouped = group_runs(runs)
    rows = []

    for group in _ordered_groups(grouped):
        final_returns = []
        success_rates = []
        entropies = []
        coverages = []

        for record in sorted(group['runs'], key=lambda item: item['seed']):
            data = load_run(record)
            metrics = data.get('metrics', {})
            if 'final_return_mean' in metrics:
                final_returns.append(metrics['final_return_mean'])
            elif 'returns' in data and len(data['returns']) > 0:
                final_returns.append(float(np.mean(data['returns'][-100:])))

            if 'eval_success_rate' in metrics:
                success_rates.append(100.0 * metrics['eval_success_rate'])
            elif 'returns' in data:
                success_rates.append(_success_rate_from_returns(data['returns']))

            if 'entropy' in metrics:
                entropies.append(metrics['entropy'])
            if 'coverage' in metrics:
                coverages.append(metrics['coverage'])

        rows.append({
            'condition': group['label'],
            'seeds': len(group['runs']),
            'return_mean': np.mean(final_returns) if final_returns else 0.0,
            'return_std': np.std(final_returns) if final_returns else 0.0,
            'success_mean': np.mean(success_rates) if success_rates else 0.0,
            'success_std': np.std(success_rates) if success_rates else 0.0,
            'entropy_mean': np.mean(entropies) if entropies else 0.0,
            'coverage_mean': np.mean(coverages) if coverages else 0.0,
        })

    with open(output_path, 'w') as f:
        f.write("# Results Summary\n\n")
        f.write("| Condition | Seeds | Final Return (mean±std) | Success Rate | Goal Entropy | Message Coverage |\n")
        f.write("|-----------|-------|-------------------------|--------------|--------------|------------------|\n")
        for row in rows:
            f.write(
                f"| {row['condition']} | {row['seeds']} | "
                f"{row['return_mean']:.4f}±{row['return_std']:.4f} | "
                f"{row['success_mean']:.1f}%±{row['success_std']:.1f}% | "
                f"{row['entropy_mean']:.3f} | {row['coverage_mean']:.4f} |\n"
            )

    print(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, default='outputs/')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    runs = find_latest_runs(args.experiment_dir)
    print(f"Found {len(runs)} latest runs:")
    for (_, _), record in sorted(runs.items(), key=lambda item: item[1]['condition_label']):
        print(f"  {record['condition_label']} seed {record['seed']}: {record['run_dir']}")
    print()

    plots_dir = args.output_dir or os.path.join(args.experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_learning_curves(runs, os.path.join(plots_dir, 'learning_curves.png'))
    plot_success_rate(runs, os.path.join(plots_dir, 'success_rate.png'))
    plot_goal_metrics(runs, os.path.join(plots_dir, 'goal_metrics.png'))
    plot_discrete_token_usage(runs, os.path.join(plots_dir, 'token_usage.png'))
    generate_summary_table(runs, os.path.join(plots_dir, 'results_summary.md'))

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
