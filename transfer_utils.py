"""Lightweight helpers for transfer source selection and validation."""

from experiment_utils import discover_runs


def validate_transfer_request(source_mode):
    """Validate source/target compatibility for the current transfer implementation."""
    if source_mode == 'social':
        raise ValueError(
            "Social-to-single-agent transfer is intentionally unsupported in this pipeline. "
            "Use discrete sources or implement a matched social target first."
        )


def discover_source_runs(base_dir, source_mode, source_task_family=None,
                         source_env=None, min_source_success=0.5):
    """Discover eligible source runs from suite-scoped metadata."""
    validate_transfer_request(source_mode)
    records = discover_runs(base_dir, allow_legacy=False)
    selected = {}
    skipped = []

    for record in sorted(records, key=lambda item: item['run_dir']):
        if record['mode'] != source_mode:
            continue
        if source_task_family and record['task_family'] != source_task_family:
            continue
        if source_env and record.get('source_env_name', record['env_name']) != source_env:
            continue

        metrics = record.get('metrics', {})
        success_rate = metrics.get('eval_success_rate')
        if success_rate is None:
            skipped.append({
                'seed': record['seed'],
                'checkpoint': record['checkpoint_path'],
                'reason': 'missing_eval_success_rate',
            })
            continue
        if success_rate < min_source_success:
            skipped.append({
                'seed': record['seed'],
                'checkpoint': record['checkpoint_path'],
                'reason': f'eval_success_rate_below_threshold:{success_rate:.3f}',
            })
            continue
        selected[record['seed']] = record

    return selected, skipped
