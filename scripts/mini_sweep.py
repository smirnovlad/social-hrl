"""3-seed mini-sweep: 6-way comparison across all modes.

Default modes: flat, continuous, discrete, social, lola, maddpg. Rationale
for including flat + continuous: the H1 story ("discrete bottleneck prevents
goal collapse") requires continuous as the collapse reference and flat as
the no-hierarchy baseline for the return curve.

Runs verify_hypotheses (--stress, --bus) for each (seed, mode) and
aggregates the main metrics with mean and std across seeds.

Usage:
    python scripts/mini_sweep.py                  # default 3 seeds, 6 modes
    MODES="discrete social"  python scripts/mini_sweep.py  # subset via env var
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

SEEDS = [42, 7, 123]
MODES = os.environ.get('MODES',
                       'flat continuous discrete social lola maddpg').split()
TIMESTEPS = int(os.environ.get('TIMESTEPS', '15000'))
USE_BUS = os.environ.get('USE_BUS', '1') == '1'
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'outputs', 'mini_sweep')
os.makedirs(ROOT, exist_ok=True)

run_dirs = []
t_start = time.time()
sweep_group = os.environ.get('WANDB_RUN_GROUP',
                             f'mini_sweep-ts{TIMESTEPS}-{time.strftime("%Y%m%d-%H%M%S")}')
for seed in SEEDS:
    out = os.path.join(ROOT, f'seed-{seed}')
    os.makedirs(out, exist_ok=True)
    print(f'[sweep] seed={seed} -> {out}', flush=True)
    cmd = [
        sys.executable, 'scripts/verify_hypotheses.py',
        '--stress',
        '--modes', *MODES,
        '--timesteps', str(TIMESTEPS),
        '--seed', str(seed),
        '--device', 'cpu',
        '--output-dir', out,
    ]
    if USE_BUS:
        cmd.append('--bus')
    env = {**os.environ, 'WANDB_RUN_GROUP': sweep_group}
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(f'[sweep] seed={seed} done in {time.time() - t0:.1f}s '
          f'(rc={result.returncode})', flush=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(1)
    # The script creates a timestamped subdir; find the newest
    subs = sorted(os.listdir(out))
    run_dirs.append(os.path.join(out, subs[-1]))

# Aggregate
KEYS = [
    'goal_space_coverage',
    'goal_vector_std',
    'entropy',
    'coverage',
    'mutual_information',
    'listener_accuracy',
    'topographic_similarity',
    'comm_recon_loss',
    'comm_ablation_delta',
    'temporal_extent',
    'eval_success_rate',
    'final_return_mean',
]

agg = defaultdict(lambda: defaultdict(list))  # mode -> key -> [values]
for d in run_dirs:
    with open(os.path.join(d, 'summary.json')) as f:
        runs = json.load(f)
    for r in runs:
        mode = r['mode']
        for k in KEYS:
            v = r.get(k)
            if v is not None:
                agg[mode][k].append(float(v))

print()
print('=' * 120)
print(f'MINI-SWEEP RESULTS  (seeds={SEEDS}, stress config, bus={USE_BUS}, '
      f'{TIMESTEPS} timesteps)')
print('=' * 120)
col_w = 22
header = f'{"metric":<24}' + ''.join(f'{m:>{col_w}}' for m in MODES)
print(header)
print('-' * len(header))
for k in KEYS:
    line = f'{k:<24}'
    for mode in MODES:
        vals = agg[mode].get(k, [])
        if not vals:
            line += f'{"n/a":>{col_w}}'
        else:
            mean, std = np.mean(vals), np.std(vals)
            line += f'{f"{mean:7.3f} +/- {std:6.3f}":>{col_w}}'
    print(line)
print('=' * 120)

# Aggregated summary file for the results doc
summary_path = os.path.join(ROOT, 'aggregated_summary.json')
summary = {}
for mode in MODES:
    summary[mode] = {}
    for k in KEYS:
        vals = agg[mode].get(k, [])
        if vals:
            summary[mode][k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'n': len(vals),
                'values': [float(v) for v in vals],
            }
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[sweep] aggregated summary -> {summary_path}')

# Robustness read on the key deltas for RQ1.
def _gap(a, b, label_a, label_b, metric='goal_space_coverage', higher_is_better=True):
    va = agg[a].get(metric, [])
    vb = agg[b].get(metric, [])
    if not va or not vb:
        return
    am, asd = np.mean(va), np.std(va)
    bm, bsd = np.mean(vb), np.std(vb)
    gap = am - bm
    noise = np.sqrt(asd ** 2 + bsd ** 2)
    direction = 'positive favors ' + label_a if higher_is_better else \
                'negative favors ' + label_a
    if abs(gap) > 3 * noise:
        verdict = 'LARGE'
    elif abs(gap) > noise:
        verdict = 'NOTABLE'
    else:
        verdict = 'WITHIN NOISE'
    print(f'  {metric}: {label_a} {am:+.3f}+/-{asd:.3f} vs '
          f'{label_b} {bm:+.3f}+/-{bsd:.3f}   '
          f'gap={gap:+.3f} (noise {noise:.3f})  -> {verdict}')

print()
print('Pairwise reads on goal_space_coverage:')
if 'discrete' in MODES and 'social' in MODES:
    _gap('discrete', 'social', 'discrete', 'social')
if 'discrete' in MODES and 'lola' in MODES:
    _gap('discrete', 'lola', 'discrete', 'lola')
if 'lola' in MODES and 'social' in MODES:
    _gap('lola', 'social', 'lola', 'social')

print()
print('Pairwise reads on final_return_mean (higher=better):')
for a in MODES:
    for b in MODES:
        if a >= b:
            continue
        _gap(a, b, a, b, metric='final_return_mean')

print(f'\n[sweep] total time: {time.time() - t_start:.1f}s')
print(f'[sweep] run dirs: {run_dirs}')
