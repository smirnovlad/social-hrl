"""RQ2 multi-seed transfer: aggregate transfer_verify across seeds.

Reuses checkpoints produced by `scripts/mini_sweep.py` (3 seeds x 4 modes;
we only need discrete and social). For each seed, runs same-family transfer
from the trained corridor (size 9, width 1) to a shifted corridor
(size 11, width 3), then reports mean +/- std across seeds.

Usage:
    # After mini_sweep has been run at least once:
    python scripts/transfer_multiseed.py

    # Custom seeds / timesteps:
    SEEDS="42 7 123" TIMESTEPS=20000 python scripts/transfer_multiseed.py
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

SEEDS = [int(s) for s in os.environ.get('SEEDS', '42 7 123').split()]
TIMESTEPS = int(os.environ.get('TIMESTEPS', '20000'))
TARGET_SIZE = int(os.environ.get('TARGET_SIZE', '11'))
TARGET_WIDTH = int(os.environ.get('TARGET_WIDTH', '3'))
MINI_SWEEP_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'mini_sweep',
)
OUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'transfer_multiseed', time.strftime('%Y%m%d-%H%M%S'),
)
os.makedirs(OUT_ROOT, exist_ok=True)


def find_ckpt(seed, mode):
    """Find the most recent final.pt for a given seed + mode from mini_sweep."""
    pattern = os.path.join(MINI_SWEEP_ROOT, f'seed-{seed}', '*', mode, 'final.pt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


print(f'[transfer-ms] seeds={SEEDS} timesteps={TIMESTEPS} '
      f'target=S{TARGET_SIZE}-W{TARGET_WIDTH}')
print(f'[transfer-ms] output: {OUT_ROOT}')

all_results = []
t_start = time.time()
for seed in SEEDS:
    disc_ckpt = find_ckpt(seed, 'discrete')
    soc_ckpt = find_ckpt(seed, 'social')
    if not disc_ckpt or not soc_ckpt:
        print(f'[transfer-ms] seed={seed} MISSING checkpoints '
              f'(discrete={bool(disc_ckpt)}, social={bool(soc_ckpt)}) '
              f'-- run scripts/mini_sweep.py first.')
        continue
    out = os.path.join(OUT_ROOT, f'seed-{seed}')
    os.makedirs(out, exist_ok=True)
    print(f'[transfer-ms] seed={seed} start', flush=True)
    cmd = [
        sys.executable, 'scripts/transfer_verify.py',
        '--discrete-ckpt', disc_ckpt,
        '--social-ckpt',   soc_ckpt,
        '--target-size', str(TARGET_SIZE),
        '--target-width', str(TARGET_WIDTH),
        '--timesteps', str(TIMESTEPS),
        '--output-dir', out,
    ]
    t0 = time.time()
    rc = subprocess.run(cmd, capture_output=True, text=True)
    print(f'[transfer-ms] seed={seed} done in {time.time() - t0:.1f}s '
          f'(rc={rc.returncode})', flush=True)
    if rc.returncode != 0:
        print(rc.stdout[-1500:])
        print(rc.stderr[-1500:])
        continue
    subs = sorted(os.listdir(out))
    run_dir = os.path.join(out, subs[-1])
    with open(os.path.join(run_dir, 'summary.json')) as f:
        per_seed = json.load(f)
    for entry in per_seed:
        entry['seed'] = seed
    all_results.extend(per_seed)


# Aggregate per-source-mode
KEYS = ['eval_mean_return', 'eval_success_rate', 'training_final_return',
        'eval_std_return']
agg = defaultdict(lambda: defaultdict(list))
for r in all_results:
    mode = r['source_mode']
    for k in KEYS:
        v = r.get(k)
        if v is not None:
            agg[mode][k].append(float(v))


print()
print('=' * 80)
print(f'MULTI-SEED TRANSFER RESULTS  (seeds={SEEDS}, {TIMESTEPS} ts, '
      f'target=S{TARGET_SIZE}-W{TARGET_WIDTH})')
print('=' * 80)
header = f'{"source":<10}' + ''.join(f'{k:>20}' for k in KEYS)
print(header)
print('-' * len(header))
for mode in ['discrete', 'social']:
    line = f'{mode:<10}'
    for k in KEYS:
        vals = agg[mode].get(k, [])
        if not vals:
            line += f'{"n/a":>20}'
        else:
            m, s = np.mean(vals), np.std(vals)
            line += f'{f"{m:+.3f} +/- {s:.3f}":>20}'
    print(line)
print('=' * 80)


# Noise-aware verdict on the social-vs-discrete eval_return gap
dv = agg['discrete'].get('eval_mean_return', [])
sv = agg['social'].get('eval_mean_return', [])
if dv and sv:
    dm, ds = np.mean(dv), np.std(dv)
    sm, ss = np.mean(sv), np.std(sv)
    gap = sm - dm
    noise = np.sqrt(ds ** 2 + ss ** 2)
    if abs(gap) > 3 * noise:
        verdict = 'LARGE'
    elif abs(gap) > noise:
        verdict = 'NOTABLE'
    else:
        verdict = 'WITHIN NOISE'
    print()
    print(f'RQ2 verdict (eval_return): '
          f'social={sm:+.3f}+/-{ss:.3f}  discrete={dm:+.3f}+/-{ds:.3f}')
    print(f'  gap (social - discrete) = {gap:+.3f} (combined noise {noise:.3f})  '
          f'-> {verdict}')


summary = {
    'seeds': SEEDS,
    'timesteps': TIMESTEPS,
    'target_size': TARGET_SIZE,
    'target_width': TARGET_WIDTH,
    'per_seed': all_results,
    'aggregate': {mode: {k: {'mean': float(np.mean(v)) if v else None,
                             'std':  float(np.std(v))  if v else None,
                             'n':    len(v),
                             'values': v}
                        for k, v in mode_map.items()}
                  for mode, mode_map in agg.items()},
}
with open(os.path.join(OUT_ROOT, 'aggregated_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n[transfer-ms] aggregated summary -> '
      f'{os.path.join(OUT_ROOT, "aggregated_summary.json")}')
print(f'[transfer-ms] total time: {time.time() - t_start:.1f}s')
