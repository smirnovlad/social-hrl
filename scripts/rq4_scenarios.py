"""RQ4: which multi-agent scenarios provide strongest regularization?

Runs discrete vs social under four coordination scenarios and reports the
goal-collapse and comm-use metrics for each. Scenarios:

  - baseline       : no bus, no turn-taking (plain corridor)
  - bus            : shared bus resource cheaper when simultaneous
  - bus_strict     : bus + arrival-time window (must enter together)
  - turn_taking    : only one agent moves per step (alternating)

Single-seed per scenario by default (fast). Set SEEDS env var to override,
e.g. SEEDS="42 7 123" to get 3-seed aggregation.

Usage:
    python scripts/rq4_scenarios.py
    SEEDS="42 7 123" python scripts/rq4_scenarios.py
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np


SCENARIOS = {
    'baseline':    {'bus': False, 'bus_window': 0,  'turn_taking': False},
    'bus':         {'bus': True,  'bus_window': 0,  'turn_taking': False},
    'bus_strict':  {'bus': True,  'bus_window': 4,  'turn_taking': False},
    'turn_taking': {'bus': False, 'bus_window': 0,  'turn_taking': True},
}
MODES = ['discrete', 'social']
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42').split()]
TIMESTEPS = int(os.environ.get('TIMESTEPS', '12000'))
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'outputs', 'rq4_scenarios')
os.makedirs(ROOT, exist_ok=True)

TS = time.strftime('%Y%m%d-%H%M%S')
print(f'[rq4] scenarios={list(SCENARIOS)} seeds={SEEDS} timesteps={TIMESTEPS}')
print(f'[rq4] output: {os.path.join(ROOT, TS)}')

# scenario -> seed -> run_dir
run_map = defaultdict(dict)
t_start = time.time()
for scen_name, scen_cfg in SCENARIOS.items():
    for seed in SEEDS:
        out = os.path.join(ROOT, TS, scen_name, f'seed-{seed}')
        os.makedirs(out, exist_ok=True)
        cmd = [
            sys.executable, 'scripts/verify_hypotheses.py',
            '--stress',
            '--modes', *MODES,
            '--timesteps', str(TIMESTEPS),
            '--seed', str(seed),
            '--device', 'cpu',
            '--output-dir', out,
        ]
        if scen_cfg['bus']:
            cmd.append('--bus')
        if scen_cfg['bus_window'] > 0:
            cmd += ['--bus-window', str(scen_cfg['bus_window'])]
        if scen_cfg['turn_taking']:
            cmd.append('--turn-taking')
        t0 = time.time()
        print(f'[rq4] scen={scen_name} seed={seed} ...', flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        print(f'[rq4] scen={scen_name} seed={seed} done in {dt:.1f}s '
              f'(rc={result.returncode})', flush=True)
        if result.returncode != 0:
            print(result.stdout[-1500:])
            print(result.stderr[-1500:])
            continue
        subs = sorted(os.listdir(out))
        run_map[scen_name][seed] = os.path.join(out, subs[-1])


KEYS = [
    'goal_space_coverage',
    'goal_vector_std',
    'listener_accuracy',
    'topographic_similarity',
    'comm_ablation_delta',
    'final_return_mean',
]

# scenario -> mode -> key -> [values]
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for scen_name, seed_map in run_map.items():
    for seed, run_dir in seed_map.items():
        try:
            with open(os.path.join(run_dir, 'summary.json')) as f:
                runs = json.load(f)
        except FileNotFoundError:
            continue
        for r in runs:
            mode = r.get('mode')
            for k in KEYS:
                v = r.get(k)
                if v is not None:
                    agg[scen_name][mode][k].append(float(v))


print()
print('=' * 120)
print(f'RQ4 SCENARIO RESULTS (seeds={SEEDS}, {TIMESTEPS} timesteps)')
print('=' * 120)
header = f'{"scenario":<14}{"mode":<10}'
for k in KEYS:
    short = k.replace('goal_space_coverage', 'gs_cov') \
             .replace('goal_vector_std', 'gv_std') \
             .replace('listener_accuracy', 'listener') \
             .replace('topographic_similarity', 'topsim') \
             .replace('comm_ablation_delta', 'abl_d') \
             .replace('final_return_mean', 'ret')
    header += f'{short:>16}'
print(header)
print('-' * len(header))
for scen_name in SCENARIOS:
    for mode in MODES:
        line = f'{scen_name:<14}{mode:<10}'
        for k in KEYS:
            vals = agg[scen_name][mode].get(k, [])
            if not vals:
                line += f'{"n/a":>16}'
            else:
                m, s = np.mean(vals), np.std(vals)
                if len(vals) > 1:
                    line += f'{f"{m:+.3f}+/-{s:.2f}":>16}'
                else:
                    line += f'{m:>16.3f}'
        print(line)
    print()
print('=' * 120)

# RQ4 verdict: which scenario produces the largest social-over-discrete lift?
print()
print('RQ4 verdict (social - discrete gap per scenario):')
print('-' * 78)
for scen_name in SCENARIOS:
    dv = agg[scen_name]['discrete'].get('goal_space_coverage', [])
    sv = agg[scen_name]['social'].get('goal_space_coverage', [])
    if not dv or not sv:
        print(f'  {scen_name:<14}: missing data')
        continue
    gap_cov = np.mean(sv) - np.mean(dv)
    abl = agg[scen_name]['social'].get('comm_ablation_delta', [])
    abl_str = f'abl_delta={np.mean(abl):+.3f}' if abl else 'abl_delta=n/a'
    lis = agg[scen_name]['social'].get('listener_accuracy', [])
    lis_str = f'listener={np.mean(lis):.3f}' if lis else 'listener=n/a'
    print(f'  {scen_name:<14}: coverage_gap(social-discrete)={gap_cov:+.3f}  '
          f'{abl_str}   {lis_str}')
print('-' * 78)
print('Interpretation: a scenario "regularizes more" if social closes the '
      'coverage gap (less negative / more positive) AND comm_ablation_delta '
      'rises (channel carries load-bearing info).')

summary_path = os.path.join(ROOT, TS, 'aggregated_summary.json')
with open(summary_path, 'w') as f:
    json.dump({
        scen: {mode: {k: {'mean': float(np.mean(v)) if v else None,
                          'std':  float(np.std(v))  if v else None,
                          'n':    len(v),
                          'values': v}
                     for k, v in mode_map.items()}
               for mode, mode_map in scen_map.items()}
        for scen, scen_map in agg.items()
    }, f, indent=2)
print(f'\n[rq4] aggregated summary -> {summary_path}')
print(f'[rq4] total time: {time.time() - t_start:.1f}s')
