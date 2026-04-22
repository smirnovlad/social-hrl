"""RQ3: vocabulary size vs goal representation quality.

Sweeps K (vocab_size) x L (message_length) on discrete and social modes.
For each (K, L, mode) cell, runs one short training run and collects:
  - goal_space_coverage  (collapse)
  - goal_vector_std
  - mutual_information   (message <-> state)
  - listener_accuracy    (probe)
  - comm_recon_loss      (bottleneck tightness)
  - temporal_extent

Small K + small L should force the most compositional pressure. If RQ3 is
real we expect: at aggressive bottlenecks (small K*L), social > discrete on
goal_space_coverage; at loose bottlenecks (large K*L) the two converge.

Usage:
    python scripts/vocab_sweep.py --timesteps 12000 --seed 42
    python scripts/vocab_sweep.py --bus --timesteps 15000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after sys.path so the harness resolves.
from verify_hypotheses import run_mode


VOCAB_GRID = [
    (3, 1),    # K=3,  L=1: very aggressive bottleneck (3 messages total)
    (3, 3),    # K=3,  L=3: 27 messages
    (10, 1),   # K=10, L=1: 10 messages
    (10, 3),   # K=10, L=3: 1000 messages (default config)
    (25, 3),   # K=25, L=3: loose bottleneck
]
MODES = ['discrete', 'social']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timesteps', type=int, default=12000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42],
                    help='Seed list; pass multiple for cross-seed std.')
    ap.add_argument('--corridor-size', type=int, default=9)
    ap.add_argument('--corridor-width', type=int, default=1,
                    help='Default 1 = single-file (stress), maximize coord pressure.')
    ap.add_argument('--listener-reward-coef', type=float, default=0.5)
    ap.add_argument('--bus', action='store_true',
                    help='Enable shared-bus env (solo=0.05, shared=0.0).')
    ap.add_argument('--bus-window', type=int, default=0,
                    help='Strict-bus arrival window (>0 = bus_strict).')
    ap.add_argument('--randomize-goals', action='store_true')
    ap.add_argument('--mutual-goal-blind', action='store_true')
    ap.add_argument('--output-dir', default='outputs/vocab_sweep')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    bus_cost_solo = 0.05 if args.bus else 0.0
    root = os.path.join(args.output_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(root, exist_ok=True)
    print(f'[vocab] grid={VOCAB_GRID} modes={MODES} timesteps={args.timesteps}')
    print(f'[vocab] bus={args.bus} corridor_width={args.corridor_width}')
    print(f'[vocab] output: {root}')

    rows = []
    t_start = time.time()
    os.environ.setdefault(
        'WANDB_RUN_GROUP',
        f'vocab_sweep-ts{args.timesteps}-{time.strftime("%Y%m%d-%H%M%S")}')
    for K, L in VOCAB_GRID:
        for mode in MODES:
            for seed in args.seeds:
                tag = f'K{K}_L{L}__{mode}__seed{seed}'
                os.environ['WANDB_TAGS'] = f'vocab_sweep,K{K},L{L}'
                print(f'\n[vocab] === {tag} ===')
                out = os.path.join(root, tag)
                # Social gets 2x timesteps to amortize 2-agent training overhead.
                ts = args.timesteps * 2 if mode == 'social' else args.timesteps
                t0 = time.time()
                try:
                    m = run_mode(
                        mode, ts, args.corridor_size, seed, out, args.device,
                        corridor_width=args.corridor_width,
                        listener_reward_coef=args.listener_reward_coef,
                        bus_cost_solo=bus_cost_solo, bus_cost_shared=0.0,
                        vocab_size=K, message_length=L,
                        bus_window=args.bus_window,
                        randomize_goals=args.randomize_goals,
                        mutual_goal_blind=args.mutual_goal_blind,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f'[vocab] {tag} FAILED: {e}')
                    continue
                dt = time.time() - t0
                m['K'] = K
                m['L'] = L
                m['seed'] = seed
                m['tag'] = tag
                rows.append(m)
                print(f'[vocab] {tag} done in {dt:.1f}s; '
                      f'coverage={m.get("goal_space_coverage")} '
                      f'joint={m.get("goal_space_coverage_joint")} '
                      f'abl_scr={m.get("comm_ablation_delta_scramble")}')

    # Aggregate per (K, L, mode) across seeds.
    from collections import defaultdict
    import numpy as _np
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['K'], r['L'], r['mode'])
        for mk in ('goal_space_coverage', 'goal_space_coverage_a',
                   'goal_space_coverage_b', 'goal_space_coverage_joint',
                   'goal_vector_std', 'mutual_information',
                   'listener_accuracy', 'topographic_similarity',
                   'comm_recon_loss', 'temporal_extent',
                   'comm_ablation_delta', 'comm_ablation_delta_scramble'):
            v = r.get(mk)
            if v is not None:
                agg[key][mk].append(float(v))

    def _ms(vals):
        if not vals:
            return None, None
        return float(_np.mean(vals)), float(_np.std(vals))

    print()
    print('=' * 120)
    print(f'VOCAB SWEEP RESULTS  seeds={args.seeds}  timesteps={args.timesteps}  '
          f'bus={args.bus} bus_window={args.bus_window} '
          f'rand_goals={args.randomize_goals} mutual_blind={args.mutual_goal_blind}')
    print('=' * 120)
    header = (f'{"K":>4}{"L":>4}  {"mode":<10}'
              f'{"coverage":>12}{"joint":>12}{"MI":>10}{"listener":>10}'
              f'{"recon":>10}{"abl_z":>10}{"abl_scr":>10}')
    print(header)
    print('-' * len(header))
    for K, L in VOCAB_GRID:
        for mode in MODES:
            key = (K, L, mode)
            line = f'{K:>4}{L:>4}  {mode:<10}'
            for mk in ('goal_space_coverage', 'goal_space_coverage_joint',
                       'mutual_information', 'listener_accuracy',
                       'comm_recon_loss', 'comm_ablation_delta',
                       'comm_ablation_delta_scramble'):
                m, s = _ms(agg[key].get(mk, []))
                if m is None:
                    line += f'{"n/a":>10}' if mk in ('mutual_information',
                                                     'comm_recon_loss',
                                                     'listener_accuracy',
                                                     'comm_ablation_delta',
                                                     'comm_ablation_delta_scramble') \
                           else f'{"n/a":>12}'
                else:
                    if mk in ('goal_space_coverage', 'goal_space_coverage_joint'):
                        line += f'{f"{m:.3f}+/-{s:.2f}":>12}'
                    else:
                        line += f'{m:>10.3f}'
            print(line)
    print('=' * 120)

    print()
    print('RQ3 directional read (mean across seeds):')
    print('  legacy pooled (per-agent-pooled coverage) and preferred joint view')
    print('-' * 96)
    header2 = (f'{"K":>3}{"L":>3}{"K*L":>6}   '
               f'{"disc pooled":>14}  {"soc pooled":>14}  '
               f'{"soc joint":>14}  {"soc abl_scr":>14}')
    print(header2)
    for K, L in VOCAB_GRID:
        d_pool, _ = _ms(agg[(K, L, 'discrete')].get('goal_space_coverage', []))
        s_pool, _ = _ms(agg[(K, L, 'social')].get('goal_space_coverage', []))
        s_joint, _ = _ms(agg[(K, L, 'social')].get('goal_space_coverage_joint', []))
        s_scr, _ = _ms(agg[(K, L, 'social')].get('comm_ablation_delta_scramble', []))
        def _f(x):
            return f'{x:.3f}' if x is not None else 'n/a'
        print(f'  {K:>2} {L:>2} {K*L:>5}   '
              f'{_f(d_pool):>14}  {_f(s_pool):>14}  '
              f'{_f(s_joint):>14}  {_f(s_scr):>14}')
    print('-' * 96)

    # RQ3 compositionality read: does topographic similarity scale with
    # bottleneck tightness? Compositional pressure should peak at small K*L.
    print()
    print('RQ3 compositionality read (topographic_similarity by bottleneck size):')
    print('-' * 78)
    for K, L in VOCAB_GRID:
        d, _ = _ms(agg[(K, L, 'discrete')].get('topographic_similarity', []))
        s, _ = _ms(agg[(K, L, 'social')].get('topographic_similarity', []))
        d_str = f'{d:+.3f}' if d is not None else '  n/a '
        s_str = f'{s:+.3f}' if s is not None else '  n/a '
        print(f'  K={K:>2} L={L}  K*L={K*L:>5}   discrete_topsim={d_str}  '
              f'social_topsim={s_str}')
    print('-' * 78)
    print('(higher topsim = more compositional mapping between states and messages)')

    with open(os.path.join(root, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print(f'\n[vocab] total time: {time.time() - t_start:.1f}s')
    print(f'[vocab] results: {root}')


if __name__ == '__main__':
    main()
