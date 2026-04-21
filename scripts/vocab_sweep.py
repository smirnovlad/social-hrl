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
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--corridor-size', type=int, default=9)
    ap.add_argument('--corridor-width', type=int, default=1,
                    help='Default 1 = single-file (stress), maximize coord pressure.')
    ap.add_argument('--listener-reward-coef', type=float, default=0.5)
    ap.add_argument('--bus', action='store_true',
                    help='Enable shared-bus env (solo=0.05, shared=0.0).')
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
    for K, L in VOCAB_GRID:
        for mode in MODES:
            tag = f'K{K}_L{L}__{mode}'
            print(f'\n[vocab] === {tag} ===')
            out = os.path.join(root, tag)
            # Social gets 2x timesteps to amortize 2-agent training overhead.
            ts = args.timesteps * 2 if mode == 'social' else args.timesteps
            t0 = time.time()
            try:
                m = run_mode(
                    mode, ts, args.corridor_size, args.seed, out, args.device,
                    corridor_width=args.corridor_width,
                    listener_reward_coef=args.listener_reward_coef,
                    bus_cost_solo=bus_cost_solo, bus_cost_shared=0.0,
                    vocab_size=K, message_length=L,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'[vocab] {tag} FAILED: {e}')
                continue
            dt = time.time() - t0
            m['K'] = K
            m['L'] = L
            m['tag'] = tag
            rows.append(m)
            print(f'[vocab] {tag} done in {dt:.1f}s; '
                  f'coverage={m.get("goal_space_coverage")} '
                  f'mi={m.get("mutual_information")} '
                  f'recon={m.get("comm_recon_loss")}')

    print()
    print('=' * 96)
    print(f'VOCAB SWEEP RESULTS  seed={args.seed}  timesteps={args.timesteps}  '
          f'bus={args.bus}')
    print('=' * 96)
    header = (f'{"K":>4}{"L":>4}  {"mode":<10}'
              f'{"coverage":>10}{"gv_std":>10}{"MI":>8}{"listener":>10}'
              f'{"topsim":>10}{"recon":>8}{"temp_ext":>10}')
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda x: (x['K'] * x['L'], x['mode'])):
        line = f'{r["K"]:>4}{r["L"]:>4}  {r["mode"]:<10}'
        for k in ['goal_space_coverage', 'goal_vector_std', 'mutual_information',
                  'listener_accuracy', 'topographic_similarity',
                  'comm_recon_loss', 'temporal_extent']:
            v = r.get(k)
            if v is None:
                line += f'{"n/a":>10}' if k != 'mutual_information' else f'{"n/a":>8}'
            else:
                width = 8 if k in ('mutual_information', 'comm_recon_loss') else 10
                line += f'{v:>{width}.3f}'
        print(line)
    print('=' * 96)

    # RQ3 directional verdict: do social's margins over discrete grow at
    # tighter bottlenecks?
    print()
    print('RQ3 directional read (social - discrete on goal_space_coverage):')
    print('-' * 78)
    by_kl = {}
    for r in rows:
        by_kl.setdefault((r['K'], r['L']), {})[r['mode']] = r.get('goal_space_coverage')
    for (K, L) in sorted(by_kl.keys(), key=lambda x: x[0] * x[1]):
        d = by_kl[(K, L)].get('discrete')
        s = by_kl[(K, L)].get('social')
        if d is None or s is None:
            continue
        kl = K * L
        gap = s - d
        print(f'  K={K:>2} L={L}  K*L={kl:>5}   discrete={d:.3f}  social={s:.3f}  '
              f'gap={gap:+.3f}')
    print('-' * 78)

    # RQ3 compositionality read: does topographic similarity scale with
    # bottleneck tightness? Compositional pressure should peak at small K*L.
    print()
    print('RQ3 compositionality read (topographic_similarity by bottleneck size):')
    print('-' * 78)
    by_kl_topsim = {}
    for r in rows:
        by_kl_topsim.setdefault((r['K'], r['L']), {})[r['mode']] = \
            r.get('topographic_similarity')
    for (K, L) in sorted(by_kl_topsim.keys(), key=lambda x: x[0] * x[1]):
        d = by_kl_topsim[(K, L)].get('discrete')
        s = by_kl_topsim[(K, L)].get('social')
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
