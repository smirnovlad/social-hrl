"""Pull wandb history for a group to local JSONs.

Usage:
    pip install wandb pandas
    # ensure ~/.netrc has machine api.wandb.ai login user password <key>
    # (or set WANDB_API_KEY)
    python pull_history.py --group mini_sweep-ts100000-20260421-232233 \\
        --keys mean_return,goal_msg_entropy,goal_space_coverage,gumbel_tau,comm_recon_loss \\
        --out wandb_history

Each run becomes wandb_history/<run-name>.json with shape:
    {"name": ..., "mode": ..., "seed": ..., "history": {key: [[step, val], ...]}}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', default='mbzuai-research')
    ap.add_argument('--project', default='social-hrl')
    ap.add_argument('--group', required=True)
    ap.add_argument('--keys', required=True,
                    help='comma-separated metric names')
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--out', default='wandb_history')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keys = [k.strip() for k in args.keys.split(',') if k.strip()]
    api = wandb.Api(timeout=args.timeout)
    runs = list(api.runs(f'{args.entity}/{args.project}',
                         filters={'group': args.group}, per_page=200))
    print(f'[pull] {len(runs)} runs in group={args.group}', flush=True)

    for i, run in enumerate(runs, 1):
        name = run.name
        out_path = os.path.join(args.out, f'{name}.json')
        if os.path.exists(out_path):
            print(f'[{i}/{len(runs)}] {name}: cached', flush=True)
            continue
        history = {}
        for key in keys:
            t0 = time.time()
            for attempt in range(3):
                try:
                    df = run.history(keys=[key], samples=args.samples,
                                     pandas=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f'  {key}: GIVING UP after 3 tries: {e}',
                              flush=True)
                        df = None
                        break
                    time.sleep(2 ** attempt)
            if df is None or len(df) == 0:
                print(f'  {key}: empty in {time.time() - t0:.1f}s', flush=True)
                continue
            steps = (df['_step'] if '_step' in df.columns
                     else df.index).tolist()
            if key not in df.columns:
                continue
            vals = df[key].tolist()
            pairs = [[int(s), float(v)] for s, v in zip(steps, vals)
                     if v == v]  # drop NaN
            history[key] = pairs
            print(f'  {key}: {len(pairs)} pts in {time.time() - t0:.1f}s',
                  flush=True)

        head = name.split('-', 1)[0]
        record = {
            'name': name,
            'group': args.group,
            'mode': head,
            'history': history,
        }
        with open(out_path, 'w') as f:
            json.dump(record, f)
        print(f'[{i}/{len(runs)}] {name}: wrote {out_path}', flush=True)

    print('[pull] done')


if __name__ == '__main__':
    main()
