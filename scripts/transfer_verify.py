"""RQ2: same-family transfer of frozen manager+comm between corridor variants.

Takes a `final.pt` checkpoint produced by verify_hypotheses.py (single seed,
short run) and trains a fresh worker on a *different* corridor configuration
while freezing the source encoder + manager + comm channel.

Transfer from:   trained corridor (e.g. size=9, width=1)
Transfer to:     a shifted corridor (e.g. size=11, width=3)

RQ2 asks whether social-trained managers produce goal representations that
transfer better than discrete-trained ones. With our fast verify budget
(~15k timesteps) and same-family transfer (~30k timesteps) this gives a
directional answer in minutes rather than hours.

Usage:
    python scripts/transfer_verify.py \\
        --discrete-ckpt outputs/bus_verify/.../discrete/final.pt \\
        --social-ckpt   outputs/bus_verify/.../social/final.pt \\
        --target-size 11 --target-width 3 --timesteps 30000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)

from evaluate_transfer import TransferTrainer


def run_one(ckpt_path, source_mode, target_size, target_width,
            timesteps, device, out_dir, eval_episodes=20):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    trainer = TransferTrainer(
        checkpoint_path=ckpt_path,
        source_mode=source_mode,
        transfer_env=None,
        device=device,
        total_timesteps=timesteps,
        num_envs=4, num_steps=64,
        freeze_encoder=True, freeze_manager=True, freeze_comm=True,
        same_family=True,
        corridor_size=target_size,
        corridor_width=target_width,
        eval_episodes=eval_episodes,
    )
    results = trainer.train()
    elapsed = time.time() - t0
    np.save(os.path.join(out_dir, 'returns.npy'), np.array(results['returns']))
    if results.get('history') is not None:
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(results['history'], f, indent=2, default=float)
    metrics = {
        'source_mode': source_mode,
        'source_ckpt': ckpt_path,
        'target_size': target_size,
        'target_width': target_width,
        'timesteps': timesteps,
        'seconds': elapsed,
        'eval_mean_return': results['eval']['mean_return'],
        'eval_success_rate': results['eval']['success_rate'],
        'eval_std_return': results['eval']['std_return'],
        'training_final_return': float(
            np.mean(results['returns'][-100:])
        ) if results['returns'] else 0.0,
        'total_episodes': len(results['returns']),
    }
    with open(os.path.join(out_dir, 'transfer_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--discrete-ckpt', required=True)
    ap.add_argument('--social-ckpt', required=True)
    ap.add_argument('--target-size', type=int, default=11)
    ap.add_argument('--target-width', type=int, default=3)
    ap.add_argument('--timesteps', type=int, default=30000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-dir', default='outputs/transfer_verify')
    ap.add_argument('--eval-episodes', type=int, default=20)
    args = ap.parse_args()

    root = os.path.join(args.output_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(root, exist_ok=True)
    print(f'[transfer] discrete={args.discrete_ckpt}')
    print(f'[transfer] social  ={args.social_ckpt}')
    print(f'[transfer] target size={args.target_size} width={args.target_width}'
          f' timesteps={args.timesteps}')
    print(f'[transfer] output: {root}')

    results = []
    for source_mode, ckpt in (('discrete', args.discrete_ckpt),
                              ('social', args.social_ckpt)):
        print(f'\n[transfer] === source={source_mode} ===')
        out = os.path.join(root, source_mode)
        m = run_one(ckpt, source_mode, args.target_size, args.target_width,
                    args.timesteps, args.device, out,
                    eval_episodes=args.eval_episodes)
        results.append(m)
        print(f'[transfer] {source_mode} done in {m["seconds"]:.1f}s; '
              f'eval_return={m["eval_mean_return"]:.3f} '
              f'success={m["eval_success_rate"]:.2f}')

    print()
    print('=' * 72)
    print(f'TRANSFER RESULTS  target=SingleAgentCorridor'
          f'-S{args.target_size}-W{args.target_width}')
    print('=' * 72)
    header = (f'{"source":<10}{"final_return":>16}{"eval_return":>14}'
              f'{"eval_success":>14}')
    print(header)
    print('-' * len(header))
    for m in results:
        line = (f'{m["source_mode"]:<10}'
                f'{m["training_final_return"]:>16.3f}'
                f'{m["eval_mean_return"]:>14.3f}'
                f'{m["eval_success_rate"]:>14.2f}')
        print(line)
    print('=' * 72)

    by = {m['source_mode']: m for m in results}
    if 'discrete' in by and 'social' in by:
        d = by['discrete']['eval_mean_return']
        s = by['social']['eval_mean_return']
        gap = s - d
        verdict = ('social > discrete' if gap > 0.02
                   else 'discrete > social' if gap < -0.02
                   else 'comparable')
        print()
        print(f'RQ2 verdict (same-family transfer): '
              f'social_return={s:.3f}  discrete_return={d:.3f}  '
              f'gap={gap:+.3f}  -> {verdict}')
        print('Note: single seed, short budgets -- directional only.')

    with open(os.path.join(root, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[transfer] results: {root}')


if __name__ == '__main__':
    main()
