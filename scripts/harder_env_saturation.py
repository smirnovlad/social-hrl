"""Does the discrete-bottleneck goal-space saturation hold on harder envs?

Runs single-agent continuous vs discrete on KeyCorridor-S3R2 and MultiRoom-N6
at moderate budgets (~30k timesteps). Reports goal_space_coverage and
goal_vector_std after training.

Interpretation:
  - If discrete saturates at coverage ~1.0 on both, the bottleneck is a universal
    regularizer at this scale and Option 6's social hypothesis must be tested
    on a genuinely compositional env (e.g. BabyAI).
  - If discrete coverage stays well below 1.0 on either env, we've found a
    regime where social pressure could plausibly add value on top.
"""
import argparse
import json
import os
import random
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

from algos.hrl_trainer import HRLTrainer
from analysis.goal_metrics import compute_all_metrics


ENVS = [
    ('MiniGrid-KeyCorridorS3R2-v0', 80),
    ('MiniGrid-MultiRoom-N6-v0', 150),
]
MODES = ['continuous', 'discrete']


def build_cfg(env_name, max_steps, total_timesteps, mode, seed):
    goal_period = 10
    return {
        'env': {
            'name': env_name,
            'max_steps': max_steps,
            'fully_observable': False,
            'corridor_size': 11, 'corridor_width': 3,
            'asymmetric_info': False, 'rendezvous_bonus': 0.0, 'num_obstacles': 0,
        },
        'encoder': {'channels': [16, 32, 64], 'hidden_dim': 64},
        'manager': {
            'goal_dim': 16, 'hidden_dim': 64, 'goal_period': goal_period,
            'use_option_critic': False,
            'option_critic': {'num_options': 8, 'termination_reg': 0.01},
        },
        'worker': {
            'hidden_dim': 64, 'intrinsic_reward_coef': 0.1, 'extrinsic_reward_coef': 1.0,
            'intrinsic_anneal': False, 'intrinsic_anneal_steps': 500_000,
            'intrinsic_warmup_steps': 0,
        },
        'communication': {
            'vocab_size': 10, 'message_length': 3,
            'tau_start': 1.0, 'tau_end': 0.3,
            'tau_anneal_steps': max(total_timesteps // 2, 2000),
            'listener_reward_coef': 0.0, 'ablation_mode': False,
        },
        'ppo': {
            'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2,
            'entropy_coef': 0.01, 'value_coef': 0.5, 'max_grad_norm': 0.5,
            'num_envs': 4, 'num_steps': 64, 'num_minibatches': 2, 'update_epochs': 3,
            'total_timesteps': total_timesteps, 'anneal_lr': True,
        },
        'sac': {'enabled': False},
        'multi_agent': {'num_agents': 1, 'shared_critic': True,
                        'comm_reward_coef': 0.0, 'listener_reward_coef': 0.0},
        'experiment': {'seed': seed, 'device': 'cpu',
                       'log_interval': 100, 'save_interval': 10_000_000,
                       'eval_episodes': 10, 'wandb_project': None,
                       'wandb_entity': None, 'output_dir': 'outputs'},
    }


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run(env_name, max_steps, mode, total_timesteps, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seed_all(seed)
    cfg = build_cfg(env_name, max_steps, total_timesteps, mode, seed)
    t0 = time.time()
    trainer = HRLTrainer(cfg, mode=mode, device='cpu', use_corridor=False)
    log = os.path.join(out_dir, 'train.log')
    with open(log, 'w') as f, redirect_stdout(f):
        results = trainer.train(output_dir=out_dir)
    dt = time.time() - t0

    metrics = compute_all_metrics(
        messages=None, vocab_size=cfg['communication']['vocab_size'],
        message_length=cfg['communication']['message_length'],
        decoded_goals=results.get('decoded_goals'),
    )
    metrics['env'] = env_name
    metrics['mode'] = mode
    metrics['seconds'] = dt
    metrics['total_timesteps'] = total_timesteps
    if results.get('eval'):
        metrics['eval_success_rate'] = results['eval']['success_rate']
        metrics['eval_mean_return'] = results['eval']['mean_return']
    metrics['final_return_mean'] = (
        float(np.mean(results['returns'][-50:])) if results['returns'] else 0.0
    )
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timesteps', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default='outputs/harder_envs')
    args = ap.parse_args()

    root = os.path.join(args.output_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(root, exist_ok=True)
    print(f'[harder] env sweep timesteps={args.timesteps} seed={args.seed}')
    print(f'[harder] output: {root}')

    all_metrics = []
    for env_name, max_steps in ENVS:
        for mode in MODES:
            tag = f'{env_name.replace("MiniGrid-", "").replace("-v0", "")}__{mode}'
            print(f'\n[harder] === {tag} ===')
            out = os.path.join(root, tag)
            try:
                m = run(env_name, max_steps, mode, args.timesteps,
                        args.seed, out)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'[harder] {tag} FAILED: {e}')
                continue
            print(f'[harder] {tag} done in {m["seconds"]:.1f}s; '
                  f'coverage={m.get("goal_space_coverage")} '
                  f'std={m.get("goal_vector_std")} '
                  f'success={m.get("eval_success_rate")}')
            all_metrics.append(m)

    print()
    print('=' * 84)
    print('HARDER ENVS SATURATION CHECK')
    print('=' * 84)
    header = f'{"env":<26}{"mode":<14}{"coverage":>12}{"std":>10}{"success":>12}'
    print(header)
    print('-' * len(header))
    for m in all_metrics:
        env = m['env'].replace('MiniGrid-', '').replace('-v0', '')
        line = f'{env:<26}{m["mode"]:<14}'
        cov = m.get('goal_space_coverage')
        std = m.get('goal_vector_std')
        sr = m.get('eval_success_rate', 0.0)
        line += f'{cov:>12.3f}' if cov is not None else f'{"n/a":>12}'
        line += f'{std:>10.3f}' if std is not None else f'{"n/a":>10}'
        line += f'{sr:>12.2f}'
        print(line)
    print('=' * 84)

    # Decide saturation verdict
    disc_covs = {m['env']: m.get('goal_space_coverage') for m in all_metrics if m['mode'] == 'discrete'}
    print()
    for env, cov in disc_covs.items():
        env_s = env.replace('MiniGrid-', '').replace('-v0', '')
        if cov is None:
            print(f'  {env_s}: no discrete coverage')
            continue
        if cov > 0.95:
            verdict = 'SATURATED  -> bottleneck maxed out, no room for social to help here'
        elif cov > 0.7:
            verdict = 'NEAR-SATURATED  -> marginal room for social'
        else:
            verdict = 'UNSATURATED  -> regime where social pressure could plausibly help'
        print(f'  {env_s}  discrete_coverage={cov:.3f}  -> {verdict}')

    with open(os.path.join(root, 'summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=float)
    print(f'\n[harder] All results in: {root}')


if __name__ == '__main__':
    main()
