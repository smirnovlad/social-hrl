"""Fast hypothesis-verification harness for the social-HRL project.

Runs the minimum viable comparison to decide whether the core hypothesis
(multi-agent coordination pressure regularizes HRL goal representations)
is showing up *at all* in this codebase -- before the user spends weeks
of compute on full suites.

What it runs (single seed, tiny corridor, CPU/MPS-friendly):
  1. continuous  -- HRL with unconstrained goals in R^d.   Expected: collapse.
  2. discrete    -- HRL with Gumbel-Softmax bottleneck.    Expected: partial collapse prevention.
  3. social      -- Two agents + comm channel + asymmetric info.
                    Expected (if hypothesis holds): least collapse, most structured goals.

For each mode it prints:
  - eval_success_rate           (did hierarchy learn anything?)
  - goal_space_coverage         (fraction of 4^4 quantile cells in R^d that see a decoded goal)
  - goal_vector_std             (mean per-dim std of decoded goals)
  - entropy / coverage          (message-space diversity, when applicable)
  - temporal_extent             (steps before goals change)
  - comm_recon_loss             (how tight the bottleneck actually is)
  - comm_ablation_delta (social) (does the channel carry any useful info?)

It then prints a verdict table comparing collapse signals across modes.

Usage:
    python scripts/verify_hypotheses.py                 # all three modes
    python scripts/verify_hypotheses.py --modes social  # subset
    python scripts/verify_hypotheses.py --timesteps 8000 --quick
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
from algos.multi_agent_trainer import MultiAgentTrainer
from algos.lola_trainer import LolaMultiAgentTrainer
from analysis.goal_metrics import compute_all_metrics


DEFAULT_MODES = ['continuous', 'discrete', 'social']
SOCIAL_MODES = {'social', 'lola'}
MULTI_AGENT_MODES = {'social', 'lola', 'maddpg'}


def build_config(total_timesteps, corridor_size, mode, seed, corridor_width=3,
                 listener_reward_coef=0.0, bus_cost_solo=0.0,
                 bus_cost_shared=0.0, vocab_size=10, message_length=3,
                 bus_window=0, turn_taking=False):
    """Minimal config for a fast corridor run."""
    goal_period = 8
    num_envs = 4
    num_steps = 64
    cfg = {
        'env': {
            'name': 'corridor',
            'max_steps': 80,
            'fully_observable': False,
            'corridor_size': corridor_size,
            'corridor_width': corridor_width,
            'asymmetric_info': (mode in SOCIAL_MODES),
            'rendezvous_bonus': 0.0,
            'num_obstacles': 0,
            'bus_cost_solo': bus_cost_solo if mode in MULTI_AGENT_MODES else 0.0,
            'bus_cost_shared': bus_cost_shared if mode in MULTI_AGENT_MODES else 0.0,
            'bus_window': bus_window if mode in MULTI_AGENT_MODES else 0,
            'turn_taking': turn_taking if mode in MULTI_AGENT_MODES else False,
        },
        'encoder': {'channels': [16, 32, 64], 'hidden_dim': 64},
        'manager': {
            'goal_dim': 16,
            'hidden_dim': 64,
            'goal_period': goal_period,
            'use_option_critic': False,
            'option_critic': {'num_options': 8, 'termination_reg': 0.01},
        },
        'worker': {
            'hidden_dim': 64,
            'intrinsic_reward_coef': 0.1,
            'extrinsic_reward_coef': 1.0,
            'intrinsic_anneal': False,
            'intrinsic_anneal_steps': 500_000,
            'intrinsic_warmup_steps': 0,
        },
        'communication': {
            'vocab_size': vocab_size,
            'message_length': message_length,
            'tau_start': 1.0,
            'tau_end': 0.3,
            'tau_anneal_steps': max(total_timesteps // 2, 2000),
            'listener_reward_coef': 0.0,
            'ablation_mode': (mode in SOCIAL_MODES),
        },
        'ppo': {
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'num_envs': num_envs,
            'num_steps': num_steps,
            'num_minibatches': 2,
            'update_epochs': 3,
            'total_timesteps': total_timesteps,
            'anneal_lr': True,
        },
        'sac': {'enabled': False},
        'multi_agent': {
            'num_agents': 2,
            'shared_critic': True,
            'comm_reward_coef': 0.1,
            # Give the channel actual coordination pressure: agent A is
            # rewarded when agent B moves toward A's goal direction. Without
            # this, the comm channel has no reason to carry partner-useful
            # structure and H3 (comm_ablation_delta) stays near zero.
            'listener_reward_coef': listener_reward_coef if mode in SOCIAL_MODES else 0.0,
        },
        'lola': {
            'coef': 1.0,
            'inner_lr': 1e-3,
            'warmup_updates': 5,
        },
        'experiment': {
            'seed': seed,
            'device': 'cpu',
            'log_interval': 50,
            'save_interval': 10_000_000,
            'eval_episodes': 10,
            'wandb_project': None,
            'wandb_entity': None,
            'output_dir': 'outputs',
        },
    }
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_mode(mode, total_timesteps, corridor_size, seed, out_dir, device,
             corridor_width=3, listener_reward_coef=0.0,
             bus_cost_solo=0.0, bus_cost_shared=0.0,
             vocab_size=10, message_length=3,
             bus_window=0, turn_taking=False):
    """Run one mode end-to-end and return a dict of metrics."""
    os.makedirs(out_dir, exist_ok=True)
    seed_all(seed)
    cfg = build_config(total_timesteps, corridor_size, mode, seed,
                       corridor_width=corridor_width,
                       listener_reward_coef=listener_reward_coef,
                       bus_cost_solo=bus_cost_solo,
                       bus_cost_shared=bus_cost_shared,
                       vocab_size=vocab_size,
                       message_length=message_length,
                       bus_window=bus_window,
                       turn_taking=turn_taking)
    t0 = time.time()

    if mode == 'social':
        trainer = MultiAgentTrainer(cfg, device=device)
    elif mode == 'lola':
        trainer = LolaMultiAgentTrainer(cfg, device=device)
    elif mode == 'maddpg':
        from algos.maddpg_trainer import MaddpgTrainer
        trainer = MaddpgTrainer(cfg, device=device)
    else:
        trainer = HRLTrainer(cfg, mode=mode, device=device, use_corridor=True)

    log_path = os.path.join(out_dir, f'{mode}.log')
    with open(log_path, 'w') as f, redirect_stdout(f):
        results = trainer.train(output_dir=out_dir, wandb_run=None)
    elapsed = time.time() - t0

    # Flatten messages and states
    all_msgs = []
    if results.get('messages'):
        for batch in results['messages']:
            if isinstance(batch, np.ndarray):
                for m in batch:
                    all_msgs.append(m)
    all_states = []
    if results.get('states'):
        for batch in results['states']:
            if isinstance(batch, np.ndarray):
                for s in batch:
                    all_states.append(s)

    metrics = compute_all_metrics(
        messages=all_msgs if all_msgs else None,
        vocab_size=cfg['communication']['vocab_size'],
        message_length=cfg['communication']['message_length'],
        states=all_states if all_states else None,
        decoded_goals=results.get('decoded_goals'),
    )
    # Replace the flat-messages-based temporal_extent (which is measured across
    # interleaved per-env messages and reports ~1 regardless of behavior) with
    # the per-env-per-step extent computed inside the trainer.
    if results.get('temporal_extent_mean') is not None:
        metrics['temporal_extent'] = results['temporal_extent_mean']
    metrics['mode'] = mode
    metrics['seconds'] = elapsed
    metrics['total_timesteps'] = total_timesteps
    if results.get('eval'):
        metrics['eval_mean_return'] = results['eval']['mean_return']
        metrics['eval_success_rate'] = results['eval']['success_rate']
    if results.get('comm_ablation_eval'):
        ce = results['comm_ablation_eval']
        metrics['eval_with_comm'] = ce['with_comm']['mean_return']
        metrics['eval_without_comm'] = ce['without_comm']['mean_return']
        metrics['comm_ablation_delta'] = ce['delta']
    if results.get('recon_loss_mean') is not None:
        metrics['comm_recon_loss'] = results['recon_loss_mean']
    metrics['final_return_mean'] = (
        float(np.mean(results['returns'][-50:])) if results['returns'] else 0.0
    )

    with open(os.path.join(out_dir, f'{mode}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    return metrics


def fmt(x):
    if x is None:
        return '  n/a'
    if isinstance(x, float):
        return f'{x:7.3f}'
    return str(x)


def verdict_table(all_metrics):
    rows = [
        ('mode',                   'mode'),
        ('runtime (s)',            'seconds'),
        ('final_return_mean',      'final_return_mean'),
        ('eval_success_rate',      'eval_success_rate'),
        ('goal_space_coverage',    'goal_space_coverage'),  # collapse signal (goal space)
        ('goal_vector_std',        'goal_vector_std'),      # collapse signal (amplitude)
        ('message entropy',        'entropy'),              # collapse signal (message space)
        ('message coverage',       'coverage'),
        ('temporal_extent',        'temporal_extent'),
        ('mutual_information',     'mutual_information'),
        ('listener_accuracy',      'listener_accuracy'),
        ('topographic_similarity', 'topographic_similarity'),
        ('comm_recon_loss',        'comm_recon_loss'),
        ('comm_ablation_delta',    'comm_ablation_delta'),
    ]
    modes = [m['mode'] for m in all_metrics]
    print()
    print('=' * 78)
    print('VERIFICATION RESULTS')
    print('=' * 78)
    header = f'{"metric":<22}' + ''.join(f'{m:>14}' for m in modes)
    print(header)
    print('-' * len(header))
    for label, key in rows:
        if key == 'mode':
            continue
        line = f'{label:<22}'
        for m in all_metrics:
            v = m.get(key)
            line += f'{fmt(v):>14}'
        print(line)
    print('=' * 78)


def print_hypothesis_verdict(all_metrics):
    """Compare continuous vs. discrete vs. social on collapse signals."""
    by_mode = {m['mode']: m for m in all_metrics}

    print()
    print('HYPOTHESIS VERDICTS (directional only -- single seed, short run)')
    print('-' * 78)

    cont = by_mode.get('continuous')
    disc = by_mode.get('discrete')
    soc = by_mode.get('social')

    # H1: discrete bottleneck reduces goal-space collapse relative to continuous.
    if cont and disc:
        c = cont.get('goal_space_coverage') or 0.0
        d = disc.get('goal_space_coverage') or 0.0
        verdict = 'SUPPORTED' if d > c + 0.02 else 'NOT SHOWN'
        print(f'H1 (bottleneck helps):            '
              f'goal_space_coverage  continuous={c:.3f}  discrete={d:.3f}  -> {verdict}')

    # H2: social pressure (discrete + comm) produces MORE diversity than discrete alone.
    if disc and soc:
        d = disc.get('goal_space_coverage') or 0.0
        s = soc.get('goal_space_coverage') or 0.0
        verdict = 'SUPPORTED' if s > d + 0.02 else 'NOT SHOWN'
        print(f'H2 (social > discrete):           '
              f'goal_space_coverage  discrete={d:.3f}  social={s:.3f}  -> {verdict}')

        de = disc.get('entropy') or 0.0
        se = soc.get('entropy') or 0.0
        verdict = 'SUPPORTED' if se > de + 0.1 else 'NOT SHOWN'
        print(f'H2b (social msg entropy higher):  '
              f'entropy              discrete={de:.3f}  social={se:.3f}  -> {verdict}')

    # H3: communication channel actually carries information.
    if soc:
        delta = soc.get('comm_ablation_delta')
        if delta is None:
            print('H3 (comm is not inert):            comm_ablation_delta not computed.')
        else:
            verdict = 'SUPPORTED' if delta > 0.02 else ('INERT' if abs(delta) < 0.02 else 'CONFOUNDED')
            print(f'H3 (comm is not inert):            '
                  f'with - without comm = {delta:+.3f}  -> {verdict}')

    # H4: temporal abstraction -- goals should hold for > 1 step.
    for m in all_metrics:
        te = m.get('temporal_extent')
        if te is None:
            continue
        verdict = 'OK' if te > 1.5 else 'PATHOLOGICAL (goals change every step)'
        print(f'H4 ({m["mode"]}: goals are temporally extended):  '
              f'temporal_extent={te:.2f}  -> {verdict}')

    # H5: bottleneck is actually tight.
    for m in all_metrics:
        rl = m.get('comm_recon_loss')
        if rl is None:
            continue
        # Rough reading: if recon_loss is near zero, the decoder is near
        # identity and the bottleneck is too loose to structure anything.
        verdict = 'TIGHT' if rl > 0.1 else ('LOOSE -- little information loss' if rl < 0.02 else 'MODERATE')
        print(f'H5 ({m["mode"]}: bottleneck is tight):  '
              f'comm_recon_loss={rl:.4f}  -> {verdict}')

    print('-' * 78)
    print('Note: single-seed, short training. Signs matter more than magnitudes.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=DEFAULT_MODES,
                    choices=['flat', 'continuous', 'discrete', 'social',
                             'lola', 'maddpg'])
    ap.add_argument('--timesteps', type=int, default=15000)
    ap.add_argument('--corridor-size', type=int, default=9)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-dir', default='outputs/verify')
    ap.add_argument('--quick', action='store_true',
                    help='Even shorter run (8000 steps) for a smoke test.')
    ap.add_argument('--stress', action='store_true',
                    help='Stress-test: narrow corridor, stronger listener reward, '
                         'more steps for social mode to amortize 2-agent overhead.')
    ap.add_argument('--corridor-width', type=int, default=3)
    ap.add_argument('--listener-reward-coef', type=float, default=0.1)
    ap.add_argument('--bus-cost-solo', type=float, default=0.0,
                    help='Per-step penalty when exactly one agent is in corridor. '
                         'With bus-cost-shared=0, makes simultaneous traversal '
                         'cheaper per-agent (suggested-approach shared bus).')
    ap.add_argument('--bus-cost-shared', type=float, default=0.0,
                    help='Per-step penalty when both agents are in corridor.')
    ap.add_argument('--bus', action='store_true',
                    help='Shortcut: enable shared-bus env (solo=0.05, shared=0.0) '
                         'to create explicit coordination pressure for social mode.')
    ap.add_argument('--vocab-size', type=int, default=10)
    ap.add_argument('--message-length', type=int, default=3)
    ap.add_argument('--bus-window', type=int, default=0,
                    help='Strict-bus arrival window (steps). 0 disables. When '
                         '>0, the shared discount only applies if both agents '
                         'entered the corridor within this many steps.')
    ap.add_argument('--turn-taking', action='store_true',
                    help='Only one agent acts per step (alternating). '
                         'Forces sequential coordination (RQ4).')
    args = ap.parse_args()

    if args.quick:
        args.timesteps = 8000

    if args.stress:
        # Force coordination: single-file corridor, strong listener reward,
        # and more steps for social only (it has 2 agents to train).
        args.corridor_width = 1
        args.listener_reward_coef = 0.5

    if args.bus:
        # Suggested-approach "shared bus cheaper when simultaneous."
        # Solo corridor traversal costs -0.05/step, shared costs 0.
        args.bus_cost_solo = 0.05
        args.bus_cost_shared = 0.0

    root = os.path.join(args.output_dir, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(root, exist_ok=True)
    print(f'[verify] modes={args.modes} timesteps={args.timesteps} '
          f'corridor_size={args.corridor_size} seed={args.seed}')
    print(f'[verify] output: {root}')

    all_metrics = []
    for mode in args.modes:
        print(f'\n[verify] === {mode} ===')
        t0 = time.time()
        # Social has 2 agents to train; give it extra steps under stress so
        # the comparison isn't confounded by per-agent sample efficiency.
        ts = args.timesteps
        if args.stress and mode in SOCIAL_MODES:
            ts = int(args.timesteps * 2.5)
        try:
            m = run_mode(mode, ts, args.corridor_size,
                         args.seed, os.path.join(root, mode), args.device,
                         corridor_width=args.corridor_width,
                         listener_reward_coef=args.listener_reward_coef,
                         bus_cost_solo=args.bus_cost_solo,
                         bus_cost_shared=args.bus_cost_shared,
                         vocab_size=args.vocab_size,
                         message_length=args.message_length,
                         bus_window=args.bus_window,
                         turn_taking=args.turn_taking)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[verify] {mode} FAILED: {e}')
            continue
        dt = time.time() - t0
        print(f'[verify] {mode} done in {dt:.1f}s; '
              f'eval_success={m.get("eval_success_rate")} '
              f'goal_space_coverage={m.get("goal_space_coverage")} '
              f'entropy={m.get("entropy")}')
        all_metrics.append(m)

    with open(os.path.join(root, 'summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=float)

    verdict_table(all_metrics)
    print_hypothesis_verdict(all_metrics)
    print(f'\n[verify] All results in: {root}')


if __name__ == '__main__':
    main()
