"""Main training script for all experiment conditions.

Usage:
    # Original modes
    python scripts/train.py --mode flat --seed 42
    python scripts/train.py --mode continuous --seed 42
    python scripts/train.py --mode discrete --seed 42
    python scripts/train.py --mode social --seed 42

    # New modes
    python scripts/train.py --mode option_critic --corridor --seed 42
    python scripts/train.py --mode sac_continuous --corridor --seed 42

    # New features
    python scripts/train.py --mode social --intrinsic-anneal --seed 42
    python scripts/train.py --mode social --listener-reward 0.1 --seed 42
    python scripts/train.py --mode social --corridor-width 1 --seed 42
    python scripts/train.py --mode social --eval-comm-ablation --seed 42
    python scripts/train.py --mode social --asymmetric-info --seed 42

    # Harder environments
    python scripts/train.py --mode flat --env MiniGrid-MultiRoom-N6-v0 --seed 42
    python scripts/train.py --mode discrete --env MiniGrid-KeyCorridorS6R3-v0 --seed 42
"""

import argparse
import os
import sys
import random
import yaml
import json
from datetime import datetime
import numpy as np
import torch
# Pin intra-op BLAS threads. These networks are tiny (128-d MLPs) so
# intra-op parallelism is pure overhead, and when the suite runs 4
# parallel jobs without this, each torch process fights the others for
# all cores.
torch.set_num_threads(1)
try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hrl_trainer import HRLTrainer
from algos.multi_agent_trainer import MultiAgentTrainer
from analysis.goal_metrics import compute_all_metrics
from experiment_utils import build_run_metadata, write_json


def load_config(config_path='configs/default.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Social HRL Training')
    parser.add_argument('--mode', type=str, default='flat',
                        choices=['flat', 'continuous', 'discrete', 'social',
                                 'option_critic', 'sac_continuous'],
                        help='Training mode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Override total timesteps')
    parser.add_argument('--env', type=str, default=None,
                        help='Override environment name')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None,
                        help='Root directory under which run_slug/timestamp will be created')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--corridor', action='store_true',
                        help='Use corridor env instead of MiniGrid (for fair social comparison)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')

    # New feature flags
    parser.add_argument('--corridor-width', type=int, default=None,
                        help='Corridor width (1=narrow blocking, 3=default)')
    parser.add_argument('--asymmetric-info', action='store_true',
                        help='In social corridor mode, mask goal tiles from agent 1. '
                             'NOTE: social+corridor defaults to True (to give the '
                             'communication channel an actual reason to exist). '
                             'Use --no-asymmetric-info to opt out.')
    parser.add_argument('--no-asymmetric-info', action='store_true',
                        help='Force asymmetric_info off (overrides the social-mode default).')
    parser.add_argument('--intrinsic-anneal', action='store_true',
                        help='Anneal intrinsic reward coefficient to 0')
    parser.add_argument('--intrinsic-anneal-steps', type=int, default=None,
                        help='Steps over which to anneal intrinsic reward')
    parser.add_argument('--listener-reward', type=float, default=None,
                        help='Listener reward coefficient (social mode)')
    parser.add_argument('--eval-comm-ablation', action='store_true',
                        help='Evaluate with/without communication at end (social mode)')
    parser.add_argument('--comm-ablation', action='store_true',
                        help='Deprecated alias for --eval-comm-ablation')
    parser.add_argument('--use-sac', action='store_true',
                        help='Use SAC instead of TD3 for continuous manager')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='Override communication vocab size K')
    parser.add_argument('--message-length', type=int, default=None,
                        help='Override communication message length L')
    parser.add_argument('--rendezvous-bonus', type=float, default=None,
                        help='Reward bonus when both agents occupy corridor simultaneously')
    parser.add_argument('--num-obstacles', type=int, default=None,
                        help='Number of random wall obstacles in corridor env rooms')

    args = parser.parse_args()

    config = load_config(args.config)
    config['experiment']['seed'] = args.seed

    # Seed all global RNGs for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.total_timesteps:
        config['ppo']['total_timesteps'] = args.total_timesteps
    if args.env:
        config['env']['name'] = args.env

    config['env'].setdefault('corridor_size', 11)
    eval_comm_ablation = args.eval_comm_ablation or args.comm_ablation

    # Apply new feature overrides
    if args.corridor_width is not None:
        config['env']['corridor_width'] = args.corridor_width
    # Social + corridor without asymmetric info leaves the comm channel with
    # no informational role: both agents see the goal, so messages are pure
    # noise. Default to asymmetric unless the user explicitly opts out.
    if args.no_asymmetric_info:
        config['env']['asymmetric_info'] = False
    elif args.asymmetric_info:
        config['env']['asymmetric_info'] = True
    elif args.mode == 'social' and args.corridor:
        config['env']['asymmetric_info'] = True
        print("[info] social+corridor: defaulting asymmetric_info=True "
              "(use --no-asymmetric-info to disable).")
    if args.intrinsic_anneal:
        config['worker']['intrinsic_anneal'] = True
    if args.intrinsic_anneal_steps is not None:
        config['worker']['intrinsic_anneal_steps'] = args.intrinsic_anneal_steps
    if args.listener_reward is not None:
        config.setdefault('multi_agent', {})['listener_reward_coef'] = args.listener_reward
        config.setdefault('communication', {})['listener_reward_coef'] = args.listener_reward
        if args.listener_reward > 0:
            print(
                "[warn] listener_reward > 0: this rewards movement toward a "
                "partner's stated goal direction. Gains in entropy / "
                "compositionality under this flag CANNOT be attributed to "
                "'social pressure regularizes HRL' alone -- they are a "
                "coordination-skill bonus. Report separately."
            )
    if eval_comm_ablation:
        config.setdefault('communication', {})['ablation_mode'] = True
    if args.mode == 'option_critic':
        config['manager']['use_option_critic'] = True
    if args.use_sac or args.mode == 'sac_continuous':
        config.setdefault('sac', {})['enabled'] = True
    if args.vocab_size is not None:
        config['communication']['vocab_size'] = args.vocab_size
    if args.message_length is not None:
        config['communication']['message_length'] = args.message_length
    if args.rendezvous_bonus is not None:
        config['env']['rendezvous_bonus'] = args.rendezvous_bonus
    if args.num_obstacles is not None:
        config['env']['num_obstacles'] = args.num_obstacles

    now = datetime.now()
    run_metadata = build_run_metadata(
        mode=args.mode,
        seed=args.seed,
        env_name=config['env']['name'],
        use_corridor=args.corridor,
        corridor_width=config['env'].get('corridor_width', 3),
        corridor_size=config['env'].get('corridor_size', 11),
        intrinsic_anneal=config['worker'].get('intrinsic_anneal', False),
        listener_reward_coef=config.get('communication', {}).get('listener_reward_coef', 0.0),
        asymmetric_info=config['env'].get('asymmetric_info', False),
        use_sac=config.get('sac', {}).get('enabled', False) or args.mode == 'sac_continuous',
        use_option_critic=args.mode == 'option_critic',
        vocab_size=config['communication']['vocab_size'],
        message_length=config['communication']['message_length'],
        rendezvous_bonus=config['env'].get('rendezvous_bonus', 0.0),
        num_obstacles=config['env'].get('num_obstacles', 0),
        eval_comm_ablation=eval_comm_ablation,
    )

    config['env']['effective_name'] = run_metadata['env_name']
    config['env']['task_family'] = run_metadata['task_family']
    config['env']['use_corridor'] = args.corridor
    config['experiment']['condition_id'] = run_metadata['condition_id']
    config['experiment']['condition_label'] = run_metadata['condition_label']
    config['experiment']['run_slug'] = run_metadata['run_slug']

    exp_name = run_metadata['run_slug']
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output_root:
        output_dir = os.path.join(args.output_root, exp_name, now.strftime("%H-%M-%S"))
    else:
        output_dir = os.path.join(
            "outputs", now.strftime("%Y-%m-%d"), exp_name, now.strftime("%H-%M-%S")
        )
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Init wandb
    run = None
    if not args.no_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it or rerun with --no-wandb."
            )
        tags = [args.mode, config['env']['name']]
        if args.intrinsic_anneal:
            tags.append('intrinsic_anneal')
        if args.listener_reward:
            tags.append('listener_reward')
        if eval_comm_ablation:
            tags.append('comm_ablation')

        run = wandb.init(
            entity="mbzuai-research",
            project="social-hrl",
            name=exp_name,
            config={
                'mode': args.mode,
                'seed': args.seed,
                'env': config['env']['effective_name'],
                'total_timesteps': config['ppo']['total_timesteps'],
                'corridor_width': config['env'].get('corridor_width', 3),
                'corridor_size': config['env'].get('corridor_size', 11),
                'asymmetric_info': config['env'].get('asymmetric_info', False),
                'intrinsic_anneal': config['worker'].get('intrinsic_anneal', False),
                'listener_reward_coef': config.get('communication', {}).get('listener_reward_coef', 0.0),
                'eval_comm_ablation': config.get('communication', {}).get('ablation_mode', False),
                'task_family': run_metadata['task_family'],
                'condition_id': run_metadata['condition_id'],
                'use_sac': config.get('sac', {}).get('enabled', False),
                'use_option_critic': config['manager'].get('use_option_critic', False),
                **{f"ppo/{k}": v for k, v in config['ppo'].items()},
                **{f"encoder/{k}": v for k, v in config['encoder'].items()},
                **{f"manager/{k}": v for k, v in config['manager'].items()
                   if not isinstance(v, dict)},
                **{f"worker/{k}": v for k, v in config['worker'].items()},
                **{f"comm/{k}": v for k, v in config['communication'].items()},
            },
            tags=tags,
        )

    print(f"=" * 60)
    print(f"Social HRL - Mode: {args.mode} - Seed: {args.seed}")
    print(f"Environment: {config['env']['effective_name']}")
    if args.corridor:
        print(
            f"Corridor size/width: {config['env'].get('corridor_size', 11)} / "
            f"{config['env'].get('corridor_width', 3)}"
        )
    if args.intrinsic_anneal:
        print(f"Intrinsic annealing: ON (over {config['worker'].get('intrinsic_anneal_steps', 500000)} steps)")
    if args.listener_reward is not None:
        print(f"Listener reward: {args.listener_reward}")
    if eval_comm_ablation:
        print(f"Comm ablation: ON (eval with/without at end)")
    if args.asymmetric_info:
        print(f"Asymmetric info: ON")
    print(f"Output: {output_dir}")
    print(f"Wandb: {'enabled' if run else 'disabled'}")
    print(f"=" * 60)

    # Create trainer based on mode
    if args.mode == 'social':
        trainer = MultiAgentTrainer(config, device=args.device)
    elif args.mode == 'option_critic':
        trainer = HRLTrainer(config, mode='discrete', device=args.device,
                             use_corridor=args.corridor)
    elif args.mode == 'sac_continuous':
        config.setdefault('sac', {})['enabled'] = True
        trainer = HRLTrainer(config, mode='continuous', device=args.device,
                             use_corridor=args.corridor)
    else:
        trainer = HRLTrainer(config, mode=args.mode, device=args.device,
                             use_corridor=args.corridor)
    results = trainer.train(output_dir=output_dir, wandb_run=run)

    # Compute and save metrics
    metrics = {}
    all_msgs = []
    if results.get('messages'):
        for batch in results['messages']:
            if isinstance(batch, np.ndarray):
                for msg in batch:
                    all_msgs.append(msg)

    all_states = []
    if results.get('states'):
        for batch in results['states']:
            if isinstance(batch, np.ndarray):
                for s in batch:
                    all_states.append(s)

    metrics = compute_all_metrics(
        messages=all_msgs if all_msgs else None,
        vocab_size=config['communication']['vocab_size'],
        message_length=config['communication']['message_length'],
        states=all_states if all_states else None,
        decoded_goals=results.get('decoded_goals'),
    )

    metrics['final_return_mean'] = np.mean(results['returns'][-100:]) if results['returns'] else 0
    metrics['final_return_std'] = np.std(results['returns'][-100:]) if results['returns'] else 0
    metrics['total_episodes'] = len(results['returns'])
    if results.get('recon_loss_mean') is not None:
        metrics['comm_recon_loss'] = results['recon_loss_mean']
    if results.get('mean_beta') is not None:
        metrics['oc_mean_beta'] = results['mean_beta']
    if results.get('eval'):
        metrics['eval_mean_return'] = results['eval']['mean_return']
        metrics['eval_std_return'] = results['eval']['std_return']
        metrics['eval_success_rate'] = results['eval']['success_rate']
    if results.get('comm_ablation_eval'):
        comm_eval = results['comm_ablation_eval']
        metrics['eval_with_comm'] = comm_eval['with_comm']['mean_return']
        metrics['eval_with_comm_std'] = comm_eval['with_comm']['std_return']
        metrics['eval_with_comm_success_rate'] = comm_eval['with_comm']['success_rate']
        metrics['eval_without_comm'] = comm_eval['without_comm']['mean_return']
        metrics['eval_without_comm_std'] = comm_eval['without_comm']['std_return']
        metrics['eval_without_comm_success_rate'] = comm_eval['without_comm']['success_rate']
        metrics['comm_ablation_delta'] = comm_eval['delta']

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

    # Save returns for plotting
    np.save(os.path.join(output_dir, 'returns.npy'), np.array(results['returns']))
    if results.get('history') is not None:
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(results['history'], f, indent=2, default=float)
    run_metadata['output_dir'] = output_dir
    write_json(os.path.join(output_dir, 'run_info.json'), run_metadata)

    # Log final metrics to wandb
    if run is not None:
        run.summary.update(metrics)
        run.finish()

    print(f"\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
