"""Main training script for all experiment conditions.

Usage:
    python scripts/train.py --mode flat --seed 42
    python scripts/train.py --mode continuous --seed 42
    python scripts/train.py --mode discrete --seed 42
    python scripts/train.py --mode social --seed 42
"""

import argparse
import os
import sys
import yaml
import json
from datetime import datetime
import numpy as np
import wandb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hrl_trainer import HRLTrainer
from algos.multi_agent_trainer import MultiAgentTrainer
from analysis.goal_metrics import compute_all_metrics


def load_config(config_path='configs/default.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Social HRL Training')
    parser.add_argument('--mode', type=str, default='flat',
                        choices=['flat', 'continuous', 'discrete', 'social'],
                        help='Training mode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Override total timesteps')
    parser.add_argument('--env', type=str, default=None,
                        help='Override environment name')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--corridor', action='store_true',
                        help='Use corridor env instead of MiniGrid (for fair social comparison)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    args = parser.parse_args()

    config = load_config(args.config)
    config['experiment']['seed'] = args.seed

    if args.total_timesteps:
        config['ppo']['total_timesteps'] = args.total_timesteps
    if args.env:
        config['env']['name'] = args.env

    now = datetime.now()
    suffix = "_corridor" if args.corridor else ""
    exp_name = f"{args.mode}{suffix}_seed{args.seed}"
    output_dir = args.output_dir or os.path.join(
        "outputs", now.strftime("%Y-%m-%d"), exp_name, now.strftime("%H-%M-%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Init wandb
    run = None
    if not args.no_wandb:
        run = wandb.init(
            entity="mbzuai-research",
            project="social-hrl",
            name=exp_name,
            config={
                'mode': args.mode,
                'seed': args.seed,
                'env': config['env']['name'],
                'total_timesteps': config['ppo']['total_timesteps'],
                **{f"ppo/{k}": v for k, v in config['ppo'].items()},
                **{f"encoder/{k}": v for k, v in config['encoder'].items()},
                **{f"manager/{k}": v for k, v in config['manager'].items()},
                **{f"worker/{k}": v for k, v in config['worker'].items()},
                **{f"comm/{k}": v for k, v in config['communication'].items()},
            },
            tags=[args.mode, config['env']['name']],
        )

    print(f"=" * 60)
    print(f"Social HRL - Mode: {args.mode} - Seed: {args.seed}")
    print(f"Environment: {config['env']['name']}")
    print(f"Output: {output_dir}")
    print(f"Wandb: {'enabled' if run else 'disabled'}")
    print(f"=" * 60)

    # Train
    if args.mode == 'social':
        trainer = MultiAgentTrainer(config, device=args.device)
    else:
        trainer = HRLTrainer(config, mode=args.mode, device=args.device,
                             use_corridor=args.corridor)
    results = trainer.train(output_dir=output_dir, wandb_run=run)

    # Compute and save metrics
    metrics = {}
    if results.get('messages'):
        all_msgs = []
        for batch in results['messages']:
            if isinstance(batch, np.ndarray):
                for msg in batch:
                    all_msgs.append(msg)
        metrics = compute_all_metrics(
            messages=all_msgs,
            vocab_size=config['communication']['vocab_size'],
            message_length=config['communication']['message_length'],
        )

    metrics['final_return_mean'] = np.mean(results['returns'][-100:]) if results['returns'] else 0
    metrics['final_return_std'] = np.std(results['returns'][-100:]) if results['returns'] else 0
    metrics['total_episodes'] = len(results['returns'])

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

    # Save returns for plotting
    np.save(os.path.join(output_dir, 'returns.npy'), np.array(results['returns']))

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
