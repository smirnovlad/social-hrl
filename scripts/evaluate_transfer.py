"""Transfer experiment: freeze trained managers, train fresh workers on new env.

Tests H3: do socially-trained goal representations transfer better?

Usage:
    python scripts/evaluate_transfer.py --checkpoint outputs/.../final.pt --source-mode discrete
    python scripts/evaluate_transfer.py --run-all --total-timesteps 500000
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import MinigridEncoder
from models.manager import Manager
from models.worker import Worker
from models.communication import CommunicationChannel
from algos.ppo import compute_gae, ppo_update
from envs.wrappers import make_vec_env
from collections import defaultdict


class TransferTrainer:
    """Train a fresh worker under a frozen manager on a new environment."""

    def __init__(self, checkpoint_path, source_mode, transfer_env, device='cuda',
                 lr=0.0003, total_timesteps=500000, num_envs=8, num_steps=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.source_mode = source_mode

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = state['config']

        hidden_dim = config['encoder']['hidden_dim']
        goal_dim = config['manager']['goal_dim']
        self.goal_period = config['manager']['goal_period']

        # Create new environment
        self.envs = make_vec_env(transfer_env, num_envs, seed=0,
                                 max_steps=config['env'].get('max_steps', 500))
        obs_shape = self.envs.single_observation_space.shape
        num_actions = self.envs.single_action_space.n

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps

        # Load frozen encoder + manager + comm_channel
        if source_mode == 'discrete':
            self.encoder = MinigridEncoder(
                obs_shape, channels=tuple(config['encoder']['channels']),
                hidden_dim=hidden_dim
            ).to(self.device)
            self.encoder.load_state_dict(state['encoder'])

            self.manager = Manager(
                input_dim=hidden_dim, goal_dim=goal_dim,
                hidden_dim=config['manager']['hidden_dim'],
            ).to(self.device)
            self.manager.load_state_dict(state['manager'])

            comm_cfg = config['communication']
            self.comm_channel = CommunicationChannel(
                goal_dim=goal_dim, vocab_size=comm_cfg['vocab_size'],
                message_length=comm_cfg['message_length'],
                tau=comm_cfg['tau_end'],  # use final (low) temperature
            ).to(self.device)
            self.comm_channel.load_state_dict(state['comm_channel'])

        elif source_mode == 'social':
            # Use agent 0's models
            self.encoder = MinigridEncoder(
                obs_shape, channels=tuple(config['encoder']['channels']),
                hidden_dim=hidden_dim
            ).to(self.device)
            enc_state = {}
            for k, v in state['encoders'].items():
                if k.startswith('0.'):
                    enc_state[k[2:]] = v
            self.encoder.load_state_dict(enc_state)

            # Social manager expects message_dim input — use zero messages
            # since we're running solo. Load without message_dim to match.
            self.manager = Manager(
                input_dim=hidden_dim, goal_dim=goal_dim,
                hidden_dim=config['manager']['hidden_dim'],
                message_dim=goal_dim,  # social manager was trained with messages
            ).to(self.device)
            mgr_state = {}
            for k, v in state['managers'].items():
                if k.startswith('0.'):
                    mgr_state[k[2:]] = v
            self.manager.load_state_dict(mgr_state)

            comm_cfg = config['communication']
            self.comm_channel = CommunicationChannel(
                goal_dim=goal_dim, vocab_size=comm_cfg['vocab_size'],
                message_length=comm_cfg['message_length'],
                tau=comm_cfg['tau_end'],
            ).to(self.device)
            cc_state = {}
            for k, v in state['comm_channels'].items():
                if k.startswith('0.'):
                    cc_state[k[2:]] = v
            self.comm_channel.load_state_dict(cc_state)

        # Freeze encoder, manager, comm_channel
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.manager.parameters():
            p.requires_grad = False
        for p in self.comm_channel.parameters():
            p.requires_grad = False

        # Fresh worker + goal_projection (trainable)
        self.worker = Worker(
            input_dim=hidden_dim, goal_dim=goal_dim,
            num_actions=num_actions, hidden_dim=config['worker']['hidden_dim'],
        ).to(self.device)

        self.goal_projection = nn.Linear(hidden_dim, goal_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.worker.parameters()) + list(self.goal_projection.parameters()),
            lr=lr, eps=1e-5,
        )

        # PPO config
        ppo_cfg = config['ppo']
        self.gamma = ppo_cfg['gamma']
        self.gae_lambda = ppo_cfg['gae_lambda']
        self.clip_eps = ppo_cfg['clip_eps']
        self.entropy_coef = ppo_cfg['entropy_coef']
        self.value_coef = ppo_cfg['value_coef']
        self.max_grad_norm = ppo_cfg['max_grad_norm']
        self.update_epochs = ppo_cfg['update_epochs']
        self.num_minibatches = ppo_cfg['num_minibatches']

        self.intrinsic_coef = config['worker']['intrinsic_reward_coef']
        self.extrinsic_coef = config['worker']['extrinsic_reward_coef']

        self.global_step = 0
        self.zero_message = torch.zeros(
            num_envs, goal_dim
        ).to(self.device) if source_mode == 'social' else None

        # Persistent state
        obs, _ = self.envs.reset()
        self.obs = torch.from_numpy(obs).to(self.device)

    def _compute_intrinsic_reward(self, obs_features, goal):
        projected = F.normalize(self.goal_projection(obs_features), dim=-1)
        goal_norm = F.normalize(goal, dim=-1)
        return -torch.norm(projected - goal_norm, dim=-1)

    def collect_rollout(self):
        T = self.num_steps
        N = self.num_envs

        obs_buf = torch.zeros(T, N, *self.envs.single_observation_space.shape).to(self.device)
        goal_buf = torch.zeros(T, N, self.manager.goal_dim).to(self.device)
        act_buf = torch.zeros(T, N, dtype=torch.long).to(self.device)
        logp_buf = torch.zeros(T, N).to(self.device)
        rew_buf = torch.zeros(T, N).to(self.device)
        done_buf = torch.zeros(T, N).to(self.device)
        val_buf = torch.zeros(T, N).to(self.device)

        obs = self.obs
        current_goal = torch.zeros(N, self.manager.goal_dim).to(self.device)
        steps_since_goal = torch.zeros(N, dtype=torch.long).to(self.device)

        episode_rewards = np.zeros(N)
        episode_returns = []

        for step in range(T):
            obs_buf[step] = obs

            with torch.no_grad():
                features = self.encoder(obs)

            # Frozen manager sets goals every c steps
            needs_new_goal = (steps_since_goal % self.goal_period == 0)
            if needs_new_goal.any():
                with torch.no_grad():
                    if self.source_mode == 'social':
                        goal, _, _, _ = self.manager(features, received_message=self.zero_message)
                    else:
                        goal, _, _, _ = self.manager(features)
                    msg_onehot, _, _ = self.comm_channel.encode(goal)
                    decoded_goal = self.comm_channel.decode(msg_onehot)

                current_goal = torch.where(
                    needs_new_goal.unsqueeze(-1).expand_as(decoded_goal),
                    decoded_goal, current_goal
                )

            goal_buf[step] = current_goal

            with torch.no_grad():
                action, log_prob, value = self.worker(features, current_goal)

            act_buf[step] = action
            logp_buf[step] = log_prob
            val_buf[step] = value

            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            reward_t = torch.from_numpy(reward).float().to(self.device)
            done_t = torch.from_numpy(done).float().to(self.device)

            next_obs_t = torch.from_numpy(next_obs).to(self.device)
            with torch.no_grad():
                next_features = self.encoder(next_obs_t)
                intrinsic = self._compute_intrinsic_reward(next_features, current_goal)
            worker_reward = self.intrinsic_coef * intrinsic + self.extrinsic_coef * reward_t

            rew_buf[step] = worker_reward
            done_buf[step] = done_t

            episode_rewards += reward
            for i, d in enumerate(done):
                if d:
                    episode_returns.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    steps_since_goal[i] = 0

            steps_since_goal += 1
            obs = next_obs_t
            self.global_step += N

        self.obs = obs

        with torch.no_grad():
            features = self.encoder(obs)
            next_val = self.worker.get_value(features, current_goal)

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_val, self.gamma, self.gae_lambda
        )

        return {
            'obs': obs_buf.reshape(-1, *self.envs.single_observation_space.shape),
            'goals': goal_buf.reshape(-1, self.manager.goal_dim),
            'actions': act_buf.reshape(-1),
            'old_log_probs': logp_buf.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
        }, episode_returns

    def update(self, rollout):
        B = rollout['obs'].shape[0]
        batch_size = B // self.num_minibatches
        all_stats = defaultdict(list)

        for _ in range(self.update_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, batch_size):
                mb_idx = indices[start:start + batch_size]
                batch = {k: v[mb_idx] for k, v in rollout.items()}

                def policy_fn(b):
                    with torch.no_grad():
                        feat = self.encoder(b['obs'])
                    return self.worker.evaluate_actions(feat, b['goals'], b['actions'])

                stats = ppo_update(
                    batch, policy_fn, self.optimizer,
                    self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                )
                for k, v in stats.items():
                    all_stats[k].append(v)

        return {k: np.mean(v) for k, v in all_stats.items()}

    def train(self):
        num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        all_returns = []

        for update in range(1, num_updates + 1):
            rollout, ep_returns = self.collect_rollout()
            self.update(rollout)
            all_returns.extend(ep_returns)

            if update % 10 == 0:
                recent = all_returns[-50:] if all_returns else [0]
                print(f"  Update {update}/{num_updates} | "
                      f"Step {self.global_step} | "
                      f"Return: {np.mean(recent):.4f} | "
                      f"Episodes: {len(all_returns)}")

        return all_returns


def find_checkpoints(base_dir, mode, seeds=(42, 123, 7)):
    """Find latest checkpoint for each seed of a given mode."""
    import glob
    results = {}
    for seed in seeds:
        pattern = f"{base_dir}/*/{mode}_seed{seed}/*/final.pt"
        files = sorted(glob.glob(pattern))
        if files:
            results[seed] = files[-1]
    return results


def main():
    parser = argparse.ArgumentParser(description='Transfer experiment')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint path')
    parser.add_argument('--source-mode', type=str, default=None,
                        choices=['discrete', 'social'])
    parser.add_argument('--run-all', action='store_true',
                        help='Run transfer for all conditions')
    parser.add_argument('--base-dir', type=str, default='outputs/')
    parser.add_argument('--transfer-env', type=str,
                        default='MiniGrid-KeyCorridorS4R3-v0')
    parser.add_argument('--total-timesteps', type=int, default=500000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/transfer')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all:
        conditions = ['discrete', 'social']
        all_results = {}

        for mode in conditions:
            checkpoints = find_checkpoints(args.base_dir, mode)
            if not checkpoints:
                print(f"No checkpoints found for {mode}")
                continue

            mode_returns = []
            for seed, ckpt_path in sorted(checkpoints.items()):
                print(f"\n{'='*60}")
                print(f"Transfer: {mode} seed {seed} -> {args.transfer_env}")
                print(f"Checkpoint: {ckpt_path}")
                print(f"{'='*60}")

                trainer = TransferTrainer(
                    checkpoint_path=ckpt_path,
                    source_mode=mode,
                    transfer_env=args.transfer_env,
                    device=args.device,
                    total_timesteps=args.total_timesteps,
                )
                returns = trainer.train()
                mode_returns.append(returns)

                # Save per-run results
                run_dir = os.path.join(args.output_dir, f"{mode}_seed{seed}")
                os.makedirs(run_dir, exist_ok=True)
                np.save(os.path.join(run_dir, 'returns.npy'), np.array(returns))

            all_results[mode] = mode_returns

        # Summary
        print(f"\n{'='*60}")
        print(f"TRANSFER RESULTS SUMMARY")
        print(f"{'='*60}")
        for mode, runs in all_results.items():
            final_means = [np.mean(r[-100:]) if len(r) >= 100 else np.mean(r) for r in runs]
            total_eps = [len(r) for r in runs]
            print(f"  {mode:12s}: final_return={np.mean(final_means):.4f}+/-{np.std(final_means):.4f}  "
                  f"episodes={np.mean(total_eps):.0f}")

        # Save summary
        summary = {}
        for mode, runs in all_results.items():
            summary[mode] = {
                'final_returns': [np.mean(r[-100:]) if len(r) >= 100 else np.mean(r) for r in runs],
                'total_episodes': [len(r) for r in runs],
            }
        with open(os.path.join(args.output_dir, 'transfer_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=float)

    elif args.checkpoint and args.source_mode:
        print(f"Transfer: {args.source_mode} -> {args.transfer_env}")
        trainer = TransferTrainer(
            checkpoint_path=args.checkpoint,
            source_mode=args.source_mode,
            transfer_env=args.transfer_env,
            device=args.device,
            total_timesteps=args.total_timesteps,
        )
        returns = trainer.train()
        np.save(os.path.join(args.output_dir, 'returns.npy'), np.array(returns))
        print(f"\nFinal return (last 100): {np.mean(returns[-100:]):.4f}")
    else:
        parser.error("Specify --run-all or --checkpoint + --source-mode")


if __name__ == '__main__':
    main()
