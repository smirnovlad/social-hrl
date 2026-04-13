"""Transfer experiments for frozen discrete managers.

Usage:
    python scripts/evaluate_transfer.py --run-all --source-mode discrete \
        --source-task-family keycorridor --source-env MiniGrid-KeyCorridorS3R2-v0
    python scripts/evaluate_transfer.py --checkpoint outputs/.../final.pt \
        --source-mode discrete --transfer-env MiniGrid-KeyCorridorS4R3-v0
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.ppo import compute_gae, ppo_update
from envs.wrappers import make_env, make_vec_env
from experiment_utils import infer_task_family, write_json
from models.communication import CommunicationChannel
from models.encoder import MinigridEncoder
from models.manager import Manager
from models.worker import Worker
from transfer_utils import discover_source_runs, validate_transfer_request


class TransferTrainer:
    """Train a fresh worker under a frozen discrete manager on a new environment."""

    def __init__(self, checkpoint_path, source_mode, transfer_env, device='cuda',
                 lr=0.0003, total_timesteps=500000, num_envs=8, num_steps=128,
                 freeze_encoder=True, freeze_manager=True, freeze_comm=True,
                 same_family=False, corridor_size=None, corridor_width=3,
                 eval_episodes=20):
        validate_transfer_request(source_mode)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.source_mode = source_mode
        self.checkpoint_path = checkpoint_path

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = state['config']

        hidden_dim = config['encoder']['hidden_dim']
        goal_dim = config['manager']['goal_dim']
        self.goal_period = config['manager']['goal_period']
        self.eval_episodes = eval_episodes
        self.same_family = same_family
        self.corridor_size = corridor_size or 15
        self.corridor_width = corridor_width
        self.max_steps = config['env'].get('max_steps', 500)

        if same_family:
            from envs.multi_agent_env import make_corridor_vec_env
            self.envs = make_corridor_vec_env(
                num_envs,
                seed=0,
                max_steps=self.max_steps,
                corridor_width=corridor_width,
                corridor_size=self.corridor_size,
            )
            self.transfer_env_name = f"SingleAgentCorridor-S{self.corridor_size}-W{corridor_width}-v0"
        else:
            self.envs = make_vec_env(transfer_env, num_envs, seed=0, max_steps=self.max_steps)
            self.transfer_env_name = transfer_env

        obs_shape = self.envs.single_observation_space.shape
        num_actions = self.envs.single_action_space.n

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps

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
            tau=comm_cfg['tau_end'],
        ).to(self.device)
        self.comm_channel.load_state_dict(state['comm_channel'])

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_manager:
            for p in self.manager.parameters():
                p.requires_grad = False
        if freeze_comm:
            for p in self.comm_channel.parameters():
                p.requires_grad = False

        self.worker = Worker(
            input_dim=hidden_dim, goal_dim=goal_dim,
            num_actions=num_actions, hidden_dim=config['worker']['hidden_dim'],
        ).to(self.device)
        self.goal_projection = nn.Linear(hidden_dim, goal_dim).to(self.device)

        trainable_params = list(self.worker.parameters()) + list(self.goal_projection.parameters())
        if not freeze_encoder:
            trainable_params += list(self.encoder.parameters())
        if not freeze_manager:
            trainable_params += list(self.manager.parameters())
        if not freeze_comm:
            trainable_params += list(self.comm_channel.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-5)

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

        obs, _ = self.envs.reset()
        self.obs = torch.from_numpy(obs).to(self.device)

        self.freeze_encoder = freeze_encoder
        self.freeze_manager = freeze_manager
        self.freeze_comm = freeze_comm

    def _compute_intrinsic_reward(self, obs_features, goal):
        projected = F.normalize(self.goal_projection(obs_features), dim=-1)
        goal_norm = F.normalize(goal, dim=-1)
        return -torch.norm(projected - goal_norm, dim=-1)

    def _select_goal(self, features):
        h = self.manager.policy(features)
        goal_mean = self.manager.goal_mean(h)
        msg_onehot, _, _ = self.comm_channel.encode(goal_mean)
        return self.comm_channel.decode(msg_onehot)

    def _select_worker_action(self, features, goal):
        logits = self.worker.policy(torch.cat([features, goal], dim=-1))
        return logits.argmax(dim=-1)

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

            needs_new_goal = (steps_since_goal % self.goal_period == 0)
            if needs_new_goal.any():
                with torch.no_grad():
                    decoded_goal = self._select_goal(features)
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

            next_obs, reward, terminated, truncated, _ = self.envs.step(action.cpu().numpy())
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
            for i, is_done in enumerate(done):
                if is_done:
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
                    if self.freeze_encoder:
                        with torch.no_grad():
                            feat = self.encoder(b['obs'])
                    else:
                        feat = self.encoder(b['obs'])
                    return self.worker.evaluate_actions(feat, b['goals'], b['actions'])

                stats = ppo_update(
                    batch, policy_fn, self.optimizer,
                    self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                )
                for k, v in stats.items():
                    all_stats[k].append(v)

        return {k: np.mean(v) for k, v in all_stats.items()}

    def _make_eval_env(self, seed):
        if self.same_family:
            from envs.multi_agent_env import SingleAgentCorridorEnv

            return SingleAgentCorridorEnv(
                size=self.corridor_size,
                corridor_length=3,
                max_steps=self.max_steps,
                corridor_width=self.corridor_width,
            )
        return make_env(self.transfer_env_name, seed=seed, max_steps=self.max_steps)()

    def evaluate(self, num_episodes=None):
        """Evaluate current transfer policy on the raw target task reward."""
        num_episodes = num_episodes or self.eval_episodes
        returns = []
        successes = []
        base_seed = 200_000

        for episode in range(num_episodes):
            env = self._make_eval_env(base_seed + episode)
            obs, _ = env.reset(seed=base_seed + episode)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            current_goal = None
            steps_since_goal = 0
            done = False
            ep_return = 0.0

            while not done:
                with torch.no_grad():
                    features = self.encoder(obs_t)
                    if current_goal is None or steps_since_goal % self.goal_period == 0:
                        current_goal = self._select_goal(features)
                    action = self._select_worker_action(features, current_goal)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                ep_return += reward
                obs_t = torch.from_numpy(next_obs).unsqueeze(0).to(self.device)
                steps_since_goal += 1

            env.close()
            returns.append(ep_return)
            successes.append(float(ep_return > 0.0))

        return {
            'mean_return': float(np.mean(returns)) if returns else 0.0,
            'std_return': float(np.std(returns)) if returns else 0.0,
            'success_rate': float(np.mean(successes)) if successes else 0.0,
        }

    def train(self):
        """Run the transfer training loop."""
        num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        all_returns = []

        freeze_str = []
        if self.freeze_encoder:
            freeze_str.append("encoder")
        if self.freeze_manager:
            freeze_str.append("manager")
        if self.freeze_comm:
            freeze_str.append("comm")
        print(f"  Frozen: {', '.join(freeze_str) if freeze_str else 'none'}")
        print(f"  Target env: {self.transfer_env_name}")

        for update in range(1, num_updates + 1):
            rollout, ep_returns = self.collect_rollout()
            self.update(rollout)
            all_returns.extend(ep_returns)

            if update % 10 == 0:
                recent = all_returns[-50:] if all_returns else [0]
                print(
                    f"  Update {update}/{num_updates} | Step {self.global_step} | "
                    f"Return: {np.mean(recent):.4f} | Episodes: {len(all_returns)}"
                )

        final_eval = self.evaluate(num_episodes=self.eval_episodes)
        print(
            f"  Final eval: return={final_eval['mean_return']:.4f} +/- "
            f"{final_eval['std_return']:.4f} | success={100 * final_eval['success_rate']:.1f}%"
        )
        return {
            'returns': all_returns,
            'eval': final_eval,
            'target': {
                'env_name': self.transfer_env_name,
                'task_family': infer_task_family(
                    'discrete',
                    self.transfer_env_name,
                    use_corridor=self.same_family,
                ),
            },
            'freeze_encoder': self.freeze_encoder,
            'freeze_manager': self.freeze_manager,
            'freeze_comm': self.freeze_comm,
        }


def save_transfer_run(output_dir, record, results, protocol_name):
    """Persist per-run transfer artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'returns.npy'), np.array(results['returns']))
    transfer_metrics = dict(results['eval'])
    transfer_metrics['training_final_return'] = (
        float(np.mean(results['returns'][-100:])) if results['returns'] else 0.0
    )
    transfer_metrics['total_episodes'] = len(results['returns'])
    write_json(os.path.join(output_dir, 'transfer_metrics.json'), transfer_metrics)
    write_json(
        os.path.join(output_dir, 'transfer_run_info.json'),
        {
            'protocol': protocol_name,
            'seed': record['seed'],
            'source_mode': record['mode'],
            'source_task_family': record['task_family'],
            'source_env_name': record.get('source_env_name', record['env_name']),
            'source_checkpoint': record['checkpoint_path'],
            'source_eval_success_rate': record.get('metrics', {}).get('eval_success_rate'),
            'source_condition_id': record['condition_id'],
            'source_condition_label': record['condition_label'],
            'target_env_name': results['target']['env_name'],
            'target_task_family': results['target']['task_family'],
            'freeze_encoder': results['freeze_encoder'],
            'freeze_manager': results['freeze_manager'],
            'freeze_comm': results['freeze_comm'],
            'training_final_return': transfer_metrics['training_final_return'],
            'total_episodes': transfer_metrics['total_episodes'],
            'target_eval': results['eval'],
        },
    )


def summarize_results(protocol_name, source_mode, selected_sources, skipped_sources, runs,
                      freeze_encoder, freeze_manager, freeze_comm, same_family,
                      transfer_env, corridor_size, corridor_width, min_source_success):
    """Build the aggregate summary payload."""
    summary = {
        'protocol': protocol_name,
        'source_mode': source_mode,
        'freeze_encoder': freeze_encoder,
        'freeze_manager': freeze_manager,
        'freeze_comm': freeze_comm,
        'same_family': same_family,
        'transfer_env': transfer_env,
        'target_env_name': (
            f"SingleAgentCorridor-S{corridor_size}-W{corridor_width}-v0"
            if same_family else transfer_env
        ),
        'corridor_size': corridor_size,
        'corridor_width': corridor_width,
        'min_source_success': min_source_success,
        'matched_sources': [
            {
                'seed': record['seed'],
                'checkpoint': record['checkpoint_path'],
                'source_task_family': record['task_family'],
                'source_env_name': record.get('source_env_name', record['env_name']),
                'source_eval_success_rate': record.get('metrics', {}).get('eval_success_rate'),
            }
            for record in sorted(selected_sources.values(), key=lambda item: item['seed'])
        ],
        'skipped_sources': skipped_sources,
        source_mode: {
            'training_final_returns': [],
            'eval_mean_returns': [],
            'eval_success_rates': [],
            'total_episodes': [],
            'source_checkpoints': [],
        },
    }

    for record, results in runs:
        summary[source_mode]['training_final_returns'].append(
            float(np.mean(results['returns'][-100:])) if results['returns'] else 0.0
        )
        summary[source_mode]['eval_mean_returns'].append(results['eval']['mean_return'])
        summary[source_mode]['eval_success_rates'].append(results['eval']['success_rate'])
        summary[source_mode]['total_episodes'].append(len(results['returns']))
        summary[source_mode]['source_checkpoints'].append(record['checkpoint_path'])

    if not runs:
        summary['status'] = 'skipped'
        summary['skip_reason'] = 'no_eligible_source_runs'

    return summary


def main():
    parser = argparse.ArgumentParser(description='Transfer experiment')
    parser.add_argument('--checkpoint', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--source-mode', type=str, default=None, choices=['discrete', 'social'])
    parser.add_argument('--run-all', action='store_true', help='Run transfer for all matching sources')
    parser.add_argument('--base-dir', type=str, default='outputs/')
    parser.add_argument('--transfer-env', type=str, default='MiniGrid-KeyCorridorS4R3-v0')
    parser.add_argument('--total-timesteps', type=int, default=500000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/transfer')
    parser.add_argument('--min-source-success', type=float, default=0.5)
    parser.add_argument('--eval-episodes', type=int, default=20)

    parser.add_argument('--no-freeze-encoder', action='store_true',
                        help='Fine-tune encoder on target env')
    parser.add_argument('--no-freeze-manager', action='store_true',
                        help='Fine-tune manager on target env')
    parser.add_argument('--no-freeze-comm', action='store_true',
                        help='Fine-tune comm channel on target env')

    parser.add_argument('--same-family', action='store_true',
                        help='Transfer within corridor variants')
    parser.add_argument('--corridor-size', type=int, default=15,
                        help='Target corridor grid size')
    parser.add_argument('--corridor-width', type=int, default=3,
                        help='Target corridor width')
    parser.add_argument('--source-task-family', type=str, default=None,
                        help='Restrict source discovery to a task family')
    parser.add_argument('--source-env', type=str, default=None,
                        help='Restrict source discovery to a specific source env name')
    parser.add_argument('--protocol-name', type=str, default=None,
                        help='Optional label stored in transfer artifacts')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    freeze_encoder = not args.no_freeze_encoder
    freeze_manager = not args.no_freeze_manager
    freeze_comm = not args.no_freeze_comm

    if args.run_all:
        source_mode = args.source_mode or 'discrete'
        validate_transfer_request(source_mode)
        selected_sources, skipped_sources = discover_source_runs(
            args.base_dir,
            source_mode,
            source_task_family=args.source_task_family,
            source_env=args.source_env,
            min_source_success=args.min_source_success,
        )

        runs = []
        for seed, record in sorted(selected_sources.items()):
            print(f"\n{'=' * 60}")
            print(f"Transfer: {source_mode} seed {seed} -> {args.transfer_env}")
            print(f"Checkpoint: {record['checkpoint_path']}")
            print(f"Freeze encoder: {freeze_encoder}")
            print(f"{'=' * 60}")

            trainer = TransferTrainer(
                checkpoint_path=record['checkpoint_path'],
                source_mode=source_mode,
                transfer_env=args.transfer_env,
                device=args.device,
                total_timesteps=args.total_timesteps,
                freeze_encoder=freeze_encoder,
                freeze_manager=freeze_manager,
                freeze_comm=freeze_comm,
                same_family=args.same_family,
                corridor_size=args.corridor_size,
                corridor_width=args.corridor_width,
                eval_episodes=args.eval_episodes,
            )
            results = trainer.train()
            runs.append((record, results))

            save_transfer_run(
                os.path.join(args.output_dir, f"{source_mode}_seed{seed}"),
                record,
                results,
                args.protocol_name or source_mode,
            )

        summary = summarize_results(
            protocol_name=args.protocol_name or source_mode,
            source_mode=source_mode,
            selected_sources=selected_sources,
            skipped_sources=skipped_sources,
            runs=runs,
            freeze_encoder=freeze_encoder,
            freeze_manager=freeze_manager,
            freeze_comm=freeze_comm,
            same_family=args.same_family,
            transfer_env=args.transfer_env,
            corridor_size=args.corridor_size,
            corridor_width=args.corridor_width,
            min_source_success=args.min_source_success,
        )
        write_json(os.path.join(args.output_dir, 'transfer_summary.json'), summary)

        print(f"\n{'=' * 60}")
        print("TRANSFER RESULTS SUMMARY")
        print(f"{'=' * 60}")
        if runs:
            mode_summary = summary[source_mode]
            print(
                f"  {source_mode:12s}: eval_return="
                f"{np.mean(mode_summary['eval_mean_returns']):.4f}+/-"
                f"{np.std(mode_summary['eval_mean_returns']):.4f}  "
                f"success={100 * np.mean(mode_summary['eval_success_rates']):.1f}%"
            )
        else:
            print("  No eligible source runs matched the requested filters.")

    elif args.checkpoint and args.source_mode:
        validate_transfer_request(args.source_mode)
        print(f"Transfer: {args.source_mode} -> {args.transfer_env}")
        trainer = TransferTrainer(
            checkpoint_path=args.checkpoint,
            source_mode=args.source_mode,
            transfer_env=args.transfer_env,
            device=args.device,
            total_timesteps=args.total_timesteps,
            freeze_encoder=freeze_encoder,
            freeze_manager=freeze_manager,
            freeze_comm=freeze_comm,
            same_family=args.same_family,
            corridor_size=args.corridor_size,
            corridor_width=args.corridor_width,
            eval_episodes=args.eval_episodes,
        )
        results = trainer.train()
        np.save(os.path.join(args.output_dir, 'returns.npy'), np.array(results['returns']))
        transfer_metrics = dict(results['eval'])
        transfer_metrics['training_final_return'] = (
            float(np.mean(results['returns'][-100:])) if results['returns'] else 0.0
        )
        transfer_metrics['total_episodes'] = len(results['returns'])
        write_json(os.path.join(args.output_dir, 'transfer_metrics.json'), transfer_metrics)
        write_json(
            os.path.join(args.output_dir, 'transfer_run_info.json'),
            {
                'protocol': args.protocol_name or 'single_checkpoint',
                'seed': None,
                'source_mode': args.source_mode,
                'source_task_family': None,
                'source_env_name': None,
                'source_checkpoint': args.checkpoint,
                'source_eval_success_rate': None,
                'source_condition_id': None,
                'source_condition_label': None,
                'target_env_name': results['target']['env_name'],
                'target_task_family': results['target']['task_family'],
                'freeze_encoder': results['freeze_encoder'],
                'freeze_manager': results['freeze_manager'],
                'freeze_comm': results['freeze_comm'],
                'training_final_return': transfer_metrics['training_final_return'],
                'total_episodes': transfer_metrics['total_episodes'],
                'target_eval': results['eval'],
            },
        )
        print(f"\nFinal eval return: {results['eval']['mean_return']:.4f}")
        print(f"Final eval success: {100 * results['eval']['success_rate']:.1f}%")
    else:
        parser.error("Specify --run-all or --checkpoint + --source-mode")


if __name__ == '__main__':
    main()
