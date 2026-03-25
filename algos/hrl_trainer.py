"""Single-agent HRL training loop.

Supports three modes:
- flat: Standard PPO (no hierarchy, no goals)
- continuous: HRL with continuous latent goals (condition a)
- discrete: HRL with discrete bottleneck goals (condition b)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from models.encoder import MinigridEncoder
from models.manager import Manager
from models.worker import Worker
from models.communication import CommunicationChannel
from algos.ppo import compute_gae, ppo_update
from envs.wrappers import make_vec_env


class FlatPolicy(nn.Module):
    """Simple flat policy for the no-hierarchy baseline."""

    def __init__(self, obs_shape, num_actions, hidden_dim=128):
        super().__init__()
        self.encoder = MinigridEncoder(obs_shape, hidden_dim=hidden_dim)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.policy(features)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.value_head(features).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        features = self.encoder(obs)
        logits = self.policy(features)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), self.value_head(features).squeeze(-1)


class HRLTrainer:
    """Training loop for single-agent HRL experiments."""

    def __init__(self, config, mode='continuous', device='cuda'):
        self.config = config
        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create environments
        env_cfg = config['env']
        ppo_cfg = config['ppo']
        self.envs = make_vec_env(
            env_cfg['name'], ppo_cfg['num_envs'],
            seed=config['experiment']['seed'],
            max_steps=env_cfg.get('max_steps'),
        )

        obs_shape = self.envs.single_observation_space.shape
        num_actions = self.envs.single_action_space.n

        self.num_envs = ppo_cfg['num_envs']
        self.num_steps = ppo_cfg['num_steps']
        self.goal_period = config['manager']['goal_period']

        # Build models
        if mode == 'flat':
            self.flat_policy = FlatPolicy(obs_shape, num_actions,
                                          config['encoder']['hidden_dim']).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.flat_policy.parameters(), lr=ppo_cfg['lr'], eps=1e-5
            )
        else:
            self.encoder = MinigridEncoder(
                obs_shape,
                channels=tuple(config['encoder']['channels']),
                hidden_dim=config['encoder']['hidden_dim']
            ).to(self.device)

            self.manager = Manager(
                input_dim=config['encoder']['hidden_dim'],
                goal_dim=config['manager']['goal_dim'],
                hidden_dim=config['manager']['hidden_dim'],
            ).to(self.device)

            self.worker = Worker(
                input_dim=config['encoder']['hidden_dim'],
                goal_dim=config['manager']['goal_dim'],
                num_actions=num_actions,
                hidden_dim=config['worker']['hidden_dim'],
            ).to(self.device)

            # Projection from encoder features to goal space for intrinsic reward
            self.goal_projection = nn.Linear(
                config['encoder']['hidden_dim'],
                config['manager']['goal_dim'],
            ).to(self.device)

            worker_params = (
                list(self.encoder.parameters()) +
                list(self.worker.parameters()) +
                list(self.goal_projection.parameters())
            )
            manager_params = list(self.manager.parameters())

            if mode == 'discrete':
                comm_cfg = config['communication']
                self.comm_channel = CommunicationChannel(
                    goal_dim=config['manager']['goal_dim'],
                    vocab_size=comm_cfg['vocab_size'],
                    message_length=comm_cfg['message_length'],
                    tau=comm_cfg['tau_start'],
                ).to(self.device)
                # Comm channel is part of the goal pipeline — add to worker optimizer
                worker_params += list(self.comm_channel.parameters())

                self.tau_start = comm_cfg['tau_start']
                self.tau_end = comm_cfg['tau_end']
                self.tau_anneal_steps = comm_cfg['tau_anneal_steps']

            self.worker_optimizer = torch.optim.Adam(
                worker_params, lr=ppo_cfg['lr'], eps=1e-5
            )
            self.manager_optimizer = torch.optim.Adam(
                manager_params, lr=ppo_cfg['lr'], eps=1e-5
            )

        # PPO config
        self.gamma = ppo_cfg['gamma']
        self.gae_lambda = ppo_cfg['gae_lambda']
        self.clip_eps = ppo_cfg['clip_eps']
        self.entropy_coef = ppo_cfg['entropy_coef']
        self.value_coef = ppo_cfg['value_coef']
        self.max_grad_norm = ppo_cfg['max_grad_norm']
        self.update_epochs = ppo_cfg['update_epochs']
        self.num_minibatches = ppo_cfg['num_minibatches']
        self.total_timesteps = ppo_cfg['total_timesteps']

        self.intrinsic_coef = config['worker']['intrinsic_reward_coef']
        self.extrinsic_coef = config['worker']['extrinsic_reward_coef']

        # Logging
        self.global_step = 0
        self.log = defaultdict(list)

    def _anneal_tau(self):
        """Anneal Gumbel-Softmax temperature."""
        if self.mode != 'discrete':
            return
        frac = min(1.0, self.global_step / self.tau_anneal_steps)
        tau = self.tau_start + frac * (self.tau_end - self.tau_start)
        self.comm_channel.set_tau(tau)
        return tau

    def _compute_intrinsic_reward(self, obs_features, goal):
        """Worker's intrinsic reward: negative distance to goal in latent space."""
        projected = self.goal_projection(obs_features)
        return -torch.norm(projected - goal, dim=-1)

    def collect_rollout_flat(self):
        """Collect rollout data for flat PPO baseline."""
        obs_buf = torch.zeros(self.num_steps, self.num_envs, *self.envs.single_observation_space.shape).to(self.device)
        act_buf = torch.zeros(self.num_steps, self.num_envs, dtype=torch.long).to(self.device)
        logp_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        rew_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        done_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        val_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)

        obs, _ = self.envs.reset()
        obs = torch.from_numpy(obs).to(self.device)
        episode_rewards = np.zeros(self.num_envs)
        episode_returns = []

        for step in range(self.num_steps):
            obs_buf[step] = obs

            with torch.no_grad():
                action, log_prob, value = self.flat_policy(obs)

            act_buf[step] = action
            logp_buf[step] = log_prob
            val_buf[step] = value

            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rew_buf[step] = torch.from_numpy(reward).float().to(self.device)
            done_buf[step] = torch.from_numpy(done).float().to(self.device)

            episode_rewards += reward
            for i, d in enumerate(done):
                if d:
                    episode_returns.append(episode_rewards[i])
                    episode_rewards[i] = 0

            obs = torch.from_numpy(next_obs).to(self.device)
            self.global_step += self.num_envs

        # Bootstrap value
        with torch.no_grad():
            _, _, next_value = self.flat_policy(obs)

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value,
            self.gamma, self.gae_lambda
        )

        return {
            'obs': obs_buf.reshape(-1, *self.envs.single_observation_space.shape),
            'actions': act_buf.reshape(-1),
            'old_log_probs': logp_buf.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
        }, episode_returns

    def collect_rollout_hrl(self):
        """Collect rollout data for hierarchical agent (continuous or discrete)."""
        T = self.num_steps
        N = self.num_envs

        # Worker buffers
        w_obs_buf = torch.zeros(T, N, *self.envs.single_observation_space.shape).to(self.device)
        w_feat_buf = torch.zeros(T, N, self.config['encoder']['hidden_dim']).to(self.device)
        w_goal_buf = torch.zeros(T, N, self.config['manager']['goal_dim']).to(self.device)
        w_act_buf = torch.zeros(T, N, dtype=torch.long).to(self.device)
        w_logp_buf = torch.zeros(T, N).to(self.device)
        w_rew_buf = torch.zeros(T, N).to(self.device)
        w_done_buf = torch.zeros(T, N).to(self.device)
        w_val_buf = torch.zeros(T, N).to(self.device)

        # Manager buffers — per-environment to keep trajectories separate
        m_feat_buf = [[] for _ in range(N)]
        m_goal_buf = [[] for _ in range(N)]
        m_logp_buf = [[] for _ in range(N)]
        m_rew_buf = [[] for _ in range(N)]
        m_done_buf = [[] for _ in range(N)]
        m_val_buf = [[] for _ in range(N)]

        # Message tracking for analysis
        messages_log = [] if self.mode == 'discrete' else None

        obs, _ = self.envs.reset()
        obs = torch.from_numpy(obs).to(self.device)

        current_goal = torch.zeros(N, self.config['manager']['goal_dim']).to(self.device)
        manager_extrinsic_reward = torch.zeros(N).to(self.device)
        steps_since_goal = torch.zeros(N, dtype=torch.long).to(self.device)

        episode_rewards = np.zeros(N)
        episode_returns = []

        for step in range(T):
            w_obs_buf[step] = obs

            with torch.no_grad():
                features = self.encoder(obs)
            w_feat_buf[step] = features

            # Manager decision every c steps
            needs_new_goal = (steps_since_goal % self.goal_period == 0)

            if needs_new_goal.any():
                with torch.no_grad():
                    goal, log_prob, value, goal_mean = self.manager(features)

                if self.mode == 'discrete':
                    self._anneal_tau()
                    with torch.no_grad():
                        msg_onehot, msg_indices, msg_logits = self.comm_channel.encode(goal)
                        decoded_goal = self.comm_channel.decode(msg_onehot)

                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(decoded_goal),
                        decoded_goal, current_goal
                    )

                    if messages_log is not None:
                        messages_log.append(msg_indices[needs_new_goal].cpu().numpy())
                else:
                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(goal),
                        goal, current_goal
                    )

                # Log manager data per-environment
                for i in range(N):
                    if needs_new_goal[i]:
                        m_feat_buf[i].append(features[i])
                        m_goal_buf[i].append(goal[i])
                        m_logp_buf[i].append(log_prob[i])
                        m_val_buf[i].append(value[i])
                        m_rew_buf[i].append(manager_extrinsic_reward[i].clone())
                        m_done_buf[i].append(torch.tensor(0.0, device=self.device))
                        manager_extrinsic_reward[i] = 0.0

                current_goal = new_goal

            w_goal_buf[step] = current_goal

            # Worker acts
            with torch.no_grad():
                action, log_prob, value = self.worker(features, current_goal)

            w_act_buf[step] = action
            w_logp_buf[step] = log_prob
            w_val_buf[step] = value

            # Step environment
            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            reward_t = torch.from_numpy(reward).float().to(self.device)
            done_t = torch.from_numpy(done).float().to(self.device)

            # Worker reward = intrinsic (goal-reaching) + extrinsic
            next_obs_t = torch.from_numpy(next_obs).to(self.device)
            with torch.no_grad():
                next_features = self.encoder(next_obs_t)
            with torch.no_grad():
                intrinsic_reward = self._compute_intrinsic_reward(next_features, current_goal)
            worker_reward = self.intrinsic_coef * intrinsic_reward + self.extrinsic_coef * reward_t

            w_rew_buf[step] = worker_reward
            w_done_buf[step] = done_t

            # Accumulate extrinsic reward for manager
            manager_extrinsic_reward += reward_t

            episode_rewards += reward
            for i, d in enumerate(done):
                if d:
                    episode_returns.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    steps_since_goal[i] = 0
                    manager_extrinsic_reward[i] = 0
                    # Mark episode boundary in manager buffer
                    if m_done_buf[i]:
                        m_done_buf[i][-1] = torch.tensor(1.0, device=self.device)

            steps_since_goal += 1
            obs = next_obs_t
            self.global_step += self.num_envs

        # Bootstrap worker value
        with torch.no_grad():
            features = self.encoder(obs)
            next_w_value = self.worker.get_value(features, current_goal)

        w_advantages, w_returns = compute_gae(
            w_rew_buf, w_val_buf, w_done_buf, next_w_value,
            self.gamma, self.gae_lambda
        )

        # Manager GAE — compute per-environment then concatenate
        all_m_features = []
        all_m_goals = []
        all_m_logprobs = []
        all_m_advantages = []
        all_m_returns = []

        for i in range(N):
            n_events = len(m_val_buf[i])
            if n_events < 2:
                continue

            vals = torch.stack(m_val_buf[i])       # (n_events,)
            rews = torch.stack(m_rew_buf[i][1:])    # (n_events-1,) rewards offset by one period
            dones = torch.stack(m_done_buf[i][1:])   # (n_events-1,)

            if len(rews) == 0:
                continue

            adv, ret = compute_gae(rews, vals[:-1], dones, vals[-1],
                                   self.gamma, self.gae_lambda)

            # Align: features/goals/logprobs correspond to vals[:-1] (drop last)
            all_m_features.append(torch.stack(m_feat_buf[i][:-1]))
            all_m_goals.append(torch.stack(m_goal_buf[i][:-1]))
            all_m_logprobs.append(torch.stack(m_logp_buf[i][:-1]))
            all_m_advantages.append(adv)
            all_m_returns.append(ret)

        m_advantages = None
        m_returns = None
        m_features = None
        m_goals = None
        m_logprobs = None

        if all_m_advantages:
            m_features = torch.cat(all_m_features)
            m_goals = torch.cat(all_m_goals)
            m_logprobs = torch.cat(all_m_logprobs)
            m_advantages = torch.cat(all_m_advantages)
            m_returns = torch.cat(all_m_returns)

        return {
            'worker': {
                'obs': w_obs_buf.reshape(-1, *self.envs.single_observation_space.shape),
                'features': w_feat_buf.reshape(-1, self.config['encoder']['hidden_dim']),
                'goals': w_goal_buf.reshape(-1, self.config['manager']['goal_dim']),
                'actions': w_act_buf.reshape(-1),
                'old_log_probs': w_logp_buf.reshape(-1),
                'advantages': w_advantages.reshape(-1),
                'returns': w_returns.reshape(-1),
            },
            'manager': {
                'features': m_features,
                'goals': m_goals,
                'old_log_probs': m_logprobs,
                'advantages': m_advantages,
                'returns': m_returns,
            },
            'messages': messages_log,
        }, episode_returns

    def update_flat(self, rollout):
        """PPO update for flat baseline."""
        B = rollout['obs'].shape[0]
        batch_size = B // self.num_minibatches

        all_stats = defaultdict(list)

        for _ in range(self.update_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                batch = {k: v[mb_idx] for k, v in rollout.items()}

                def policy_fn(b):
                    return self.flat_policy.evaluate_actions(b['obs'], b['actions'])

                stats = ppo_update(
                    batch, policy_fn, self.optimizer,
                    self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                )
                for k, v in stats.items():
                    all_stats[k].append(v)

        return {k: np.mean(v) for k, v in all_stats.items()}

    def update_hrl(self, rollout):
        """PPO update for hierarchical agent (both manager and worker)."""
        w_data = rollout['worker']
        B = w_data['obs'].shape[0]
        batch_size = B // self.num_minibatches

        all_stats = defaultdict(list)

        for _ in range(self.update_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                batch = {k: v[mb_idx] for k, v in w_data.items()}

                def worker_policy_fn(b):
                    features = self.encoder(b['obs'])
                    return self.worker.evaluate_actions(features, b['goals'], b['actions'])

                stats = ppo_update(
                    batch, worker_policy_fn, self.worker_optimizer,
                    self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                )
                for k, v in stats.items():
                    all_stats[f'worker_{k}'].append(v)

        # Manager update (if we have enough data)
        m_data = rollout['manager']
        if m_data['advantages'] is not None and m_data['features'] is not None:
            M = m_data['advantages'].shape[0]
            if M > 4:
                m_batch_size = max(1, M // 2)
                for _ in range(self.update_epochs):
                    indices = torch.randperm(M)
                    for start in range(0, M, m_batch_size):
                        end = min(start + m_batch_size, M)
                        mb_idx = indices[start:end]

                        batch = {
                            'obs_features': m_data['features'][mb_idx],
                            'old_log_probs': m_data['old_log_probs'][mb_idx],
                            'advantages': m_data['advantages'][mb_idx],
                            'returns': m_data['returns'][mb_idx],
                            'goals': m_data['goals'][mb_idx],
                        }

                        def manager_policy_fn(b):
                            return self.manager.evaluate_actions(
                                b['obs_features'], b['goals']
                            )

                        stats = ppo_update(
                            batch, manager_policy_fn, self.manager_optimizer,
                            self.clip_eps, self.entropy_coef * 0.1, self.value_coef, self.max_grad_norm
                        )
                        for k, v in stats.items():
                            all_stats[f'manager_{k}'].append(v)

        return {k: np.mean(v) for k, v in all_stats.items()}

    def train(self, output_dir='outputs', wandb_run=None):
        """Main training loop."""
        os.makedirs(output_dir, exist_ok=True)

        num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        start_time = time.time()

        all_returns = []
        all_messages = []

        print(f"Training mode: {self.mode}")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Num updates: {num_updates}")
        print(f"Device: {self.device}")
        print()

        lr = self.config['ppo']['lr']

        for update in range(1, num_updates + 1):
            # Learning rate annealing
            if self.config['ppo']['anneal_lr']:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * self.config['ppo']['lr']
                if self.mode == 'flat':
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr
                else:
                    for pg in self.worker_optimizer.param_groups:
                        pg['lr'] = lr
                    for pg in self.manager_optimizer.param_groups:
                        pg['lr'] = lr

            # Collect rollout
            if self.mode == 'flat':
                rollout, ep_returns = self.collect_rollout_flat()
                stats = self.update_flat(rollout)
            else:
                rollout, ep_returns = self.collect_rollout_hrl()
                stats = self.update_hrl(rollout)

                if rollout.get('messages') and rollout['messages']:
                    all_messages.extend(rollout['messages'])

            all_returns.extend(ep_returns)

            # Logging
            if update % self.config['experiment']['log_interval'] == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed

                recent_returns = all_returns[-50:] if all_returns else [0]
                mean_return = np.mean(recent_returns)

                print(f"Update {update}/{num_updates} | "
                      f"Step {self.global_step} | "
                      f"Mean Return: {mean_return:.2f} | "
                      f"SPS: {sps:.0f} | "
                      f"Loss: {stats.get('loss', stats.get('worker_loss', 0)):.4f}")

                if wandb_run is not None:
                    log_data = {
                        'global_step': self.global_step,
                        'mean_return': mean_return,
                        'sps': sps,
                        'learning_rate': lr if self.config['ppo']['anneal_lr'] else self.config['ppo']['lr'],
                        'episodes': len(all_returns),
                    }
                    log_data.update(stats)
                    wandb_run.log(log_data, step=self.global_step)

            # Save checkpoint
            if self.global_step > 0 and self.global_step % self.config['experiment']['save_interval'] == 0:
                self.save(os.path.join(output_dir, f'checkpoint_{self.global_step}.pt'))

        # Final save
        self.save(os.path.join(output_dir, 'final.pt'))

        return {
            'returns': all_returns,
            'messages': all_messages,
        }

    def save(self, path):
        """Save model checkpoint."""
        state = {
            'mode': self.mode,
            'global_step': self.global_step,
            'config': self.config,
        }

        if self.mode == 'flat':
            state['flat_policy'] = self.flat_policy.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
        else:
            state['encoder'] = self.encoder.state_dict()
            state['manager'] = self.manager.state_dict()
            state['worker'] = self.worker.state_dict()
            state['goal_projection'] = self.goal_projection.state_dict()
            state['worker_optimizer'] = self.worker_optimizer.state_dict()
            state['manager_optimizer'] = self.manager_optimizer.state_dict()
            if self.mode == 'discrete':
                state['comm_channel'] = self.comm_channel.state_dict()

        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        if self.mode == 'flat':
            self.flat_policy.load_state_dict(state['flat_policy'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            self.encoder.load_state_dict(state['encoder'])
            self.manager.load_state_dict(state['manager'])
            self.worker.load_state_dict(state['worker'])
            if 'goal_projection' in state:
                self.goal_projection.load_state_dict(state['goal_projection'])
            self.worker_optimizer.load_state_dict(state['worker_optimizer'])
            self.manager_optimizer.load_state_dict(state['manager_optimizer'])
            if self.mode == 'discrete' and 'comm_channel' in state:
                self.comm_channel.load_state_dict(state['comm_channel'])

        self.global_step = state['global_step']
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
