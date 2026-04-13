"""Single-agent HRL training loop.

Supports multiple modes:
- flat: Standard PPO (no hierarchy, no goals)
- continuous: HRL with continuous goals, TD3 or SAC manager + PPO worker
- discrete: HRL with discrete bottleneck goals, PPO for both
- option_critic: HRL with learned termination via Option-Critic
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from models.encoder import MinigridEncoder
from models.manager import Manager
from models.worker import Worker
from models.communication import CommunicationChannel
from algos.ppo import compute_gae, ppo_update
from algos.td3 import ManagerTD3
from envs.wrappers import make_env, make_vec_env


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

    def __init__(self, config, mode='continuous', device='cuda', use_corridor=False):
        self.config = config
        self.mode = mode
        self.use_corridor = use_corridor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create environments
        env_cfg = config['env']
        ppo_cfg = config['ppo']
        self.env_name = env_cfg['name']
        self.max_steps = env_cfg.get('max_steps')
        self.corridor_width = env_cfg.get('corridor_width', 3)
        self.corridor_size = env_cfg.get('corridor_size', 11)
        self.eval_episodes = config['experiment'].get('eval_episodes', 20)
        if use_corridor:
            from envs.multi_agent_env import make_corridor_vec_env
            self.envs = make_corridor_vec_env(
                ppo_cfg['num_envs'],
                seed=config['experiment']['seed'],
                max_steps=env_cfg.get('max_steps', 200),
                corridor_width=self.corridor_width,
                corridor_size=self.corridor_size,
            )
        else:
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

        # Option-Critic and SAC flags
        self.use_option_critic = config['manager'].get('use_option_critic', False)
        sac_cfg = config.get('sac', {})
        self.use_sac = sac_cfg.get('enabled', False) and mode == 'continuous'

        # Build models
        if mode == 'flat':
            self.flat_policy = FlatPolicy(obs_shape, num_actions,
                                          config['encoder']['hidden_dim']).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.flat_policy.parameters(), lr=ppo_cfg['lr'], eps=1e-5
            )
        else:
            hidden_dim = config['encoder']['hidden_dim']
            goal_dim = config['manager']['goal_dim']

            self.encoder = MinigridEncoder(
                obs_shape,
                channels=tuple(config['encoder']['channels']),
                hidden_dim=hidden_dim
            ).to(self.device)

            self.worker = Worker(
                input_dim=hidden_dim,
                goal_dim=goal_dim,
                num_actions=num_actions,
                hidden_dim=config['worker']['hidden_dim'],
            ).to(self.device)

            # Projection from encoder features to goal space for intrinsic reward
            self.goal_projection = nn.Linear(hidden_dim, goal_dim).to(self.device)

            worker_params = (
                list(self.encoder.parameters()) +
                list(self.worker.parameters()) +
                list(self.goal_projection.parameters())
            )

            if self.use_option_critic:
                # Option-Critic: learned termination
                from models.option_critic import OptionCriticManager
                oc_cfg = config['manager'].get('option_critic', {})
                self.option_critic = OptionCriticManager(
                    input_dim=hidden_dim,
                    goal_dim=goal_dim,
                    num_options=oc_cfg.get('num_options', 8),
                    hidden_dim=config['manager']['hidden_dim'],
                    termination_reg=oc_cfg.get('termination_reg', 0.01),
                ).to(self.device)

                comm_cfg = config['communication']
                self.comm_channel = CommunicationChannel(
                    goal_dim=goal_dim,
                    vocab_size=comm_cfg['vocab_size'],
                    message_length=comm_cfg['message_length'],
                    tau=comm_cfg['tau_start'],
                ).to(self.device)
                self.comm_optimizer = torch.optim.Adam(
                    self.comm_channel.parameters(), lr=ppo_cfg['lr'], eps=1e-5
                )

                self.manager_optimizer = torch.optim.Adam(
                    self.option_critic.parameters(), lr=ppo_cfg['lr'] * 0.1, eps=1e-5
                )

                self.tau_start = comm_cfg['tau_start']
                self.tau_end = comm_cfg['tau_end']
                self.tau_anneal_steps = comm_cfg['tau_anneal_steps']

            elif mode == 'continuous':
                if self.use_sac:
                    from algos.sac import ManagerSAC
                    self.manager_sac = ManagerSAC(
                        state_dim=hidden_dim,
                        goal_dim=goal_dim,
                        hidden_dim=config['manager']['hidden_dim'],
                        device=self.device,
                        lr=sac_cfg.get('lr', ppo_cfg['lr']),
                        gamma=ppo_cfg['gamma'],
                        tau=sac_cfg.get('tau', 0.005),
                        alpha=sac_cfg.get('alpha', 0.2),
                        auto_alpha=sac_cfg.get('auto_alpha', True),
                        target_entropy=sac_cfg.get('target_entropy'),
                        buffer_size=sac_cfg.get('buffer_size', 200000),
                        batch_size=sac_cfg.get('batch_size', 256),
                        warmup_steps=sac_cfg.get('warmup_steps', 1000),
                    )
                else:
                    # TD3 for the manager (off-policy, handles sparse updates)
                    self.manager_td3 = ManagerTD3(
                        state_dim=hidden_dim,
                        goal_dim=goal_dim,
                        hidden_dim=config['manager']['hidden_dim'],
                        device=self.device,
                        lr=ppo_cfg['lr'],
                        gamma=ppo_cfg['gamma'],
                    )

            elif mode == 'discrete':
                # PPO manager for discrete goals (Gumbel-Softmax makes it finite)
                self.manager = Manager(
                    input_dim=hidden_dim,
                    goal_dim=goal_dim,
                    hidden_dim=config['manager']['hidden_dim'],
                ).to(self.device)

                comm_cfg = config['communication']
                self.comm_channel = CommunicationChannel(
                    goal_dim=goal_dim,
                    vocab_size=comm_cfg['vocab_size'],
                    message_length=comm_cfg['message_length'],
                    tau=comm_cfg['tau_start'],
                ).to(self.device)
                # Comm channel gets its own optimizer to avoid stale gradient corruption
                self.comm_optimizer = torch.optim.Adam(
                    self.comm_channel.parameters(), lr=ppo_cfg['lr'], eps=1e-5
                )

                self.manager_optimizer = torch.optim.Adam(
                    self.manager.parameters(), lr=ppo_cfg['lr'] * 0.1, eps=1e-5
                )

                self.tau_start = comm_cfg['tau_start']
                self.tau_end = comm_cfg['tau_end']
                self.tau_anneal_steps = comm_cfg['tau_anneal_steps']

            self.worker_optimizer = torch.optim.Adam(
                worker_params, lr=ppo_cfg['lr'], eps=1e-5
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

        # Intrinsic reward annealing
        self.intrinsic_anneal = config['worker'].get('intrinsic_anneal', False)
        self.intrinsic_anneal_steps = config['worker'].get('intrinsic_anneal_steps', 500000)

        # Logging
        self.global_step = 0
        self.log = defaultdict(list)

        # Persistent obs across rollouts
        obs, _ = self.envs.reset()
        self.obs = torch.from_numpy(obs).to(self.device)

    def _get_intrinsic_coef(self):
        """Get current intrinsic reward coefficient (with optional annealing)."""
        if not self.intrinsic_anneal:
            return self.intrinsic_coef
        frac = min(1.0, self.global_step / self.intrinsic_anneal_steps)
        return self.intrinsic_coef * (1.0 - frac)

    def _anneal_tau(self):
        """Anneal Gumbel-Softmax temperature (discrete/option_critic mode only)."""
        if self.mode not in ('discrete',) and not self.use_option_critic:
            return
        frac = min(1.0, self.global_step / self.tau_anneal_steps)
        tau = self.tau_start + frac * (self.tau_end - self.tau_start)
        self.comm_channel.set_tau(tau)
        return tau

    def _compute_intrinsic_reward(self, obs_features, goal):
        """Worker's intrinsic reward: negative distance on unit sphere."""
        projected = F.normalize(self.goal_projection(obs_features), dim=-1)
        goal_norm = F.normalize(goal, dim=-1)
        return -torch.norm(projected - goal_norm, dim=-1)

    # ------------------------------------------------------------------
    # Flat PPO rollout + update (unchanged)
    # ------------------------------------------------------------------

    def collect_rollout_flat(self):
        obs_buf = torch.zeros(self.num_steps, self.num_envs, *self.envs.single_observation_space.shape).to(self.device)
        act_buf = torch.zeros(self.num_steps, self.num_envs, dtype=torch.long).to(self.device)
        logp_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        rew_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        done_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        val_buf = torch.zeros(self.num_steps, self.num_envs).to(self.device)

        obs = self.obs
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

        self.obs = obs
        with torch.no_grad():
            _, _, next_value = self.flat_policy(obs)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, self.gamma, self.gae_lambda
        )
        return {
            'obs': obs_buf.reshape(-1, *self.envs.single_observation_space.shape),
            'actions': act_buf.reshape(-1),
            'old_log_probs': logp_buf.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
        }, episode_returns

    def update_flat(self, rollout):
        B = rollout['obs'].shape[0]
        batch_size = B // self.num_minibatches
        all_stats = defaultdict(list)
        for _ in range(self.update_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, batch_size):
                mb_idx = indices[start:start + batch_size]
                batch = {k: v[mb_idx] for k, v in rollout.items()}
                def policy_fn(b):
                    return self.flat_policy.evaluate_actions(b['obs'], b['actions'])
                stats = ppo_update(
                    batch, policy_fn, self.optimizer,
                    self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                )
                for k, v in stats.items():
                    all_stats[f'worker_{k}'].append(v)
        return {k: np.mean(v) for k, v in all_stats.items()}

    # ------------------------------------------------------------------
    # HRL rollout collection (shared by continuous, discrete, option_critic)
    # ------------------------------------------------------------------

    def collect_rollout_hrl(self):
        T = self.num_steps
        N = self.num_envs

        # Worker buffers
        w_obs_buf = torch.zeros(T, N, *self.envs.single_observation_space.shape).to(self.device)
        w_goal_buf = torch.zeros(T, N, self.config['manager']['goal_dim']).to(self.device)
        w_act_buf = torch.zeros(T, N, dtype=torch.long).to(self.device)
        w_logp_buf = torch.zeros(T, N).to(self.device)
        w_rew_buf = torch.zeros(T, N).to(self.device)
        w_done_buf = torch.zeros(T, N).to(self.device)
        w_val_buf = torch.zeros(T, N).to(self.device)

        # Manager buffers — per-environment
        m_feat_buf = [[] for _ in range(N)]
        m_goal_buf = [[] for _ in range(N)]
        m_rew_buf = [[] for _ in range(N)]
        m_done_buf = [[] for _ in range(N)]
        # PPO-specific (discrete/option_critic only)
        is_ppo_manager = self.mode == 'discrete' or self.use_option_critic
        m_logp_buf = [[] for _ in range(N)] if is_ppo_manager else None
        m_val_buf = [[] for _ in range(N)] if is_ppo_manager else None

        # Option-Critic specific buffers
        m_beta_buf = [[] for _ in range(N)] if self.use_option_critic else None
        m_option_buf = [[] for _ in range(N)] if self.use_option_critic else None

        messages_log = [] if (self.mode == 'discrete' or self.use_option_critic) else None
        states_log = [] if (self.mode == 'discrete' or self.use_option_critic) else None

        obs = self.obs
        current_goal = torch.zeros(N, self.config['manager']['goal_dim']).to(self.device)
        manager_extrinsic_reward = torch.zeros(N).to(self.device)
        steps_since_goal = torch.zeros(N, dtype=torch.long).to(self.device)

        # Option-Critic state
        if self.use_option_critic:
            current_option = torch.zeros(N, dtype=torch.long).to(self.device)

        episode_rewards = np.zeros(N)
        episode_returns = []

        for step in range(T):
            w_obs_buf[step] = obs

            with torch.no_grad():
                features = self.encoder(obs)

            # Determine which envs need a new goal
            if self.use_option_critic:
                if step == 0:
                    needs_new_goal = torch.ones(N, dtype=torch.bool, device=self.device)
                    betas = torch.zeros(N, device=self.device)
                else:
                    terminate, betas = self.option_critic.should_terminate(features, current_option)
                    needs_new_goal = terminate | (steps_since_goal == 0)
            else:
                needs_new_goal = (steps_since_goal % self.goal_period == 0)

            if needs_new_goal.any():
                if self.use_option_critic:
                    # Option-Critic: select option with learned termination
                    with torch.no_grad():
                        option, log_prob, value, goal = self.option_critic(features)
                        self._anneal_tau()
                        msg_onehot, msg_indices, _ = self.comm_channel.encode(goal)
                        decoded_goal = self.comm_channel.decode(msg_onehot)

                    new_option = torch.where(needs_new_goal, option, current_option)
                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(decoded_goal),
                        decoded_goal, current_goal
                    )

                    if messages_log is not None:
                        messages_log.append(msg_indices[needs_new_goal].cpu().numpy())
                    if states_log is not None:
                        states_log.append(features[needs_new_goal].detach().cpu().numpy())

                    for i in range(N):
                        if needs_new_goal[i]:
                            m_feat_buf[i].append(features[i])
                            m_goal_buf[i].append(goal[i])
                            m_logp_buf[i].append(log_prob[i])
                            m_val_buf[i].append(value[i])
                            m_rew_buf[i].append(manager_extrinsic_reward[i].clone())
                            m_done_buf[i].append(torch.tensor(0.0, device=self.device))
                            m_option_buf[i].append(option[i])
                            m_beta_buf[i].append(betas[i])
                            manager_extrinsic_reward[i] = 0.0

                    current_option = new_option
                    current_goal = new_goal
                    steps_since_goal = torch.where(needs_new_goal,
                                                    torch.zeros_like(steps_since_goal),
                                                    steps_since_goal)

                elif self.mode == 'continuous':
                    # TD3 or SAC deterministic/stochastic policy + exploration
                    with torch.no_grad():
                        if self.use_sac:
                            goal = self.manager_sac.select_goal(features, add_noise=True)
                        else:
                            goal = self.manager_td3.select_goal(features, add_noise=True)
                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(goal),
                        goal, current_goal
                    )
                    # Store manager data per-env
                    for i in range(N):
                        if needs_new_goal[i]:
                            m_feat_buf[i].append(features[i].detach())
                            m_goal_buf[i].append(goal[i].detach())
                            m_rew_buf[i].append(manager_extrinsic_reward[i].item())
                            m_done_buf[i].append(0.0)
                            manager_extrinsic_reward[i] = 0.0

                elif self.mode == 'discrete':
                    with torch.no_grad():
                        goal, log_prob, value, _ = self.manager(features)
                    self._anneal_tau()
                    with torch.no_grad():
                        msg_onehot, msg_indices, _ = self.comm_channel.encode(goal)
                        decoded_goal = self.comm_channel.decode(msg_onehot)
                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(decoded_goal),
                        decoded_goal, current_goal
                    )
                    if messages_log is not None:
                        messages_log.append(msg_indices[needs_new_goal].cpu().numpy())
                    if states_log is not None:
                        states_log.append(features[needs_new_goal].detach().cpu().numpy())
                    for i in range(N):
                        if needs_new_goal[i]:
                            m_feat_buf[i].append(features[i])
                            m_goal_buf[i].append(goal[i])
                            m_logp_buf[i].append(log_prob[i])
                            m_val_buf[i].append(value[i])
                            m_rew_buf[i].append(manager_extrinsic_reward[i].clone())
                            m_done_buf[i].append(torch.tensor(0.0, device=self.device))
                            manager_extrinsic_reward[i] = 0.0

                if not self.use_option_critic:
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

            # Worker reward
            next_obs_t = torch.from_numpy(next_obs).to(self.device)
            with torch.no_grad():
                next_features = self.encoder(next_obs_t)
                intrinsic_reward = self._compute_intrinsic_reward(next_features, current_goal)
            worker_reward = self._get_intrinsic_coef() * intrinsic_reward + self.extrinsic_coef * reward_t

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
                    if self.use_option_critic:
                        if m_done_buf[i]:
                            m_done_buf[i][-1] = torch.tensor(1.0, device=self.device)
                    elif self.mode == 'discrete' and m_done_buf[i]:
                        m_done_buf[i][-1] = torch.tensor(1.0, device=self.device)
                    elif self.mode == 'continuous' and m_done_buf[i]:
                        m_done_buf[i][-1] = 1.0
                    manager_extrinsic_reward[i] = 0

            if not self.use_option_critic:
                steps_since_goal += 1
            else:
                steps_since_goal += 1
            obs = next_obs_t
            self.global_step += self.num_envs

        self.obs = obs

        # Bootstrap worker value
        with torch.no_grad():
            features = self.encoder(obs)
            next_w_value = self.worker.get_value(features, current_goal)

        w_advantages, w_returns = compute_gae(
            w_rew_buf, w_val_buf, w_done_buf, next_w_value,
            self.gamma, self.gae_lambda
        )

        # Build manager data depending on mode
        manager_data = {}

        if self.mode == 'continuous' and not self.use_option_critic:
            # Feed transitions to TD3/SAC replay buffer
            manager_obj = self.manager_sac if self.use_sac else self.manager_td3
            for i in range(N):
                n_events = len(m_feat_buf[i])
                if n_events < 2:
                    continue
                for j in range(n_events - 1):
                    manager_obj.add_transition(
                        state=m_feat_buf[i][j].unsqueeze(0),
                        goal=m_goal_buf[i][j].unsqueeze(0),
                        reward=torch.tensor([m_rew_buf[i][j + 1]]),
                        next_state=m_feat_buf[i][j + 1].unsqueeze(0),
                        done=torch.tensor([m_done_buf[i][j]]),
                    )

        elif self.mode == 'discrete' or self.use_option_critic:
            # Build PPO data for manager
            all_m_features, all_m_goals, all_m_logprobs = [], [], []
            all_m_advantages, all_m_returns = [], []
            all_m_options = [] if self.use_option_critic else None

            for i in range(N):
                n_events = len(m_val_buf[i])
                if n_events < 2:
                    continue
                vals = torch.stack(m_val_buf[i])
                rews = torch.stack(m_rew_buf[i][1:])
                dones = torch.stack(m_done_buf[i][1:])
                if len(rews) == 0:
                    continue
                adv, ret = compute_gae(rews, vals[:-1], dones, vals[-1],
                                       self.gamma, self.gae_lambda)
                all_m_features.append(torch.stack(m_feat_buf[i][:-1]))
                all_m_goals.append(torch.stack(m_goal_buf[i][:-1]))
                all_m_logprobs.append(torch.stack(m_logp_buf[i][:-1]))
                all_m_advantages.append(adv)
                all_m_returns.append(ret)
                if self.use_option_critic:
                    all_m_options.append(torch.stack(m_option_buf[i][:-1]))

            if all_m_advantages:
                manager_data = {
                    'features': torch.cat(all_m_features),
                    'goals': torch.cat(all_m_goals),
                    'old_log_probs': torch.cat(all_m_logprobs),
                    'advantages': torch.cat(all_m_advantages),
                    'returns': torch.cat(all_m_returns),
                }
                if self.use_option_critic and all_m_options:
                    manager_data['options'] = torch.cat(all_m_options)

        return {
            'worker': {
                'obs': w_obs_buf.reshape(-1, *self.envs.single_observation_space.shape),
                'goals': w_goal_buf.reshape(-1, self.config['manager']['goal_dim']),
                'actions': w_act_buf.reshape(-1),
                'old_log_probs': w_logp_buf.reshape(-1),
                'advantages': w_advantages.reshape(-1),
                'returns': w_returns.reshape(-1),
            },
            'manager': manager_data,
            'messages': messages_log,
            'states': states_log,
        }, episode_returns

    # ------------------------------------------------------------------
    # HRL update
    # ------------------------------------------------------------------

    def update_hrl(self, rollout):
        w_data = rollout['worker']
        B = w_data['obs'].shape[0]
        batch_size = B // self.num_minibatches

        all_stats = defaultdict(list)

        # Worker PPO update
        for _ in range(self.update_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, batch_size):
                mb_idx = indices[start:start + batch_size]
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

        # Manager update
        if self.mode == 'continuous' and not self.use_option_critic:
            # TD3/SAC updates
            manager_obj = self.manager_sac if self.use_sac else self.manager_td3
            for _ in range(self.goal_period):
                mgr_stats = manager_obj.update()
                if mgr_stats is not None:
                    for k, v in mgr_stats.items():
                        all_stats[k].append(v)

        elif self.mode == 'discrete' or self.use_option_critic:
            m_data = rollout['manager']
            if m_data and m_data.get('advantages') is not None:
                M = m_data['advantages'].shape[0]
                if M > 4:
                    m_batch_size = max(1, M // 2)
                    for _ in range(1):
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
                            if self.use_option_critic:
                                batch['options'] = m_data['options'][mb_idx]
                                def manager_policy_fn(b):
                                    return self.option_critic.evaluate_actions(
                                        b['obs_features'], b['options']
                                    )
                            else:
                                def manager_policy_fn(b):
                                    return self.manager.evaluate_actions(
                                        b['obs_features'], b['goals']
                                    )
                            stats = ppo_update(
                                batch, manager_policy_fn, self.manager_optimizer,
                                0.1, self.entropy_coef * 0.1, self.value_coef, self.max_grad_norm
                            )
                            for k, v in stats.items():
                                all_stats[f'manager_{k}'].append(v)

            # Communication channel reconstruction loss
            if m_data and m_data.get('goals') is not None:
                raw_goals = m_data['goals'].detach()
                msg_onehot, _, logits = self.comm_channel.encode(raw_goals)
                reconstructed = self.comm_channel.decode(msg_onehot)
                recon_loss = F.mse_loss(reconstructed, raw_goals)

                # Sender entropy bonus to encourage diverse messages
                probs = F.softmax(logits, dim=-1)
                sender_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                comm_loss = recon_loss - 0.05 * sender_entropy

                self.comm_optimizer.zero_grad()
                comm_loss.backward()
                nn.utils.clip_grad_norm_(self.comm_channel.parameters(), self.max_grad_norm)
                self.comm_optimizer.step()

                all_stats['comm_recon_loss'].append(recon_loss.item())
                all_stats['comm_sender_entropy'].append(sender_entropy.item())

        return {k: np.mean(v) for k, v in all_stats.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _make_eval_env(self, seed):
        """Create a single evaluation environment."""
        if self.use_corridor:
            from envs.multi_agent_env import SingleAgentCorridorEnv

            return SingleAgentCorridorEnv(
                size=self.corridor_size,
                corridor_length=3,
                max_steps=self.max_steps or 200,
                corridor_width=self.corridor_width,
            )
        return make_env(self.env_name, seed=seed, max_steps=self.max_steps)()

    def _select_flat_action(self, obs):
        features = self.flat_policy.encoder(obs)
        logits = self.flat_policy.policy(features)
        return logits.argmax(dim=-1)

    def _select_worker_action(self, features, goal):
        logits = self.worker.policy(torch.cat([features, goal], dim=-1))
        return logits.argmax(dim=-1)

    def _select_discrete_goal(self, features):
        h = self.manager.policy(features)
        goal_mean = self.manager.goal_mean(h)
        msg_onehot, _, _ = self.comm_channel.encode(goal_mean)
        return self.comm_channel.decode(msg_onehot)

    def _select_option_goal(self, features, current_option=None, steps_since_goal=0):
        if current_option is None:
            option_logits = self.option_critic.policy_over_options(features)
            current_option = option_logits.argmax(dim=-1)
        elif steps_since_goal > 0:
            term_logits = self.option_critic.termination_head(features)
            beta = torch.sigmoid(
                term_logits.gather(1, current_option.unsqueeze(-1)).squeeze(-1)
            )
            if bool((beta > 0.5).item()):
                option_logits = self.option_critic.policy_over_options(features)
                current_option = option_logits.argmax(dim=-1)

        goal = self.option_critic.option_goals(current_option)
        msg_onehot, _, _ = self.comm_channel.encode(goal)
        return current_option, self.comm_channel.decode(msg_onehot)

    def evaluate(self, num_episodes=20):
        """Evaluate the current policy on the training task."""
        returns = []
        successes = []
        base_seed = self.config['experiment']['seed'] + 100_000

        for episode in range(num_episodes):
            env = self._make_eval_env(base_seed + episode)
            obs, _ = env.reset(seed=base_seed + episode)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            current_goal = None
            current_option = None
            steps_since_goal = 0
            done = False
            ep_return = 0.0

            while not done:
                with torch.no_grad():
                    if self.mode == 'flat':
                        action = self._select_flat_action(obs_t)
                    else:
                        features = self.encoder(obs_t)
                        if self.use_option_critic:
                            current_option, current_goal = self._select_option_goal(
                                features, current_option=current_option,
                                steps_since_goal=steps_since_goal,
                            )
                        elif self.mode == 'continuous' and current_goal is None:
                            manager = self.manager_sac if self.use_sac else self.manager_td3
                            current_goal = manager.select_goal(features, add_noise=False)
                        elif self.mode == 'continuous' and steps_since_goal % self.goal_period == 0:
                            manager = self.manager_sac if self.use_sac else self.manager_td3
                            current_goal = manager.select_goal(features, add_noise=False)
                        elif self.mode == 'discrete' and (
                            current_goal is None or steps_since_goal % self.goal_period == 0
                        ):
                            current_goal = self._select_discrete_goal(features)

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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, output_dir='outputs', wandb_run=None):
        os.makedirs(output_dir, exist_ok=True)

        num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        start_time = time.time()

        all_returns = []
        all_messages = []
        all_states = []

        mode_name = 'option_critic' if self.use_option_critic else self.mode
        if self.use_sac:
            mode_name = 'sac_continuous'
        print(f"Training mode: {mode_name}")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Num updates: {num_updates}")
        print(f"Device: {self.device}")
        if self.intrinsic_anneal:
            print(f"Intrinsic reward annealing: {self.intrinsic_coef} -> 0 over {self.intrinsic_anneal_steps} steps")
        print()

        lr = self.config['ppo']['lr']

        for update in range(1, num_updates + 1):
            # Learning rate annealing (worker + flat only)
            if self.config['ppo']['anneal_lr']:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * self.config['ppo']['lr']
                if self.mode == 'flat':
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr
                elif self.mode in ('continuous', 'discrete') or self.use_option_critic:
                    for pg in self.worker_optimizer.param_groups:
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
                if rollout.get('states') and rollout['states']:
                    all_states.extend(rollout['states'])

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
                      f"Loss: {stats.get('worker_loss', 0):.4f}")

                if wandb_run is not None:
                    log_data = {
                        'global_step': self.global_step,
                        'mean_return': mean_return,
                        'sps': sps,
                        'learning_rate': lr,
                        'episodes': len(all_returns),
                        'intrinsic_coef': self._get_intrinsic_coef(),
                    }
                    log_data.update(stats)
                    wandb_run.log(log_data, step=self.global_step)

            # Save checkpoint
            if self.global_step > 0 and self.global_step % self.config['experiment']['save_interval'] == 0:
                self.save(os.path.join(output_dir, f'checkpoint_{self.global_step}.pt'))

        # Final save
        self.save(os.path.join(output_dir, 'final.pt'))
        final_eval = self.evaluate(num_episodes=self.eval_episodes)
        print(
            f"Final eval: return={final_eval['mean_return']:.3f} +/- "
            f"{final_eval['std_return']:.3f} | success={100 * final_eval['success_rate']:.1f}%"
        )

        return {
            'returns': all_returns,
            'messages': all_messages,
            'states': all_states,
            'eval': final_eval,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        state = {
            'mode': self.mode,
            'global_step': self.global_step,
            'config': self.config,
            'use_option_critic': self.use_option_critic,
            'use_sac': self.use_sac,
        }

        if self.mode == 'flat':
            state['flat_policy'] = self.flat_policy.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
        else:
            state['encoder'] = self.encoder.state_dict()
            state['worker'] = self.worker.state_dict()
            state['goal_projection'] = self.goal_projection.state_dict()
            state['worker_optimizer'] = self.worker_optimizer.state_dict()

            if self.use_option_critic:
                state['option_critic'] = self.option_critic.state_dict()
                state['manager_optimizer'] = self.manager_optimizer.state_dict()
                state['comm_channel'] = self.comm_channel.state_dict()
                state['comm_optimizer'] = self.comm_optimizer.state_dict()
            elif self.mode == 'continuous':
                if self.use_sac:
                    state['manager_sac'] = self.manager_sac.state_dict()
                else:
                    state['manager_td3'] = self.manager_td3.state_dict()
            elif self.mode == 'discrete':
                state['manager'] = self.manager.state_dict()
                state['manager_optimizer'] = self.manager_optimizer.state_dict()
                state['comm_channel'] = self.comm_channel.state_dict()
                state['comm_optimizer'] = self.comm_optimizer.state_dict()

        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)

        if self.mode == 'flat':
            self.flat_policy.load_state_dict(state['flat_policy'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            self.encoder.load_state_dict(state['encoder'])
            self.worker.load_state_dict(state['worker'])
            if 'goal_projection' in state:
                self.goal_projection.load_state_dict(state['goal_projection'])
            self.worker_optimizer.load_state_dict(state['worker_optimizer'])

            if self.use_option_critic and 'option_critic' in state:
                self.option_critic.load_state_dict(state['option_critic'])
                self.manager_optimizer.load_state_dict(state['manager_optimizer'])
                self.comm_channel.load_state_dict(state['comm_channel'])
            elif self.mode == 'continuous':
                if self.use_sac and 'manager_sac' in state:
                    self.manager_sac.load_state_dict(state['manager_sac'])
                elif 'manager_td3' in state:
                    self.manager_td3.load_state_dict(state['manager_td3'])
            elif self.mode == 'discrete':
                self.manager.load_state_dict(state['manager'])
                self.manager_optimizer.load_state_dict(state['manager_optimizer'])
                if 'comm_channel' in state:
                    self.comm_channel.load_state_dict(state['comm_channel'])

        self.global_step = state['global_step']
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
