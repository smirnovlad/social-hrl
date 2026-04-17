"""MAPPO training loop for social HRL (two communicating agents).

Each agent has its own encoder, manager, worker, and communication channel.
Agent A's message is embedded and fed to Agent B's manager (and vice versa).
A shared centralized critic sees both agents' features for variance reduction.

Features:
- Listener reward: reward agent for partner following its message
- Communication ablation: zero out messages at eval time
- Intrinsic reward annealing
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
from analysis.goal_metrics import temporal_extent as _temporal_extent
from envs.multi_agent_env import MultiAgentWrapper


class SharedCritic(nn.Module):
    """Centralized critic that observes both agents' features."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features_a, features_b):
        x = torch.cat([features_a, features_b], dim=-1)
        return self.net(x).squeeze(-1)


class MultiAgentTrainer:
    """MAPPO training for two communicating HRL agents."""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        env_cfg = config['env']
        ppo_cfg = config['ppo']
        comm_cfg = config['communication']
        self.max_steps = env_cfg.get('max_steps', 200)
        self.corridor_size = env_cfg.get('corridor_size', 11)
        self.corridor_width = env_cfg.get('corridor_width', 3)
        self.rendezvous_bonus = env_cfg.get('rendezvous_bonus', 0.0)
        self.num_obstacles = env_cfg.get('num_obstacles', 0)
        self.bus_cost_solo = env_cfg.get('bus_cost_solo', 0.0)
        self.bus_cost_shared = env_cfg.get('bus_cost_shared', 0.0)
        self.bus_window = env_cfg.get('bus_window', 0)
        self.turn_taking = env_cfg.get('turn_taking', False)
        self.eval_episodes = config['experiment'].get('eval_episodes', 20)
        hidden_dim = config['encoder']['hidden_dim']
        goal_dim = config['manager']['goal_dim']
        message_dim = goal_dim  # embed_message outputs goal_dim

        # Create environments (N independent two-agent envs)
        self.num_envs = ppo_cfg['num_envs']
        self.envs = [
            MultiAgentWrapper(
                size=self.corridor_size, corridor_length=3,
                max_steps=self.max_steps,
                seed=config['experiment']['seed'] + i,
                corridor_width=self.corridor_width,
                asymmetric_info=env_cfg.get('asymmetric_info', False),
                rendezvous_bonus=self.rendezvous_bonus,
                num_obstacles=self.num_obstacles,
                bus_cost_solo=self.bus_cost_solo,
                bus_cost_shared=self.bus_cost_shared,
                bus_window=self.bus_window,
                turn_taking=self.turn_taking,
            )
            for i in range(self.num_envs)
        ]

        obs_shape = self.envs[0].observation_space.shape
        num_actions = self.envs[0].action_space.n

        self.num_steps = ppo_cfg['num_steps']
        self.goal_period = config['manager']['goal_period']

        # Per-agent models (separate instances)
        self.encoders = nn.ModuleList([
            MinigridEncoder(obs_shape, channels=tuple(config['encoder']['channels']),
                           hidden_dim=hidden_dim)
            for _ in range(2)
        ]).to(self.device)

        self.managers = nn.ModuleList([
            Manager(input_dim=hidden_dim, goal_dim=goal_dim,
                   hidden_dim=config['manager']['hidden_dim'],
                   message_dim=message_dim)
            for _ in range(2)
        ]).to(self.device)

        self.workers = nn.ModuleList([
            Worker(input_dim=hidden_dim, goal_dim=goal_dim,
                  num_actions=num_actions, hidden_dim=config['worker']['hidden_dim'])
            for _ in range(2)
        ]).to(self.device)

        self.comm_channels = nn.ModuleList([
            CommunicationChannel(goal_dim=goal_dim, vocab_size=comm_cfg['vocab_size'],
                                message_length=comm_cfg['message_length'],
                                tau=comm_cfg['tau_start'])
            for _ in range(2)
        ]).to(self.device)

        self.goal_projections = nn.ModuleList([
            nn.Linear(hidden_dim, goal_dim)
            for _ in range(2)
        ]).to(self.device)

        # Shared centralized critic (for manager advantage)
        self.shared_critic = SharedCritic(hidden_dim).to(self.device)

        # Single optimizer for all parameters
        main_params = (
            list(self.encoders.parameters()) +
            list(self.managers.parameters()) +
            list(self.workers.parameters()) +
            list(self.goal_projections.parameters()) +
            list(self.shared_critic.parameters())
        )
        self.optimizer = torch.optim.Adam(main_params, lr=ppo_cfg['lr'], eps=1e-5)
        self.comm_optimizers = [
            torch.optim.Adam(self.comm_channels[a].parameters(), lr=ppo_cfg['lr'], eps=1e-5)
            for a in range(2)
        ]

        # Config
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
        self.comm_reward_coef = config['multi_agent']['comm_reward_coef']

        # Intrinsic reward annealing and warmup
        self.intrinsic_anneal = config['worker'].get('intrinsic_anneal', False)
        self.intrinsic_anneal_steps = config['worker'].get('intrinsic_anneal_steps', 500000)
        self.intrinsic_warmup_steps = config['worker'].get('intrinsic_warmup_steps', 0)

        # Listener reward
        self.listener_reward_coef = config.get('communication', {}).get('listener_reward_coef', 0.0)
        ma_listener = config.get('multi_agent', {}).get('listener_reward_coef', 0.0)
        if ma_listener > 0:
            self.listener_reward_coef = ma_listener

        # Communication ablation
        self.comm_ablation = config.get('communication', {}).get('ablation_mode', False)

        self.tau_start = comm_cfg['tau_start']
        self.tau_end = comm_cfg['tau_end']
        self.tau_anneal_steps = comm_cfg['tau_anneal_steps']

        # State
        self.global_step = 0

        # Initialize env observations
        self.obs = [None, None]  # per-agent obs across all envs
        obs_a_list, obs_b_list = [], []
        for env in self.envs:
            (obs_a, obs_b), _ = env.reset()
            obs_a_list.append(obs_a)
            obs_b_list.append(obs_b)
        self.obs[0] = torch.from_numpy(np.stack(obs_a_list)).to(self.device)
        self.obs[1] = torch.from_numpy(np.stack(obs_b_list)).to(self.device)

        # Persistent partner messages (one-step delay)
        msg_dim = comm_cfg['message_length'] * comm_cfg['vocab_size']
        self.partner_msg = [
            torch.zeros(self.num_envs, msg_dim).to(self.device),
            torch.zeros(self.num_envs, msg_dim).to(self.device),
        ]

        # Persistent episode reward accumulators across rollouts
        self._episode_rewards = [np.zeros(self.num_envs), np.zeros(self.num_envs)]

    def _get_intrinsic_coef(self):
        """Get current intrinsic reward coefficient (with warmup then optional anneal)."""
        target = self.intrinsic_coef
        # Warmup: linearly ramp from 0 to target
        if self.intrinsic_warmup_steps > 0 and self.global_step < self.intrinsic_warmup_steps:
            target = target * (self.global_step / self.intrinsic_warmup_steps)
        # Anneal: decay from target to 0 (after warmup)
        if self.intrinsic_anneal:
            anneal_start = self.intrinsic_warmup_steps
            elapsed = max(0, self.global_step - anneal_start)
            frac = min(1.0, elapsed / self.intrinsic_anneal_steps)
            target = target * (1.0 - frac)
        return target

    def _anneal_tau(self):
        frac = min(1.0, self.global_step / self.tau_anneal_steps)
        tau = self.tau_start + frac * (self.tau_end - self.tau_start)
        for cc in self.comm_channels:
            cc.set_tau(tau)

    def _compute_intrinsic_reward(self, agent_idx, obs_features, goal):
        projected = F.normalize(self.goal_projections[agent_idx](obs_features), dim=-1)
        goal_norm = F.normalize(goal, dim=-1)
        return -torch.norm(projected - goal_norm, dim=-1)

    def collect_rollout(self, ablate_comm=False):
        """Collect a rollout from all environments.

        Args:
            ablate_comm: If True, zero out partner messages (for ablation testing).
        """
        T = self.num_steps
        N = self.num_envs

        # Per-agent buffers
        bufs = [{} for _ in range(2)]
        for a in range(2):
            bufs[a] = {
                'obs': torch.zeros(T, N, *self.envs[0].observation_space.shape).to(self.device),
                'goals': torch.zeros(T, N, self.config['manager']['goal_dim']).to(self.device),
                'actions': torch.zeros(T, N, dtype=torch.long).to(self.device),
                'logprobs': torch.zeros(T, N).to(self.device),
                'rewards': torch.zeros(T, N).to(self.device),
                'dones': torch.zeros(T, N).to(self.device),
                'values': torch.zeros(T, N).to(self.device),
            }

        # Manager PPO buffers per-agent per-env
        m_feat_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_goal_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_logp_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_val_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_rew_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_done_buf = [[[] for _ in range(N)] for _ in range(2)]
        m_msg_buf = [[[] for _ in range(N)] for _ in range(2)]  # partner msg embeddings

        messages_log = [[], []]
        states_log = [[], []]

        current_goal = [
            torch.zeros(N, self.config['manager']['goal_dim']).to(self.device),
            torch.zeros(N, self.config['manager']['goal_dim']).to(self.device),
        ]
        manager_ext_reward = [
            torch.zeros(N).to(self.device),
            torch.zeros(N).to(self.device),
        ]
        steps_since_goal = torch.zeros(N, dtype=torch.long).to(self.device)

        episode_returns = []

        for step in range(T):
            self._anneal_tau()
            needs_new_goal = (steps_since_goal % self.goal_period == 0)

            for a in range(2):
                bufs[a]['obs'][step] = self.obs[a]

            # Encode both agents
            with torch.no_grad():
                features = [self.encoders[a](self.obs[a]) for a in range(2)]

            # Manager decisions
            if needs_new_goal.any():
                new_msgs = [None, None]
                for a in range(2):
                    partner = 1 - a
                    # Embed partner's previous message
                    with torch.no_grad():
                        if ablate_comm:
                            partner_embed = torch.zeros(
                                N, self.config['manager']['goal_dim']
                            ).to(self.device)
                        else:
                            partner_embed = self.comm_channels[partner].embed_message(
                                self.partner_msg[partner]
                            )
                        goal, log_prob, value, _ = self.managers[a](
                            features[a], received_message=partner_embed
                        )
                        msg_onehot, msg_indices, _ = self.comm_channels[a].encode(goal)
                        decoded_goal = F.normalize(
                            self.comm_channels[a].decode(msg_onehot), dim=-1
                        )

                    new_goal = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(decoded_goal),
                        decoded_goal, current_goal[a]
                    )

                    # Store new message for partner (one-step delay)
                    new_msgs[a] = torch.where(
                        needs_new_goal.unsqueeze(-1).expand_as(msg_onehot),
                        msg_onehot, self.partner_msg[a]
                    )

                    messages_log[a].append(msg_indices[needs_new_goal].cpu().numpy())
                    states_log[a].append(features[a][needs_new_goal].detach().cpu().numpy())

                    # Manager PPO data
                    # Use shared critic for value
                    with torch.no_grad():
                        shared_val = self.shared_critic(features[0], features[1])

                    for i in range(N):
                        if needs_new_goal[i]:
                            m_feat_buf[a][i].append(features[a][i])
                            m_goal_buf[a][i].append(goal[i])
                            m_logp_buf[a][i].append(log_prob[i])
                            m_val_buf[a][i].append(shared_val[i])
                            m_rew_buf[a][i].append(manager_ext_reward[a][i].item())
                            m_done_buf[a][i].append(0.0)
                            m_msg_buf[a][i].append(partner_embed[i])
                            manager_ext_reward[a][i] = 0.0

                    current_goal[a] = new_goal

                # Update partner messages after both agents have acted
                for a in range(2):
                    self.partner_msg[a] = new_msgs[a].detach()

            for a in range(2):
                bufs[a]['goals'][step] = current_goal[a]

            # Workers act
            actions = []
            for a in range(2):
                with torch.no_grad():
                    action, log_prob, value = self.workers[a](features[a], current_goal[a])
                bufs[a]['actions'][step] = action
                bufs[a]['logprobs'][step] = log_prob
                bufs[a]['values'][step] = value
                actions.append(action.cpu().numpy())

            # Step all environments
            next_obs = [[], []]
            rewards = [np.zeros(N), np.zeros(N)]
            dones = np.zeros(N, dtype=bool)

            for env_idx, env in enumerate(self.envs):
                (obs_a, obs_b), (rew_a, rew_b), done, truncated, info = env.step(
                    (actions[0][env_idx], actions[1][env_idx])
                )
                next_obs[0].append(obs_a)
                next_obs[1].append(obs_b)
                rewards[0][env_idx] = rew_a
                rewards[1][env_idx] = rew_b
                dones[env_idx] = done or truncated

                if done or truncated:
                    (obs_a, obs_b), _ = env.reset()
                    next_obs[0][-1] = obs_a
                    next_obs[1][-1] = obs_b

            next_obs_t = [
                torch.from_numpy(np.stack(next_obs[a])).to(self.device)
                for a in range(2)
            ]
            done_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)

            # Worker rewards
            intrinsic_coef = self._get_intrinsic_coef()
            for a in range(2):
                reward_t = torch.from_numpy(rewards[a].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    next_feat = self.encoders[a](next_obs_t[a])
                    intrinsic = self._compute_intrinsic_reward(a, next_feat, current_goal[a])
                worker_reward = intrinsic_coef * intrinsic + self.extrinsic_coef * reward_t
                bufs[a]['rewards'][step] = worker_reward
                bufs[a]['dones'][step] = done_t
                manager_ext_reward[a] += reward_t

            # Listener reward: reward agent A proportional to how well
            # agent B's state matches A's message (and vice versa)
            if self.listener_reward_coef > 0 and not ablate_comm:
                with torch.no_grad():
                    for a in range(2):
                        partner = 1 - a
                        # How close is the partner's achieved state to our goal direction?
                        partner_feat = self.encoders[partner](next_obs_t[partner])
                        partner_proj = F.normalize(
                            self.goal_projections[partner](partner_feat), dim=-1
                        )
                        our_goal_norm = F.normalize(current_goal[a], dim=-1)
                        listener_sim = (partner_proj * our_goal_norm).sum(dim=-1)
                        bufs[a]['rewards'][step] += self.listener_reward_coef * listener_sim

            # Episode tracking
            for env_idx in range(N):
                self._episode_rewards[0][env_idx] += rewards[0][env_idx]
                self._episode_rewards[1][env_idx] += rewards[1][env_idx]
                if dones[env_idx]:
                    total = self._episode_rewards[0][env_idx] + self._episode_rewards[1][env_idx]
                    episode_returns.append(total / 2)  # average of both agents
                    self._episode_rewards[0][env_idx] = 0
                    self._episode_rewards[1][env_idx] = 0
                    steps_since_goal[env_idx] = 0
                    for a in range(2):
                        if m_done_buf[a][env_idx]:
                            m_done_buf[a][env_idx][-1] = 1.0
                        manager_ext_reward[a][env_idx] = 0

            steps_since_goal += 1
            self.obs = next_obs_t
            self.global_step += self.num_envs

        # Bootstrap worker values and compute GAE
        worker_rollouts = []
        for a in range(2):
            with torch.no_grad():
                feat = self.encoders[a](self.obs[a])
                next_val = self.workers[a].get_value(feat, current_goal[a])
            advantages, returns = compute_gae(
                bufs[a]['rewards'], bufs[a]['values'], bufs[a]['dones'],
                next_val, self.gamma, self.gae_lambda
            )
            worker_rollouts.append({
                'obs': bufs[a]['obs'].reshape(-1, *self.envs[0].observation_space.shape),
                'goals': bufs[a]['goals'].reshape(-1, self.config['manager']['goal_dim']),
                'actions': bufs[a]['actions'].reshape(-1),
                'old_log_probs': bufs[a]['logprobs'].reshape(-1),
                'advantages': advantages.reshape(-1),
                'returns': returns.reshape(-1),
            })

        # Manager GAE per-agent per-env
        manager_rollouts = [{}, {}]
        for a in range(2):
            all_feat, all_goal, all_logp, all_adv, all_ret, all_msg = [], [], [], [], [], []
            for i in range(N):
                n_events = len(m_val_buf[a][i])
                if n_events < 2:
                    continue
                vals = torch.stack(m_val_buf[a][i])
                rews = torch.tensor(m_rew_buf[a][i][1:], device=self.device)
                dns = torch.tensor(m_done_buf[a][i][1:], device=self.device)
                if len(rews) == 0:
                    continue
                adv, ret = compute_gae(rews, vals[:-1], dns, vals[-1],
                                       self.gamma, self.gae_lambda)
                all_feat.append(torch.stack(m_feat_buf[a][i][:-1]))
                all_goal.append(torch.stack(m_goal_buf[a][i][:-1]))
                all_logp.append(torch.stack(m_logp_buf[a][i][:-1]))
                all_msg.append(torch.stack(m_msg_buf[a][i][:-1]))
                all_adv.append(adv)
                all_ret.append(ret)

            if all_adv:
                manager_rollouts[a] = {
                    'features': torch.cat(all_feat),
                    'goals': torch.cat(all_goal),
                    'old_log_probs': torch.cat(all_logp),
                    'received_messages': torch.cat(all_msg),
                    'advantages': torch.cat(all_adv),
                    'returns': torch.cat(all_ret),
                }

        # Compute temporal_extent correctly per env (across both agents).
        # bufs[a]['goals'] has shape (T, N, D); per-env sequences are what we
        # actually want to measure for goal persistence.
        per_env_extents = []
        for a in range(2):
            goal_seq = bufs[a]['goals'].detach().cpu().numpy()  # (T, N, D)
            for i in range(N):
                per_env_extents.append(
                    float(_temporal_extent(goal_seq[:, i, :], threshold=0.01))
                )

        return (worker_rollouts, manager_rollouts, messages_log, states_log,
                episode_returns, per_env_extents)

    def update(self, worker_rollouts, manager_rollouts):
        all_stats = defaultdict(list)

        # Worker PPO update (both agents together)
        for a in range(2):
            w_data = worker_rollouts[a]
            B = w_data['obs'].shape[0]
            batch_size = B // self.num_minibatches

            for _ in range(self.update_epochs):
                indices = torch.randperm(B)
                for start in range(0, B, batch_size):
                    mb_idx = indices[start:start + batch_size]
                    batch = {k: v[mb_idx] for k, v in w_data.items()}
                    agent_idx = a
                    def worker_policy_fn(b, idx=agent_idx):
                        feat = self.encoders[idx](b['obs'])
                        return self.workers[idx].evaluate_actions(feat, b['goals'], b['actions'])
                    stats = ppo_update(
                        batch, worker_policy_fn, self.optimizer,
                        self.clip_eps, self.entropy_coef, self.value_coef, self.max_grad_norm
                    )
                    for k, v in stats.items():
                        all_stats[f'worker_{k}'].append(v)

        # Manager PPO update (both agents)
        for a in range(2):
            m_data = manager_rollouts[a]
            if not m_data or 'advantages' not in m_data:
                continue
            M = m_data['advantages'].shape[0]
            if M <= 4:
                continue
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
                        'received_messages': m_data['received_messages'][mb_idx],
                    }
                    agent_idx = a
                    def manager_policy_fn(b, idx=agent_idx):
                        return self.managers[idx].evaluate_actions(
                            b['obs_features'], b['goals'],
                            received_message=b['received_messages']
                        )
                    stats = ppo_update(
                        batch, manager_policy_fn, self.optimizer,
                        0.1, self.entropy_coef * 0.1, self.value_coef, self.max_grad_norm
                    )
                    for k, v in stats.items():
                        all_stats[f'manager_{k}'].append(v)

        # Communication channel reconstruction loss (both agents).
        # Normalize goals to the unit sphere before passing through the
        # bottleneck: otherwise recon_loss + sender_entropy has a trivial
        # minimum at raw_goals -> 0 (silent magnitude collapse).
        for a in range(2):
            m_data = manager_rollouts[a]
            if not m_data or 'goals' not in m_data:
                continue
            raw_goals = F.normalize(m_data['goals'].detach(), dim=-1)
            msg_onehot, _, logits = self.comm_channels[a].encode(raw_goals)
            reconstructed = F.normalize(
                self.comm_channels[a].decode(msg_onehot), dim=-1
            )
            recon_loss = F.mse_loss(reconstructed, raw_goals)

            probs = F.softmax(logits, dim=-1)
            sender_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            comm_loss = recon_loss - 0.05 * sender_entropy

            self.comm_optimizers[a].zero_grad()
            comm_loss.backward()
            nn.utils.clip_grad_norm_(self.comm_channels[a].parameters(), 1.0)
            self.comm_optimizers[a].step()

            all_stats['comm_recon_loss'].append(recon_loss.item())
            all_stats['comm_sender_entropy'].append(sender_entropy.item())

        return {k: np.mean(v) for k, v in all_stats.items()}

    def evaluate(self, num_episodes=20, ablate_comm=False):
        """Evaluate agents with optional communication ablation.

        Args:
            num_episodes: Number of episodes to evaluate.
            ablate_comm: If True, zero out partner messages.
        Returns:
            Dict with mean returns for each agent and combined.
        """
        returns_a, returns_b = [], []
        successes = []

        for ep in range(num_episodes):
            env = MultiAgentWrapper(
                size=self.corridor_size,
                corridor_length=3,
                max_steps=self.max_steps,
                seed=self.config['experiment']['seed'] + 100_000 + ep,
                corridor_width=self.corridor_width,
                asymmetric_info=self.config['env'].get('asymmetric_info', False),
                rendezvous_bonus=self.rendezvous_bonus,
                num_obstacles=self.num_obstacles,
                bus_cost_solo=self.bus_cost_solo,
                bus_cost_shared=self.bus_cost_shared,
                bus_window=self.bus_window,
                turn_taking=self.turn_taking,
            )
            (obs_a, obs_b), _ = env.reset()
            obs = [
                torch.from_numpy(obs_a).unsqueeze(0).to(self.device),
                torch.from_numpy(obs_b).unsqueeze(0).to(self.device),
            ]
            msg_dim = self.config['communication']['message_length'] * self.config['communication']['vocab_size']
            partner_msg = [
                torch.zeros(1, msg_dim).to(self.device),
                torch.zeros(1, msg_dim).to(self.device),
            ]
            goal = [
                torch.zeros(1, self.config['manager']['goal_dim']).to(self.device),
                torch.zeros(1, self.config['manager']['goal_dim']).to(self.device),
            ]
            ep_reward = [0.0, 0.0]
            step = 0
            done = False

            while not done and step < self.max_steps:
                with torch.no_grad():
                    feats = [self.encoders[a](obs[a]) for a in range(2)]

                    if step % self.goal_period == 0:
                        for a in range(2):
                            partner = 1 - a
                            if ablate_comm:
                                p_embed = torch.zeros(
                                    1, self.config['manager']['goal_dim']
                                ).to(self.device)
                            else:
                                p_embed = self.comm_channels[partner].embed_message(
                                    partner_msg[partner]
                                )
                            g, _, _, _ = self.managers[a](feats[a], received_message=p_embed)
                            msg_oh, _, _ = self.comm_channels[a].encode(g)
                            goal[a] = F.normalize(
                                self.comm_channels[a].decode(msg_oh), dim=-1
                            )
                            partner_msg[a] = msg_oh.detach()

                    acts = []
                    for a in range(2):
                        action, _, _ = self.workers[a](feats[a], goal[a])
                        acts.append(action.item())

                (obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(
                    tuple(acts)
                )
                obs = [
                    torch.from_numpy(obs_a).unsqueeze(0).to(self.device),
                    torch.from_numpy(obs_b).unsqueeze(0).to(self.device),
                ]
                ep_reward[0] += rew_a
                ep_reward[1] += rew_b
                done = terminated or truncated
                step += 1

            returns_a.append(ep_reward[0])
            returns_b.append(ep_reward[1])
            successes.append(float(((ep_reward[0] + ep_reward[1]) / 2.0) > 0.0))

        return {
            'mean_return_a': np.mean(returns_a),
            'mean_return_b': np.mean(returns_b),
            'mean_return': np.mean([(a + b) / 2 for a, b in zip(returns_a, returns_b)]),
            'std_return': np.std([(a + b) / 2 for a, b in zip(returns_a, returns_b)]),
            'success_rate': np.mean(successes) if successes else 0.0,
        }

    def train(self, output_dir='outputs', wandb_run=None):
        os.makedirs(output_dir, exist_ok=True)

        num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        start_time = time.time()

        all_returns = []
        all_messages = []
        all_states = []
        all_decoded_goals = []
        all_temporal_extents = []
        recent_recon_loss = []

        print(f"Training mode: social (2 agents)")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Num updates: {num_updates}")
        print(f"Device: {self.device}")
        if self.intrinsic_anneal:
            print(f"Intrinsic reward annealing: {self.intrinsic_coef} -> 0 over {self.intrinsic_anneal_steps} steps")
        if self.listener_reward_coef > 0:
            print(f"Listener reward coefficient: {self.listener_reward_coef}")
        if self.comm_ablation:
            print(f"Communication ablation: will evaluate with/without comm at end")
        print()

        ppo_cfg = self.config['ppo']
        lr = ppo_cfg['lr']

        for update in range(1, num_updates + 1):
            if ppo_cfg['anneal_lr']:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * ppo_cfg['lr']
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            (worker_rollouts, manager_rollouts, messages_log, states_log_batch,
             ep_returns, per_env_extents) = self.collect_rollout()
            stats = self.update(worker_rollouts, manager_rollouts)
            if per_env_extents:
                all_temporal_extents.extend(per_env_extents)

            for a in range(2):
                if messages_log[a]:
                    all_messages.extend(messages_log[a])
                if states_log_batch[a]:
                    all_states.extend(states_log_batch[a])
                m_data = manager_rollouts[a] if manager_rollouts else None
                if isinstance(m_data, dict) and 'goals' in m_data:
                    # Agent-side goals are pre-bottleneck. Decode to see what
                    # the worker actually consumes (unit-normalized, as used
                    # everywhere else in the bottleneck path).
                    with torch.no_grad():
                        raw = F.normalize(m_data['goals'].detach(), dim=-1)
                        msg_oh, _, _ = self.comm_channels[a].encode(raw)
                        decoded = F.normalize(
                            self.comm_channels[a].decode(msg_oh), dim=-1
                        ).cpu().numpy()
                    if decoded.ndim == 2 and len(decoded) > 0:
                        all_decoded_goals.append(decoded)
            if 'comm_recon_loss' in stats:
                recent_recon_loss.append(stats['comm_recon_loss'])
            all_returns.extend(ep_returns)

            if update % self.config['experiment']['log_interval'] == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed if elapsed > 0 else 0
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

            if self.global_step > 0 and self.global_step % self.config['experiment']['save_interval'] == 0:
                self.save(os.path.join(output_dir, f'checkpoint_{self.global_step}.pt'))

        self.save(os.path.join(output_dir, 'final.pt'))
        final_eval = self.evaluate(num_episodes=self.eval_episodes, ablate_comm=False)
        print(
            f"Final eval: return={final_eval['mean_return']:.3f} +/- "
            f"{final_eval['std_return']:.3f} | success={100 * final_eval['success_rate']:.1f}%"
        )

        # Communication ablation evaluation
        comm_eval = None
        if self.comm_ablation:
            print("\n--- Communication Ablation Evaluation ---")
            eval_with_comm = final_eval
            eval_without_comm = self.evaluate(
                num_episodes=self.eval_episodes, ablate_comm=True
            )
            print(f"  With comm:    return={eval_with_comm['mean_return']:.3f} +/- {eval_with_comm['std_return']:.3f}")
            print(f"  Without comm: return={eval_without_comm['mean_return']:.3f} +/- {eval_without_comm['std_return']:.3f}")
            print(f"  Delta: {eval_with_comm['mean_return'] - eval_without_comm['mean_return']:.3f}")
            comm_eval = {
                'with_comm': eval_with_comm,
                'without_comm': eval_without_comm,
                'delta': eval_with_comm['mean_return'] - eval_without_comm['mean_return'],
            }

            if wandb_run is not None:
                wandb_run.summary['eval_with_comm'] = eval_with_comm['mean_return']
                wandb_run.summary['eval_without_comm'] = eval_without_comm['mean_return']
                wandb_run.summary['comm_ablation_delta'] = (
                    eval_with_comm['mean_return'] - eval_without_comm['mean_return']
                )

        decoded_goals_arr = (
            np.concatenate(all_decoded_goals, axis=0)
            if all_decoded_goals else None
        )
        return {
            'returns': all_returns,
            'messages': all_messages,
            'states': all_states,
            'decoded_goals': decoded_goals_arr,
            'recon_loss_mean': float(np.mean(recent_recon_loss[-50:])) if recent_recon_loss else None,
            'temporal_extent_mean': float(np.mean(all_temporal_extents)) if all_temporal_extents else None,
            'eval': final_eval,
            'comm_ablation_eval': comm_eval,
        }

    def save(self, path):
        state = {
            'mode': 'social',
            'global_step': self.global_step,
            'config': self.config,
            'encoders': self.encoders.state_dict(),
            'managers': self.managers.state_dict(),
            'workers': self.workers.state_dict(),
            'comm_channels': self.comm_channels.state_dict(),
            'goal_projections': self.goal_projections.state_dict(),
            'shared_critic': self.shared_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'comm_optimizers': [opt.state_dict() for opt in self.comm_optimizers],
        }
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.encoders.load_state_dict(state['encoders'])
        self.managers.load_state_dict(state['managers'])
        self.workers.load_state_dict(state['workers'])
        self.comm_channels.load_state_dict(state['comm_channels'])
        self.goal_projections.load_state_dict(state['goal_projections'])
        self.shared_critic.load_state_dict(state['shared_critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        if 'comm_optimizers' in state:
            for a in range(2):
                self.comm_optimizers[a].load_state_dict(state['comm_optimizers'][a])
        self.global_step = state['global_step']
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
