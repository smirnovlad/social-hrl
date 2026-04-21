"""SAC (Soft Actor-Critic) for the manager policy.

Alternative to TD3 with built-in entropy maximization, which directly
fights goal collapse without needing a discrete bottleneck.
Follows Haarnoja et al. (ICML 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from algos.td3 import ReplayBuffer


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActorContinuous(nn.Module):
    """Stochastic actor with squashed Gaussian (tanh-Normal)."""

    def __init__(self, state_dim, goal_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, goal_dim)
        self.log_std_head = nn.Linear(hidden_dim, goal_dim)

    def forward(self, state):
        h = self.shared(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Sample action with log_prob corrected for tanh squashing.

        Returns:
            action: (batch, goal_dim) in [-1, 1].
            log_prob: (batch,) corrected log probability.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # pre-squash
        action = torch.tanh(u)

        # Log probability with tanh correction
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)

        return action, log_prob

    def deterministic(self, state):
        """Deterministic output (for evaluation)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class SACCritic(nn.Module):
    """Twin Q-networks for SAC (same architecture as TD3)."""

    def __init__(self, state_dim, goal_dim, hidden_dim=128):
        super().__init__()
        input_dim = state_dim + goal_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ManagerSAC:
    """SAC trainer for the manager policy.

    Built-in entropy maximization provides automatic anti-collapse:
    the actor is rewarded for producing diverse goals.
    """

    def __init__(self, state_dim, goal_dim, hidden_dim=128, device='cpu',
                 lr=0.0003, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_entropy=None,
                 buffer_size=200000, batch_size=256, warmup_steps=1000):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.goal_dim = goal_dim

        # Networks
        self.actor = SACActorContinuous(state_dim, goal_dim, hidden_dim).to(device)
        self.critic = SACCritic(state_dim, goal_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy is not None else -goal_dim
            self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), device=device)

        self.replay_buffer = ReplayBuffer(state_dim, goal_dim, buffer_size)
        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_goal(self, state, add_noise=True):
        """Select a goal given encoded state features.

        Args:
            state: (batch, state_dim) tensor.
            add_noise: If False, use deterministic output.
        Returns:
            goal: (batch, goal_dim) tensor.
        """
        with torch.no_grad():
            if add_noise:
                goal, _ = self.actor.sample(state)
            else:
                goal = self.actor.deterministic(state)
        return goal

    def add_transition(self, state, goal, reward, next_state, done):
        """Add manager transition to replay buffer."""
        state_np = state.cpu().numpy()
        goal_np = goal.cpu().numpy()
        reward_np = reward.cpu().numpy().reshape(-1, 1)
        next_state_np = next_state.cpu().numpy()
        done_np = done.cpu().numpy().reshape(-1, 1)
        self.replay_buffer.add_batch(
            state_np, goal_np, reward_np, next_state_np, done_np
        )

    def update(self):
        """Run one SAC update step."""
        if self.replay_buffer.size < self.warmup_steps:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        states = batch['states']
        goals = batch['goals']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # --- Critic update ---
        with torch.no_grad():
            next_goals, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_goals)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_prob.unsqueeze(-1))

        current_q1, current_q2 = self.critic(states, goals)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        new_goals, log_prob = self.actor.sample(states)
        q1, q2 = self.critic(states, new_goals)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - min_q.squeeze(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        stats = {
            'manager_critic_loss': critic_loss.item(),
            'manager_actor_loss': actor_loss.item(),
            'manager_q_mean': current_q1.mean().item(),
            'manager_entropy': -log_prob.mean().item(),
            'manager_alpha': self.alpha.item(),
        }

        # --- Alpha (temperature) update ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            stats['manager_alpha_loss'] = alpha_loss.item()

        # Soft target update
        self._soft_update(self.critic, self.critic_target)
        self.total_updates += 1

        return stats

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def state_dict(self):
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'total_updates': self.total_updates,
        }
        if self.auto_alpha:
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        if 'log_alpha' in state:
            self.log_alpha = state['log_alpha'].to(self.log_alpha.device)
            if self.auto_alpha:
                self.log_alpha.requires_grad_(True)
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                        lr=self.alpha_optimizer.defaults['lr'])
                if 'alpha_optimizer' in state:
                    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.total_updates = state['total_updates']
