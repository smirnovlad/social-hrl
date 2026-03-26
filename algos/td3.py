"""TD3 (Twin Delayed DDPG) for the manager policy.

Used in continuous mode where the manager outputs deterministic goals
and is trained off-policy with a replay buffer. Follows the approach
from HIRO (Nachum et al., 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as nnf
import copy


class ReplayBuffer:
    """Simple replay buffer for (state, goal, reward, next_state, done) tuples."""

    def __init__(self, state_dim, goal_dim, max_size=200000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.goals = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, goal, reward, next_state, done):
        """Add a single transition."""
        self.states[self.ptr] = state
        self.goals[self.ptr] = goal
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, goals, rewards, next_states, dones):
        """Add a batch of transitions."""
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.add(states[i], goals[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size, device='cpu'):
        """Sample a random batch."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.FloatTensor(self.states[idx]).to(device),
            'goals': torch.FloatTensor(self.goals[idx]).to(device),
            'rewards': torch.FloatTensor(self.rewards[idx]).to(device),
            'next_states': torch.FloatTensor(self.next_states[idx]).to(device),
            'dones': torch.FloatTensor(self.dones[idx]).to(device),
        }


class TD3Critic(nn.Module):
    """Twin Q-networks for TD3."""

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

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.q1(x)


class TD3Actor(nn.Module):
    """Deterministic policy that outputs goal vectors."""

    def __init__(self, state_dim, goal_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )

    def forward(self, state):
        return torch.tanh(self.net(state))


class ManagerTD3:
    """TD3 trainer for the manager policy.

    The manager observes encoded features and outputs goal vectors.
    Trained off-policy with replay buffer, twin critics, delayed updates.
    """

    def __init__(self, state_dim, goal_dim, hidden_dim=128, device='cpu',
                 lr=0.0003, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2, exploration_noise=0.1,
                 buffer_size=200000, batch_size=256, warmup_steps=1000):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # Networks
        self.actor = TD3Actor(state_dim, goal_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TD3Critic(state_dim, goal_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(state_dim, goal_dim, buffer_size)
        self.total_updates = 0
        self.goal_dim = goal_dim

    def select_goal(self, state, add_noise=True):
        """Select a goal given encoded state features.

        Args:
            state: (batch, state_dim) tensor.
            add_noise: Whether to add exploration noise.
        Returns:
            goal: (batch, goal_dim) tensor.
        """
        with torch.no_grad():
            goal = self.actor(state)
            if add_noise:
                noise = torch.randn_like(goal) * self.exploration_noise
                goal = goal + noise
        return goal

    def add_transition(self, state, goal, reward, next_state, done):
        """Add manager transition to replay buffer.

        Args:
            state: (N, state_dim) features when goal was set.
            goal: (N, goal_dim) goal that was set.
            reward: (N,) cumulative extrinsic reward over goal period.
            next_state: (N, state_dim) features when next goal is set.
            done: (N,) whether episode ended during goal period.
        """
        state_np = state.cpu().numpy()
        goal_np = goal.cpu().numpy()
        reward_np = reward.cpu().numpy().reshape(-1, 1)
        next_state_np = next_state.cpu().numpy()
        done_np = done.cpu().numpy().reshape(-1, 1)
        self.replay_buffer.add_batch(
            state_np, goal_np, reward_np, next_state_np, done_np
        )

    def update(self):
        """Run one TD3 update step. Returns stats dict or None if not enough data."""
        if self.replay_buffer.size < self.warmup_steps:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        states = batch['states']
        goals = batch['goals']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(goals) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_goals = self.actor_target(next_states) + noise

            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_goals)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Critic update
        current_q1, current_q2 = self.critic(states, goals)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        stats = {
            'manager_critic_loss': critic_loss.item(),
            'manager_q_mean': current_q1.mean().item(),
        }

        # Delayed actor update
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft target update
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            stats['manager_actor_loss'] = actor_loss.item()

        return stats

    def _soft_update(self, source, target):
        """Polyak averaging for target network."""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_updates': self.total_updates,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.total_updates = state['total_updates']
