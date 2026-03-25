"""Worker (low-level) goal-conditioned policy for HRL.

The worker receives the encoded observation and a goal vector,
and outputs a distribution over primitive actions.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Worker(nn.Module):
    """Low-level goal-conditioned policy over discrete actions."""

    def __init__(self, input_dim, goal_dim, num_actions, hidden_dim=128):
        """
        Args:
            input_dim: Dimension of encoded observation.
            goal_dim: Dimension of goal vector.
            num_actions: Number of discrete actions (7 for Minigrid).
            hidden_dim: Hidden layer size.
        """
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(input_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_features, goal):
        """
        Args:
            obs_features: (batch, input_dim) encoded observation.
            goal: (batch, goal_dim) goal vector from manager.
        Returns:
            action: (batch,) sampled action.
            log_prob: (batch,) log probability.
            value: (batch,) estimated value.
        """
        x = torch.cat([obs_features, goal], dim=-1)

        logits = self.policy(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_head(x).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(self, obs_features, goal, actions):
        """Evaluate log_prob and entropy for given actions (PPO update)."""
        x = torch.cat([obs_features, goal], dim=-1)

        logits = self.policy(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_head(x).squeeze(-1)

        return log_prob, entropy, value

    @torch.no_grad()
    def get_value(self, obs_features, goal):
        """Get value estimate only (for GAE computation)."""
        x = torch.cat([obs_features, goal], dim=-1)
        return self.value_head(x).squeeze(-1)
