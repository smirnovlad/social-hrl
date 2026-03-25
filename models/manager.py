"""Manager (high-level) policy for HRL.

The manager observes the encoded state and outputs a latent goal vector
every c steps. Optionally conditions on a received message from a partner agent.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Manager(nn.Module):
    """High-level policy that outputs latent goal vectors."""

    def __init__(self, input_dim, goal_dim=16, hidden_dim=128, message_dim=0):
        """
        Args:
            input_dim: Dimension of encoded observation.
            goal_dim: Dimension of goal vector g.
            hidden_dim: Hidden layer size.
            message_dim: Dimension of received message embedding (0 if no communication).
        """
        super().__init__()

        total_input = input_dim + message_dim

        self.policy = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.goal_mean = nn.Linear(hidden_dim, goal_dim)
        self.goal_logstd = nn.Parameter(torch.zeros(goal_dim))

        self.value_head = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.goal_dim = goal_dim

    def forward(self, obs_features, received_message=None):
        """
        Args:
            obs_features: (batch, input_dim) encoded observation.
            received_message: (batch, message_dim) or None.
        Returns:
            goal: (batch, goal_dim) sampled goal vector.
            log_prob: (batch,) log probability of the goal.
            value: (batch,) estimated state value.
            goal_mean: (batch, goal_dim) mean of goal distribution.
        """
        if received_message is not None:
            x = torch.cat([obs_features, received_message], dim=-1)
        else:
            x = obs_features

        h = self.policy(x)
        mean = self.goal_mean(h)
        std = self.goal_logstd.exp().expand_as(mean)

        dist = Normal(mean, std)
        goal = dist.rsample()
        log_prob = dist.log_prob(goal).sum(dim=-1)

        value = self.value_head(x).squeeze(-1)

        return goal, log_prob, value, mean

    def evaluate_actions(self, obs_features, goals, received_message=None):
        """Evaluate log_prob and entropy for given goals (used in PPO update)."""
        if received_message is not None:
            x = torch.cat([obs_features, received_message], dim=-1)
        else:
            x = obs_features

        h = self.policy(x)
        mean = self.goal_mean(h)
        std = self.goal_logstd.exp().expand_as(mean)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(goals).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(x).squeeze(-1)

        return log_prob, entropy, value
