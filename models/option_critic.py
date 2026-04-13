"""Option-Critic architecture for HRL with learned termination.

Replaces the fixed goal_period manager with options that learn
when to terminate, following Bacon et al. (AAAI 2017).
Each option maps to a goal vector via a learned embedding.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli


class OptionCriticManager(nn.Module):
    """Option-Critic policy with learned termination conditions.

    Instead of a fixed goal period, each option has a termination function
    that decides when to switch to a new option based on the current state.
    """

    def __init__(self, input_dim, goal_dim=16, num_options=8, hidden_dim=128,
                 message_dim=0, termination_reg=0.01):
        """
        Args:
            input_dim: Dimension of encoded observation.
            goal_dim: Dimension of goal vector.
            num_options: Number of options (discrete choices).
            hidden_dim: Hidden layer size.
            message_dim: Dimension of received message (0 if no communication).
            termination_reg: Regularization to prevent premature termination.
        """
        super().__init__()

        self.num_options = num_options
        self.goal_dim = goal_dim
        self.termination_reg = termination_reg
        total_input = input_dim + message_dim

        # Policy over options: selects which option to execute
        self.policy_over_options = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

        # Each option maps to a goal vector
        self.option_goals = nn.Embedding(num_options, goal_dim)

        # Termination function: per-option probability of terminating
        self.termination_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

        # Value function for policy-over-options
        self.value_head = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q-function per option for advantage computation
        self.q_head = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(self, features, received_message=None):
        """Select a new option.

        Args:
            features: (batch, input_dim) encoded observation.
            received_message: (batch, message_dim) or None.
        Returns:
            option: (batch,) selected option index.
            log_prob: (batch,) log probability.
            value: (batch,) state value.
            goal: (batch, goal_dim) goal vector for the selected option.
        """
        if received_message is not None:
            x = torch.cat([features, received_message], dim=-1)
        else:
            x = features

        logits = self.policy_over_options(x)
        dist = Categorical(logits=logits)
        option = dist.sample()
        log_prob = dist.log_prob(option)
        value = self.value_head(x).squeeze(-1)
        goal = self.option_goals(option)

        return option, log_prob, value, goal

    def should_terminate(self, features, current_option):
        """Decide whether the current option should terminate.

        Args:
            features: (batch, input_dim) raw encoder features (no message).
            current_option: (batch,) current option indices.
        Returns:
            terminate: (batch,) bool tensor.
            beta: (batch,) termination probability (for loss computation).
        """
        term_logits = self.termination_head(features)  # (batch, num_options)
        # Gather the termination logit for the current option
        beta = torch.sigmoid(
            term_logits.gather(1, current_option.unsqueeze(-1)).squeeze(-1)
        )
        terminate = torch.bernoulli(beta).bool()
        return terminate, beta

    def get_option_advantage(self, features, current_option, received_message=None):
        """Compute advantage of current option for termination gradient.

        A(s, omega) = Q(s, omega) - V(s)

        Args:
            features: (batch, input_dim).
            current_option: (batch,).
            received_message: (batch, message_dim) or None.
        Returns:
            advantage: (batch,) Q(s, omega) - V(s).
        """
        if received_message is not None:
            x = torch.cat([features, received_message], dim=-1)
        else:
            x = features

        q_values = self.q_head(x)  # (batch, num_options)
        q_omega = q_values.gather(1, current_option.unsqueeze(-1)).squeeze(-1)
        v = self.value_head(x).squeeze(-1)
        return q_omega - v

    def evaluate_actions(self, features, options, received_message=None):
        """Evaluate log_prob and entropy for given options (PPO update).

        Args:
            features: (batch, input_dim).
            options: (batch,) option indices taken.
            received_message: (batch, message_dim) or None.
        Returns:
            log_prob: (batch,).
            entropy: (batch,).
            value: (batch,).
        """
        if received_message is not None:
            x = torch.cat([features, received_message], dim=-1)
        else:
            x = features

        logits = self.policy_over_options(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(options)
        entropy = dist.entropy()
        value = self.value_head(x).squeeze(-1)

        return log_prob, entropy, value

    def termination_loss(self, features, current_option, advantages, beta):
        """Compute termination gradient loss.

        Following Bacon et al. 2017:
        L_term = beta * (advantage + reg)

        Encourages termination when switching to a better option (positive advantage)
        and penalizes premature termination via termination_reg.

        Args:
            features: (batch, input_dim).
            current_option: (batch,).
            advantages: (batch,) Q(s, omega) - V(s).
            beta: (batch,) termination probabilities.
        Returns:
            loss: Scalar.
        """
        # Advantage-based termination: terminate when current option is worse than average
        # Plus regularization to prevent always-terminate
        loss = (beta * (advantages.detach() + self.termination_reg)).mean()
        return loss
