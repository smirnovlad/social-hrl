"""Discrete communication channel with Gumbel-Softmax.

Encodes continuous goal vectors into discrete messages and decodes them back.
Used both as an information bottleneck and for multi-agent communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=True):
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: (batch, num_categories) unnormalized log probabilities.
        tau: Temperature. Lower = more discrete.
        hard: If True, returns one-hot in forward but soft gradients in backward.
    Returns:
        y: (batch, num_categories) sample.
    """
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # Straight-through: forward uses hard, backward uses soft
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


class CommunicationChannel(nn.Module):
    """Encodes goals into discrete messages and decodes them back.

    Architecture:
        Sender: goal (R^d) -> MLP -> L sets of K logits -> Gumbel-Softmax -> L one-hot vectors
        Decoder: L one-hot vectors -> MLP -> reconstructed goal (R^d)

    The discrete message m = (m_1, ..., m_L) where m_i in {1,...,K} serves as:
    1. An information bottleneck forcing goal compression
    2. A communicable message for multi-agent coordination
    """

    def __init__(self, goal_dim, vocab_size=10, message_length=3,
                 hidden_dim=64, tau=1.0):
        """
        Args:
            goal_dim: Dimension of continuous goal vector.
            vocab_size: K - number of symbols per token.
            message_length: L - number of tokens in a message.
            hidden_dim: Hidden layer size for sender/decoder.
            tau: Initial Gumbel-Softmax temperature.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.message_length = message_length
        self.tau = tau

        # Sender: continuous goal -> discrete message
        self.sender = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_length * vocab_size),
        )

        # Decoder: discrete message -> reconstructed goal
        self.decoder = nn.Sequential(
            nn.Linear(message_length * vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )

        # Message embedding for partner agent (different from decoder)
        # Partner doesn't reconstruct the goal - it gets a learned embedding
        self.message_embedder = nn.Sequential(
            nn.Linear(message_length * vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),  # Same dim as goal for simplicity
        )

        self.message_dim = goal_dim  # Output dim for partner's manager input

    def encode(self, goal, hard=True):
        """Encode a continuous goal into a discrete message.

        Args:
            goal: (batch, goal_dim) continuous goal vector.
            hard: Whether to use hard Gumbel-Softmax.
        Returns:
            message_onehot: (batch, L * K) concatenated one-hot vectors.
            message_indices: (batch, L) discrete token indices.
            logits: (batch, L, K) raw logits (for entropy computation).
        """
        raw = self.sender(goal)  # (batch, L * K)
        logits = raw.view(-1, self.message_length, self.vocab_size)  # (batch, L, K)

        # Apply Gumbel-Softmax to each token position independently
        samples = []
        for i in range(self.message_length):
            sample = gumbel_softmax(logits[:, i], tau=self.tau, hard=hard)
            samples.append(sample)

        message_onehot = torch.cat(samples, dim=-1)  # (batch, L * K)
        message_indices = logits.argmax(dim=-1)  # (batch, L)

        return message_onehot, message_indices, logits

    def decode(self, message_onehot):
        """Decode a discrete message back into a continuous goal.

        Args:
            message_onehot: (batch, L * K) concatenated one-hot vectors.
        Returns:
            goal_reconstructed: (batch, goal_dim) reconstructed goal.
        """
        return self.decoder(message_onehot)

    def embed_message(self, message_onehot):
        """Embed a received message for the partner agent's manager.

        This is separate from decode() because the partner doesn't need
        to reconstruct the original goal - it needs a useful representation
        for its own decision-making.

        Args:
            message_onehot: (batch, L * K) from partner agent.
        Returns:
            embedding: (batch, message_dim) for partner's manager input.
        """
        return self.message_embedder(message_onehot)

    def set_tau(self, tau):
        """Update Gumbel-Softmax temperature."""
        self.tau = tau
