"""PPO utilities: GAE computation and policy update step.

Used by both single-agent and multi-agent training loops.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T,) or (T, N) tensor of rewards.
        values: (T,) or (T, N) tensor of value estimates.
        dones: (T,) or (T, N) tensor of done flags.
        next_value: () or (N,) value estimate for the state after last step.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
    Returns:
        advantages: Same shape as rewards.
        returns: Same shape as rewards (advantages + values).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(batch, policy_fn, optimizer, clip_eps=0.2, entropy_coef=0.01,
               value_coef=0.5, max_grad_norm=0.5):
    """Single PPO update step.

    Args:
        batch: Dict with keys:
            'obs_features': (B, D) encoded observations
            'actions': (B,) actions taken
            'old_log_probs': (B,) log probs under old policy
            'advantages': (B,) GAE advantages
            'returns': (B,) discounted returns
            'goals': (B, goal_dim) goals (for worker) or None
            'messages': (B, msg_dim) received messages (for manager) or None
        policy_fn: Callable(batch) -> (new_log_probs, entropy, values)
            A function that evaluates the current policy on the batch.
        optimizer: torch.optim.Optimizer
        clip_eps: PPO clip epsilon.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Max gradient norm for clipping.
    Returns:
        Dict with loss components for logging.
    """
    new_log_probs, entropy, values = policy_fn(batch)

    advantages = batch['advantages']
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss (clipped surrogate)
    log_ratio = new_log_probs - batch['old_log_probs']
    log_ratio = torch.clamp(log_ratio, -20, 20)  # prevent inf in exp()
    ratio = log_ratio.exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (clamp predictions to prevent explosion)
    value_loss = 0.5 * (values - batch['returns']).pow(2).mean()

    # Entropy bonus
    entropy_loss = -entropy.mean()

    # Total loss
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Skip update if loss is NaN
    if torch.isnan(loss) or torch.isinf(loss):
        return {
            'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
            'entropy': 0.0, 'approx_kl': 0.0,
        }

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group['params']],
        max_grad_norm
    )
    optimizer.step()

    approx_kl = ((ratio - 1) - log_ratio).mean().item()
    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': -entropy_loss.item(),
        'approx_kl': approx_kl,
    }
