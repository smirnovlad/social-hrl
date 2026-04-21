"""LOLA (Learning with Opponent-Learning Awareness) variant of MAPPO+comm.

Foerster et al. (2018) note that independent learners in a social setting can
fall into suboptimal equilibria because each agent treats the partner as a
fixed part of the environment. LOLA corrects for this by having agent i take
a gradient step that accounts for agent j's upcoming gradient step:

    theta_i <- theta_i - alpha * grad_theta_i L_i(theta_i, theta_j + Delta_j)
    where Delta_j = -alpha * grad_theta_j L_j(theta_i, theta_j)

A first-order Taylor expansion gives the LOLA correction term:
    grad_theta_i L_i(theta_i, theta_j + Delta_j)
        approx grad_theta_i L_i - alpha * grad_theta_i [grad_theta_j L_i dot grad_theta_j L_j]

In general HRL-with-PPO, the PPO surrogate for agent i does not depend on
agent j's parameters, so LOLA correction vanishes. To get a non-trivial
signal we apply LOLA to the **communication channels**: the listener reward
provides an explicit differentiable coupling between agent i's goal and agent
j's comm channel (agent j must encode/decode signals compatible with i's).
So LOLA here answers: "if partner updates their comm channel, how will that
affect my reconstruction loss?"

This is a narrow, principled application of LOLA: the two comm-channels are
a 2-player game on the listener-alignment payoff, and LOLA gives each channel
awareness of the other's imminent update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from algos.multi_agent_trainer import MultiAgentTrainer


class LolaMultiAgentTrainer(MultiAgentTrainer):
    """MAPPO + LOLA correction on the communication-channel update."""

    def __init__(self, config, device='cuda'):
        super().__init__(config, device=device)
        lola_cfg = config.get('lola', {})
        self.lola_coef = lola_cfg.get('coef', 1.0)
        self.lola_inner_lr = lola_cfg.get('inner_lr', 1e-3)
        self.lola_warmup = lola_cfg.get('warmup_updates', 0)
        self._lola_update_count = 0

    def _channel_pair_losses(self, goals_a, goals_b):
        """Compute reconstruction-style losses for each channel on the *partner's* goals.

        Agent a's comm channel is scored on how well it decodes messages that
        *agent b* would want to send. This is the natural coupling point: if
        a's decoder changes, b's sender effectiveness changes in turn.

        Returns:
            L_a: mean squared reconstruction loss for agent a on partner goals
            L_b: mean squared reconstruction loss for agent b on partner goals
        """
        # Agent a sends, agent b decodes (b's listener loss on a's goals)
        msg_a, _, _ = self.comm_channels[0].encode(goals_a)
        recon_b_from_a = F.normalize(self.comm_channels[1].decode(msg_a), dim=-1)
        L_b = F.mse_loss(recon_b_from_a, goals_a)

        # Agent b sends, agent a decodes (a's listener loss on b's goals)
        msg_b, _, _ = self.comm_channels[1].encode(goals_b)
        recon_a_from_b = F.normalize(self.comm_channels[0].decode(msg_b), dim=-1)
        L_a = F.mse_loss(recon_a_from_b, goals_b)

        return L_a, L_b

    def update(self, worker_rollouts, manager_rollouts):
        """Run standard PPO updates, then replace comm update with LOLA step."""
        all_stats = defaultdict(list)

        # --- standard worker + manager PPO updates (copied from parent) ---
        # We can't call super().update() directly because it also runs the
        # comm update; we need to intercept before that.
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
                        return self.workers[idx].evaluate_actions(
                            feat, b['goals'], b['actions']
                        )
                    from algos.ppo import ppo_update
                    stats = ppo_update(
                        batch, worker_policy_fn, self.optimizer,
                        self.clip_eps, self.entropy_coef, self.value_coef,
                        self.max_grad_norm,
                    )
                    for k, v in stats.items():
                        all_stats[f'worker_{k}'].append(v)

        for a in range(2):
            m_data = manager_rollouts[a]
            if not m_data or 'advantages' not in m_data:
                continue
            M = m_data['advantages'].shape[0]
            if M <= 4:
                continue
            m_batch_size = max(1, M // 2)
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
                from algos.ppo import ppo_update
                stats = ppo_update(
                    batch, manager_policy_fn, self.optimizer,
                    0.1, self.entropy_coef * 0.1, self.value_coef,
                    self.max_grad_norm,
                )
                for k, v in stats.items():
                    all_stats[f'manager_{k}'].append(v)

        # --- LOLA-corrected comm update ---
        goals_a, goals_b = self._extract_goals(manager_rollouts)
        if goals_a is None or goals_b is None or len(goals_a) < 4 or len(goals_b) < 4:
            # Fall back to plain recon update if we don't have a minibatch worth.
            self._plain_comm_update(manager_rollouts, all_stats)
            return {k: np.mean(v) for k, v in all_stats.items()}

        # Align lengths
        n = min(len(goals_a), len(goals_b))
        goals_a = goals_a[:n]
        goals_b = goals_b[:n]

        lola_active = (self._lola_update_count >= self.lola_warmup)
        self._lola_update_count += 1

        for focal in range(2):
            opponent = 1 - focal
            # Opponent's loss + its gradient w.r.t. opponent params (inner step)
            self.comm_optimizers[focal].zero_grad()
            self.comm_optimizers[opponent].zero_grad()

            L_a, L_b = self._channel_pair_losses(goals_a, goals_b)
            L_focal = L_a if focal == 0 else L_b
            L_opp = L_b if focal == 0 else L_a

            opp_params = list(self.comm_channels[opponent].parameters())

            if lola_active:
                # First-order LOLA correction on focal's loss:
                # L_focal_corrected = L_focal - inner_lr * (grad_opp L_focal) . (grad_opp L_opp)
                # We need create_graph=True on the inner gradient so that
                # focal's params can backprop through it.
                g_focal_wrt_opp = torch.autograd.grad(
                    L_focal, opp_params, create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                g_opp_wrt_opp = torch.autograd.grad(
                    L_opp, opp_params, create_graph=True, retain_graph=True,
                    allow_unused=True,
                )
                lola_term = 0.0
                for gf, go in zip(g_focal_wrt_opp, g_opp_wrt_opp):
                    if gf is None or go is None:
                        continue
                    lola_term = lola_term + (gf * go).sum()
                L_focal_total = L_focal - self.lola_inner_lr * self.lola_coef * lola_term
                all_stats[f'lola_term_agent{focal}'].append(
                    float(lola_term.detach().item()) if torch.is_tensor(lola_term) else 0.0
                )
            else:
                L_focal_total = L_focal

            # Step only focal's comm channel.
            self.comm_optimizers[focal].zero_grad()
            L_focal_total.backward()
            nn.utils.clip_grad_norm_(
                self.comm_channels[focal].parameters(), 1.0
            )
            self.comm_optimizers[focal].step()

            all_stats[f'comm_recon_loss_agent{focal}'].append(float(L_focal.item()))

        # Log combined recon loss for comparability with MAPPO.
        all_stats['comm_recon_loss'].append(
            float(np.mean(all_stats.get('comm_recon_loss_agent0', [0]) +
                          all_stats.get('comm_recon_loss_agent1', [0])))
        )

        return {k: np.mean(v) for k, v in all_stats.items()}

    def _extract_goals(self, manager_rollouts):
        """Pull goal tensors from the manager rollouts (detached)."""
        g = [None, None]
        for a in range(2):
            m_data = manager_rollouts[a]
            if m_data and 'goals' in m_data:
                goals = F.normalize(m_data['goals'].detach(), dim=-1)
                g[a] = goals
        return g[0], g[1]

    def _plain_comm_update(self, manager_rollouts, all_stats):
        """Fallback to plain reconstruction comm update when LOLA can't apply."""
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
            nn.utils.clip_grad_norm_(
                self.comm_channels[a].parameters(), 1.0
            )
            self.comm_optimizers[a].step()
            all_stats['comm_recon_loss'].append(recon_loss.item())
