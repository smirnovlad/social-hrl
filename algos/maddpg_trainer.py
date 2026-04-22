"""MADDPG for the two-agent corridor env (flat, non-HRL).

Lowe et al. (2017): off-policy multi-agent actor-critic with centralized
critics (that observe all agents) and decentralized actors. We implement the
canonical discrete-action variant using Gumbel-Softmax straight-through for
the actor outputs, as in the original paper's cooperative-navigation setting.

Scope: flat actors (no manager/worker hierarchy) on
TwoAgentCorridorEnv + MultiAgentWrapper, with the same bus/rendezvous cost
model used elsewhere. Goal: a head-to-head reference point against MAPPO-HRL.

Architecture per agent:
    encoder     : Minigrid image -> hidden features
    actor       : features -> action logits (Gumbel-Softmax sample)
    critic      : concat(features_a, features_b, action_a_oh, action_b_oh)
                  -> Q(s, a_all) per agent

Target networks with Polyak averaging; replay buffer; per-step TD update.

Results dict is trimmed relative to the MAPPO-HRL trainer (no goals, no comm
metrics) but exposes 'returns' and 'eval' so it plugs into verify_hypotheses.
"""

import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.multi_agent_env import MultiAgentWrapper
from models.encoder import MinigridEncoder


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class FlatActor(nn.Module):
    def __init__(self, hidden_dim, num_actions, actor_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, num_actions),
        )

    def forward(self, features):
        return self.net(features)


class CentralizedQCritic(nn.Module):
    """Q(features_a, features_b, a_a_oh, a_b_oh) -> scalar."""

    def __init__(self, feat_dim, num_actions, hidden_dim=128):
        super().__init__()
        in_dim = 2 * feat_dim + 2 * num_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fa, fb, a_a, a_b):
        return self.net(torch.cat([fa, fb, a_a, a_b], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.buf = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = self._rng.sample(self.buf, batch_size)
        return batch

    def __len__(self):
        return len(self.buf)


class MaddpgTrainer:
    """Flat MADDPG on the two-agent corridor with Gumbel-Softmax actors."""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        env_cfg = config['env']
        self.max_steps = env_cfg.get('max_steps', 200)
        self.corridor_size = env_cfg.get('corridor_size', 11)
        self.corridor_width = env_cfg.get('corridor_width', 3)
        self.rendezvous_bonus = env_cfg.get('rendezvous_bonus', 0.0)
        self.num_obstacles = env_cfg.get('num_obstacles', 0)
        self.bus_cost_solo = env_cfg.get('bus_cost_solo', 0.0)
        self.bus_cost_shared = env_cfg.get('bus_cost_shared', 0.0)
        self.bus_window = env_cfg.get('bus_window', 0)
        self.turn_taking = env_cfg.get('turn_taking', False)

        self.eval_episodes = config['experiment'].get('eval_episodes', 10)
        self.total_timesteps = config['ppo']['total_timesteps']
        self.seed = config['experiment']['seed']
        self.num_envs = 1  # MADDPG is off-policy; single-env collection is fine

        # Env
        self.env = MultiAgentWrapper(
            size=self.corridor_size, corridor_length=3,
            max_steps=self.max_steps, seed=self.seed,
            corridor_width=self.corridor_width,
            asymmetric_info=env_cfg.get('asymmetric_info', False),
            rendezvous_bonus=self.rendezvous_bonus,
            num_obstacles=self.num_obstacles,
            bus_cost_solo=self.bus_cost_solo,
            bus_cost_shared=self.bus_cost_shared,
            bus_window=self.bus_window,
            turn_taking=self.turn_taking,
        )
        obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        hidden_dim = config['encoder']['hidden_dim']
        self.feat_dim = hidden_dim

        # Per-agent encoders + actors (decentralized execution).
        self.encoders = nn.ModuleList([
            MinigridEncoder(obs_shape,
                            channels=tuple(config['encoder']['channels']),
                            hidden_dim=hidden_dim)
            for _ in range(2)
        ]).to(self.device)
        self.actors = nn.ModuleList([
            FlatActor(hidden_dim, self.num_actions) for _ in range(2)
        ]).to(self.device)

        # Centralized critics (one per agent).
        self.critics = nn.ModuleList([
            CentralizedQCritic(hidden_dim, self.num_actions) for _ in range(2)
        ]).to(self.device)

        # Target nets.
        import copy
        self.target_encoders = copy.deepcopy(self.encoders).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        for m in (self.target_encoders, self.target_actors, self.target_critics):
            for p in m.parameters():
                p.requires_grad = False

        lr = config['ppo']['lr']
        # Shared encoder lr with actor, critic has its own param group.
        self.actor_optims = [
            torch.optim.Adam(
                list(self.actors[a].parameters())
                + list(self.encoders[a].parameters()),
                lr=lr, eps=1e-5,
            )
            for a in range(2)
        ]
        self.critic_optims = [
            torch.optim.Adam(self.critics[a].parameters(), lr=lr, eps=1e-5)
            for a in range(2)
        ]

        self.gamma = config['ppo']['gamma']
        self.tau_polyak = 0.01
        self.gumbel_tau = 1.0
        # Standard MADDPG hyperparams.
        self.batch_size = 64
        self.buffer = ReplayBuffer(capacity=50_000, seed=self.seed)
        self.warmup_steps = 500
        self.update_every = 4
        self.global_step = 0

    def _encode(self, obs_t, agent_idx, target=False):
        mod = self.target_encoders[agent_idx] if target else self.encoders[agent_idx]
        return mod(obs_t)

    def _act(self, features, agent_idx, target=False, noise_std=0.0):
        mod = self.target_actors[agent_idx] if target else self.actors[agent_idx]
        logits = mod(features)
        if noise_std > 0:
            logits = logits + noise_std * torch.randn_like(logits)
        # Gumbel-Softmax straight-through for discrete actions.
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = F.softmax((logits + gumbels) / self.gumbel_tau, dim=-1)
        idx = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        y = y_hard - y_soft.detach() + y_soft
        return y, idx.squeeze(-1)

    def _sample_action_from_obs(self, obs_tuple, noise_std=0.0):
        """Select a discrete action per agent for env stepping."""
        actions = []
        onehots = []
        for a in range(2):
            obs_t = torch.from_numpy(obs_tuple[a]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._encode(obs_t, a)
                oh, idx = self._act(feat, a, noise_std=noise_std)
            actions.append(int(idx.item()))
            onehots.append(oh.squeeze(0).cpu().numpy())
        return tuple(actions), onehots

    def _one_hot(self, action_indices):
        # action_indices: LongTensor of shape (B,)
        return F.one_hot(action_indices, self.num_actions).float()

    def _update_once(self):
        if len(self.buffer) < self.batch_size or len(self.buffer) < self.warmup_steps:
            return {}
        batch = self.buffer.sample(self.batch_size)
        # Unpack: each transition is (obs_a, obs_b, act_a, act_b, rew_a, rew_b,
        #                             next_obs_a, next_obs_b, done)
        obs_a = torch.from_numpy(np.stack([t[0] for t in batch])).to(self.device)
        obs_b = torch.from_numpy(np.stack([t[1] for t in batch])).to(self.device)
        act_a = torch.tensor([t[2] for t in batch], dtype=torch.long, device=self.device)
        act_b = torch.tensor([t[3] for t in batch], dtype=torch.long, device=self.device)
        rew_a = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)
        rew_b = torch.tensor([t[5] for t in batch], dtype=torch.float32, device=self.device)
        next_obs_a = torch.from_numpy(np.stack([t[6] for t in batch])).to(self.device)
        next_obs_b = torch.from_numpy(np.stack([t[7] for t in batch])).to(self.device)
        done = torch.tensor([t[8] for t in batch], dtype=torch.float32, device=self.device)

        act_a_oh = self._one_hot(act_a)
        act_b_oh = self._one_hot(act_b)
        rewards = [rew_a, rew_b]

        stats = {}

        # --- Critic updates ---
        with torch.no_grad():
            n_feat_a = self._encode(next_obs_a, 0, target=True)
            n_feat_b = self._encode(next_obs_b, 1, target=True)
            n_act_a, _ = self._act(n_feat_a, 0, target=True)
            n_act_b, _ = self._act(n_feat_b, 1, target=True)

        for a in range(2):
            feat_a = self._encode(obs_a, 0).detach()
            feat_b = self._encode(obs_b, 1).detach()
            q = self.critics[a](feat_a, feat_b, act_a_oh, act_b_oh)
            with torch.no_grad():
                tq = self.target_critics[a](n_feat_a, n_feat_b, n_act_a, n_act_b)
                target = rewards[a] + self.gamma * (1.0 - done) * tq
            loss_q = F.mse_loss(q, target)
            self.critic_optims[a].zero_grad()
            loss_q.backward()
            nn.utils.clip_grad_norm_(self.critics[a].parameters(), 1.0)
            self.critic_optims[a].step()
            stats[f'critic_loss_a{a}'] = float(loss_q.item())

        # --- Actor updates ---
        for a in range(2):
            # Policy gradient: maximize Q under own current actor, others held.
            feat_a = self._encode(obs_a, 0)
            feat_b = self._encode(obs_b, 1)
            cur_act_a, _ = self._act(feat_a, 0)
            cur_act_b, _ = self._act(feat_b, 1)
            if a == 0:
                act_other = cur_act_b.detach()
                act_own = cur_act_a
            else:
                act_other = cur_act_a.detach()
                act_own = cur_act_b
            if a == 0:
                q = self.critics[a](feat_a, feat_b.detach(), act_own, act_other)
            else:
                q = self.critics[a](feat_a.detach(), feat_b, act_other, act_own)
            loss_pi = -q.mean()
            self.actor_optims[a].zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(
                list(self.actors[a].parameters())
                + list(self.encoders[a].parameters()), 1.0)
            self.actor_optims[a].step()
            stats[f'actor_loss_a{a}'] = float(loss_pi.item())

        # Polyak target update.
        _soft_update(self.target_encoders, self.encoders, self.tau_polyak)
        _soft_update(self.target_actors, self.actors, self.tau_polyak)
        _soft_update(self.target_critics, self.critics, self.tau_polyak)
        return stats

    def train(self, output_dir=None, wandb_run=None):
        """Run a full MADDPG training loop."""
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        (obs_a, obs_b), _ = self.env.reset()
        ep_ret_a, ep_ret_b = 0.0, 0.0
        returns = []

        # Explore with higher logit noise initially, decaying linearly.
        noise_std_init = 1.0
        noise_std_end = 0.1

        log_interval = max(self.total_timesteps // 50, 200)
        last_stats = {}
        t0 = time.time()
        history = []
        while self.global_step < self.total_timesteps:
            frac = min(1.0, self.global_step / max(1, self.total_timesteps))
            noise_std = noise_std_init + frac * (noise_std_end - noise_std_init)
            actions, _ = self._sample_action_from_obs(
                (obs_a, obs_b), noise_std=noise_std
            )
            (next_obs_a, next_obs_b), (r_a, r_b), terminated, truncated, _ = \
                self.env.step(actions)
            done = float(terminated)
            self.buffer.push((
                obs_a, obs_b, actions[0], actions[1], float(r_a), float(r_b),
                next_obs_a, next_obs_b, done,
            ))
            ep_ret_a += r_a
            ep_ret_b += r_b
            obs_a, obs_b = next_obs_a, next_obs_b

            if terminated or truncated:
                returns.append((ep_ret_a + ep_ret_b) / 2.0)
                ep_ret_a, ep_ret_b = 0.0, 0.0
                (obs_a, obs_b), _ = self.env.reset()

            self.global_step += 1
            if self.global_step % self.update_every == 0:
                stats = self._update_once() or {}
                if stats:
                    last_stats = stats

            if wandb_run is not None and self.global_step % log_interval == 0:
                mean_ret = float(np.mean(returns[-50:])) if returns else 0.0
                sps = self.global_step / max(1e-6, time.time() - t0)
                log_data = {
                    'global_step': self.global_step,
                    'mean_return': mean_ret,
                    'sps': sps,
                    'noise_std': noise_std,
                    'replay_buffer_size': len(self.buffer),
                    'episodes': len(returns),
                }
                log_data.update(last_stats)
                history.append(log_data.copy())
                wandb_run.log(log_data, step=self.global_step)
            elif self.global_step % log_interval == 0:
                mean_ret = float(np.mean(returns[-50:])) if returns else 0.0
                sps = self.global_step / max(1e-6, time.time() - t0)
                log_data = {
                    'global_step': self.global_step,
                    'mean_return': mean_ret,
                    'sps': sps,
                    'noise_std': noise_std,
                    'replay_buffer_size': len(self.buffer),
                    'episodes': len(returns),
                }
                log_data.update(last_stats)
                history.append(log_data.copy())

        eval_result = self.evaluate(num_episodes=self.eval_episodes)
        dt = time.time() - t0
        print(f"[maddpg] total train time {dt:.1f}s  "
              f"final return={np.mean(returns[-50:]) if returns else 0:.3f}  "
              f"eval success={eval_result['success_rate']:.2f}")

        if output_dir:
            try:
                torch.save({
                    'mode': 'maddpg',
                    'global_step': self.global_step,
                    'config': self.config,
                    'encoders': self.encoders.state_dict(),
                    'actors': self.actors.state_dict(),
                    'critics': self.critics.state_dict(),
                }, os.path.join(output_dir, 'final.pt'))
            except Exception:
                pass

        return {
            'returns': returns,
            'eval': eval_result,
            # Keep shape-compatible with verify harness.
            'messages': None,
            'states': None,
            'decoded_goals': None,
            'recon_loss_mean': None,
            'temporal_extent_mean': None,
            'comm_ablation_eval': None,
            'history': history,
        }

    def evaluate(self, num_episodes=10):
        returns = []
        successes = []
        for ep in range(num_episodes):
            env = MultiAgentWrapper(
                size=self.corridor_size, corridor_length=3,
                max_steps=self.max_steps,
                seed=self.seed + 100_000 + ep,
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
            done = False
            ep_ret_a, ep_ret_b = 0.0, 0.0
            reached_both = False
            while not done:
                actions = []
                for a, o in enumerate((obs_a, obs_b)):
                    obs_t = torch.from_numpy(o).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self._encode(obs_t, a)
                        logits = self.actors[a](feat)
                        actions.append(int(logits.argmax(dim=-1).item()))
                (obs_a, obs_b), (r_a, r_b), term, trunc, info = env.step(tuple(actions))
                ep_ret_a += r_a
                ep_ret_b += r_b
                if info.get('agent_dones', [False, False]) == [True, True]:
                    reached_both = True
                done = term or trunc
            returns.append((ep_ret_a + ep_ret_b) / 2.0)
            successes.append(float(reached_both))
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'success_rate': float(np.mean(successes)),
        }
