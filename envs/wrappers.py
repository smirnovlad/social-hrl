"""Environment wrappers for Minigrid."""

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np


class MinigridWrapper(gym.Wrapper):
    """Wraps Minigrid to return image observations as numpy arrays.

    Minigrid returns dict observations with 'image', 'direction', 'mission'.
    We extract just the image for our CNN encoder.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['image']

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs['image'], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault('base_reward', reward)
        info.setdefault('success', bool(terminated and reward > 0.0))
        return obs['image'], reward, terminated, truncated, info


class DistanceShapingWrapper(gym.Wrapper):
    """Potential-based distance shaping for sparse-reward MiniGrid tasks.

    Adds a dense per-step reward equal to alpha * (prev_dist - curr_dist), where
    `dist` is the Manhattan distance from the agent to the current subgoal.
    Subgoals change as progress is made (e.g., KeyCorridor: key -> door -> object).
    When the subgoal changes we reset the baseline so we don't double-count the
    discontinuity as a single-step reward spike.

    This preserves the optimal policy (potential-based shaping) while providing
    a learning signal that lets PPO find the solution in a feasible number of
    steps.

    Supported envs: KeyCorridor*, MultiRoom*. For anything else the wrapper
    is a no-op passthrough.
    """

    def __init__(self, env, alpha=0.05):
        super().__init__(env)
        self.alpha = alpha
        unwrapped = env.unwrapped
        cls_name = unwrapped.__class__.__name__
        if 'KeyCorridor' in cls_name:
            self._env_kind = 'keycorridor'
        elif 'MultiRoom' in cls_name:
            self._env_kind = 'multiroom'
        else:
            self._env_kind = None
        self._prev_subgoal = None
        self._prev_dist = None

    def _agent_pos(self):
        return tuple(self.env.unwrapped.agent_pos)

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_grid_object(self, predicate):
        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and predicate(cell):
                    return (x, y)
        return None

    def _current_subgoal(self):
        """Return (subgoal_tag, target_pos) for potential-based shaping."""
        env = self.env.unwrapped
        if self._env_kind == 'keycorridor':
            # Phase 1: no key carried -> go to key
            carrying = getattr(env, 'carrying', None)
            if carrying is None or getattr(carrying, 'type', None) != 'key':
                key_pos = self._find_grid_object(lambda c: c.type == 'key')
                if key_pos is not None:
                    return ('key', key_pos)
            # Phase 2: key in hand, door still locked -> go to door
            door_pos = self._find_grid_object(
                lambda c: c.type == 'door' and getattr(c, 'is_locked', False)
            )
            if door_pos is not None:
                return ('door', door_pos)
            # Phase 3: door open -> go to target object
            obj = getattr(env, 'obj', None)
            if obj is not None and getattr(obj, 'cur_pos', None) is not None:
                return ('obj', tuple(obj.cur_pos))
            return None
        if self._env_kind == 'multiroom':
            goal_pos = getattr(env, 'goal_pos', None)
            if goal_pos is None:
                # Fall back to searching for a goal tile
                goal_pos = self._find_grid_object(lambda c: c.type == 'goal')
            if goal_pos is not None:
                return ('goal', tuple(goal_pos))
            return None
        return None

    def _reset_shaping(self):
        self._prev_subgoal = self._current_subgoal()
        if self._prev_subgoal is not None:
            self._prev_dist = self._manhattan(self._agent_pos(), self._prev_subgoal[1])
        else:
            self._prev_dist = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._reset_shaping()
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault('base_reward', reward)
        if self._env_kind is not None:
            subgoal = self._current_subgoal()
            if subgoal is not None:
                curr_dist = self._manhattan(self._agent_pos(), subgoal[1])
                if (self._prev_subgoal is not None
                        and subgoal[0] == self._prev_subgoal[0]
                        and self._prev_dist is not None):
                    reward = reward + self.alpha * (self._prev_dist - curr_dist)
                self._prev_subgoal = subgoal
                self._prev_dist = curr_dist
        info.setdefault('success', bool(terminated and info.get('base_reward', 0.0) > 0.0))
        return obs, reward, terminated, truncated, info


def _env_needs_shaping(env_name):
    return ('KeyCorridor' in env_name) or ('MultiRoom' in env_name)


def make_env(env_name, seed=0, max_steps=None, shape_rewards=None, shaping_alpha=0.05):
    """Create a single Minigrid environment with wrappers.

    Args:
        env_name: Minigrid environment ID.
        seed: Random seed.
        max_steps: Override max steps (None = use default).
        shape_rewards: If True, apply distance shaping. If None (default),
            auto-enable for KeyCorridor* and MultiRoom*.
        shaping_alpha: Scale factor for the shaping bonus.
    Returns:
        Wrapped gymnasium environment.
    """
    if shape_rewards is None:
        shape_rewards = _env_needs_shaping(env_name)

    def _init():
        kwargs = {}
        if max_steps is not None:
            kwargs['max_steps'] = max_steps
        env = gym.make(env_name, **kwargs)
        env = MinigridWrapper(env)
        if shape_rewards:
            env = DistanceShapingWrapper(env, alpha=shaping_alpha)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(env_name, num_envs, seed=0, max_steps=None,
                 shape_rewards=None, shaping_alpha=0.05):
    """Create vectorized Minigrid environments.

    Args:
        env_name: Minigrid environment ID.
        num_envs: Number of parallel environments.
        seed: Base random seed.
        max_steps: Override max steps.
        shape_rewards: If True, apply distance shaping. If None, auto-enable
            for KeyCorridor* and MultiRoom*.
        shaping_alpha: Scale factor for the shaping bonus.
    Returns:
        gymnasium.vector.SyncVectorEnv
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, seed + i, max_steps,
                  shape_rewards=shape_rewards, shaping_alpha=shaping_alpha)
         for i in range(num_envs)]
    )
    return envs
