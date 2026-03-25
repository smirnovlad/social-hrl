"""Environment wrappers for Minigrid."""

import gymnasium as gym
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
        return obs['image'], reward, terminated, truncated, info


def make_env(env_name, seed=0, max_steps=None):
    """Create a single Minigrid environment with wrappers.

    Args:
        env_name: Minigrid environment ID.
        seed: Random seed.
        max_steps: Override max steps (None = use default).
    Returns:
        Wrapped gymnasium environment.
    """
    def _init():
        kwargs = {}
        if max_steps is not None:
            kwargs['max_steps'] = max_steps
        env = gym.make(env_name, **kwargs)
        env = MinigridWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(env_name, num_envs, seed=0, max_steps=None):
    """Create vectorized Minigrid environments.

    Args:
        env_name: Minigrid environment ID.
        num_envs: Number of parallel environments.
        seed: Base random seed.
        max_steps: Override max steps.
    Returns:
        gymnasium.vector.SyncVectorEnv
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, seed + i, max_steps) for i in range(num_envs)]
    )
    return envs
