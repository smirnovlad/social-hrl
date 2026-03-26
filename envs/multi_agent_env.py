"""Multi-agent Minigrid wrapper for social HRL experiments.

Two agents operate in the same Minigrid. They have separate goal locations
and must navigate through shared narrow corridors, creating coordination pressure.
"""

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv


class TwoAgentCorridorEnv(MiniGridEnv):
    """Custom two-agent Minigrid with a shared narrow corridor.

    Layout:
        Room A (left) -- narrow corridor (1 cell wide) -- Room B (right)
        Agent A starts in Room A, needs to reach goal in Room B.
        Agent B starts in Room B, needs to reach goal in Room A.
        The corridor is 1 cell wide, so agents block each other.

    The environment runs two agents sequentially within each step.
    Communication happens between steps.
    """

    def __init__(self, size=11, corridor_length=3, max_steps=200, **kwargs):
        self.corridor_length = corridor_length
        self._size = size

        mission_space = MissionSpace(mission_func=lambda: "navigate to goal")

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

        # Override: we manage two agents manually
        self.agent_positions = [None, None]
        self.agent_dirs = [0, 2]  # A faces right, B faces left
        self.goal_positions = [None, None]
        self.current_agent = 0
        self.agent_dones = [False, False]

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        mid_y = height // 2
        corridor_start = width // 2 - self.corridor_length // 2
        corridor_end = corridor_start + self.corridor_length

        # Create walls to form two rooms with a corridor
        for y in range(1, height - 1):
            if y < mid_y - 1 or y > mid_y + 1:  # Corridor is 3 cells high
                for x in [corridor_start, corridor_end - 1]:
                    if 1 <= x < width - 1:
                        self.grid.set(x, y, Wall())

        # Agent A starts top-left area
        self.agent_positions[0] = (2, 2)
        self.agent_dirs[0] = 0

        # Agent B starts bottom-right area
        self.agent_positions[1] = (width - 3, height - 3)
        self.agent_dirs[1] = 2

        # Goal for A is in B's starting area
        gx_a, gy_a = width - 3, height - 4
        self.grid.set(gx_a, gy_a, Goal())
        self.goal_positions[0] = (gx_a, gy_a)

        # Goal for B is in A's starting area
        gx_b, gy_b = 2, 3
        self.grid.set(gx_b, gy_b, Goal())
        self.goal_positions[1] = (gx_b, gy_b)

        # Set the default agent position for MiniGridEnv
        self.agent_pos = np.array(self.agent_positions[0])
        self.agent_dir = self.agent_dirs[0]

    def get_obs_for_agent(self, agent_idx):
        """Get the observation for a specific agent."""
        old_pos = self.agent_pos.copy()
        old_dir = self.agent_dir

        self.agent_pos = np.array(self.agent_positions[agent_idx])
        self.agent_dir = self.agent_dirs[agent_idx]

        obs = self.gen_obs()

        self.agent_pos = old_pos
        self.agent_dir = old_dir

        return obs['image']

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.agent_dones = [False, False]
        self.current_agent = 0

        obs_a = self.get_obs_for_agent(0)
        obs_b = self.get_obs_for_agent(1)

        return (obs_a, obs_b), info

    def step_agent(self, agent_idx, action):
        """Execute one action for one agent.

        Returns:
            obs: Agent's new observation.
            reward: Reward for this agent.
            done: Whether this agent reached its goal.
        """
        if self.agent_dones[agent_idx]:
            return self.get_obs_for_agent(agent_idx), 0.0, True

        pos = list(self.agent_positions[agent_idx])
        dir_ = self.agent_dirs[agent_idx]

        # Direction vectors: 0=right, 1=down, 2=left, 3=up
        dir_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        reward = -0.01  # Small step penalty

        if action == 0:  # Turn left
            self.agent_dirs[agent_idx] = (dir_ - 1) % 4
        elif action == 1:  # Turn right
            self.agent_dirs[agent_idx] = (dir_ + 1) % 4
        elif action == 2:  # Move forward
            dx, dy = dir_vec[dir_]
            new_x, new_y = pos[0] + dx, pos[1] + dy

            # Check bounds and walls
            cell = self.grid.get(new_x, new_y)
            other_pos = self.agent_positions[1 - agent_idx]

            if cell is None or isinstance(cell, Goal):
                if (new_x, new_y) != tuple(other_pos):  # Can't overlap
                    self.agent_positions[agent_idx] = (new_x, new_y)

                    # Check if reached goal
                    if (new_x, new_y) == self.goal_positions[agent_idx]:
                        reward = 1.0
                        self.agent_dones[agent_idx] = True
                else:
                    reward = -0.05  # Penalty for trying to move into other agent

        obs = self.get_obs_for_agent(agent_idx)
        return obs, reward, self.agent_dones[agent_idx]

    def step(self, actions):
        """Step both agents.

        Args:
            actions: Tuple (action_a, action_b).
        Returns:
            obs: Tuple (obs_a, obs_b).
            rewards: Tuple (reward_a, reward_b).
            dones: Tuple (done_a, done_b).
            truncated: Whether episode is truncated (max steps).
            info: Dict.
        """
        self.step_count += 1

        # Randomly decide who moves first to avoid bias
        order = [0, 1] if np.random.random() > 0.5 else [1, 0]

        obs_list = [None, None]
        reward_list = [0.0, 0.0]
        done_list = [False, False]

        for idx in order:
            obs, rew, done = self.step_agent(idx, actions[idx])
            obs_list[idx] = obs
            reward_list[idx] = rew
            done_list[idx] = done

        # Coordination bonus: if both agents finish, extra reward
        if all(done_list):
            reward_list = [r + 0.5 for r in reward_list]

        truncated = self.step_count >= self.max_steps
        all_done = all(done_list) or truncated

        info = {
            'agent_dones': done_list,
            'agent_positions': list(self.agent_positions),
        }

        return tuple(obs_list), tuple(reward_list), all_done, truncated, info


class MultiAgentWrapper:
    """Wrapper that presents TwoAgentCorridorEnv with a gym-like interface.

    Manages observations and actions for two agents, providing a clean
    interface for the multi-agent trainer.
    """

    def __init__(self, size=11, corridor_length=3, max_steps=200, seed=0):
        self.env = TwoAgentCorridorEnv(
            size=size,
            corridor_length=corridor_length,
            max_steps=max_steps,
        )
        self.observation_space = self.env.observation_space['image']
        self.action_space = self.env.action_space
        self.seed = seed

    def reset(self):
        (obs_a, obs_b), info = self.env.reset(seed=self.seed)
        self.seed += 1
        return (obs_a, obs_b), info

    def step(self, actions):
        return self.env.step(actions)
