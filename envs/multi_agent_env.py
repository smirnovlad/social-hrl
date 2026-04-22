"""Multi-agent Minigrid wrapper for social HRL experiments.

Two agents operate in the same Minigrid. They have separate goal locations
and must navigate through shared narrow corridors, creating coordination pressure.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv


class TwoAgentCorridorEnv(MiniGridEnv):
    """Custom two-agent Minigrid with a shared narrow corridor.

    Layout:
        Room A (left) -- corridor -- Room B (right)
        Agent A starts in Room A, needs to reach goal in Room B.
        Agent B starts in Room B, needs to reach goal in Room A.

    Args:
        corridor_width: Width of the corridor (1=narrow/blocking, 3=wide).
        asymmetric_info: If True, agent 1 cannot see goal tiles.
    """

    def __init__(self, size=11, corridor_length=3, max_steps=200,
                 corridor_width=3, asymmetric_info=False,
                 rendezvous_bonus=0.0, num_obstacles=0,
                 bus_cost_solo=0.0, bus_cost_shared=0.0,
                 bus_window=0, turn_taking=False,
                 randomize_goals=False, mutual_goal_blind=False, **kwargs):
        self.corridor_length = corridor_length
        self._size = size
        self.corridor_width = corridor_width
        self.asymmetric_info = asymmetric_info
        self.rendezvous_bonus = rendezvous_bonus
        self.num_obstacles = num_obstacles
        # Per-episode randomization of goal tiles. When False, goals sit at the
        # fixed positions that gave the old "two agents can solve it
        # independently" setup; when True, each agent's goal is resampled in
        # the opposite room every reset, so the partner's goal location becomes
        # episode-dependent information.
        self.randomize_goals = randomize_goals
        # Hard information asymmetry: each agent's partial view has its OWN
        # goal tile masked, but the partner's goal tile remains visible. With
        # randomize_goals=True this makes the partner's message the only path
        # to learning where you need to navigate, so the comm channel is
        # forced to carry non-trivial coordination information.
        self.mutual_goal_blind = mutual_goal_blind
        # Shared-bus resource: being in the corridor alone costs `bus_cost_solo`
        # per step; both-agents-in-corridor costs `bus_cost_shared` (typically 0
        # or small). Cheaper-per-agent-when-simultaneous => coordination pressure.
        self.bus_cost_solo = bus_cost_solo
        self.bus_cost_shared = bus_cost_shared
        # Strict-bus time window: if >0, the shared discount only applies when
        # both agents entered the corridor within this many steps of each
        # other. A later-arriving partner does not rescue a solo entry -- this
        # enforces "coordinate arrival times" rather than just "be simultaneous
        # at some point." 0 disables the window check.
        self.bus_window = bus_window
        # Turn-taking coordination: if True, exactly one agent moves per global
        # step (alternating). The non-active agent's action is ignored. This
        # is the RQ4 turn-taking scenario.
        self.turn_taking = turn_taking
        self._bus_enter_step = [None, None]

        mission_space = MissionSpace(mission_func=lambda: "navigate to goal")

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

        # Only left/right/forward are meaningful in this env
        self.action_space = spaces.Discrete(3)

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

        half = self.corridor_width // 2

        # Create walls to form two rooms with a corridor
        for y in range(1, height - 1):
            if y < mid_y - half or y > mid_y + half:
                for x in [corridor_start, corridor_end - 1]:
                    if 1 <= x < width - 1:
                        self.grid.set(x, y, Wall())

        # Agent A starts top-left area
        self.agent_positions[0] = (2, 2)
        self.agent_dirs[0] = 0

        # Agent B starts bottom-right area
        self.agent_positions[1] = (width - 3, height - 3)
        self.agent_dirs[1] = 2

        # Goal placement. With randomize_goals=False we preserve the fixed
        # layout the earlier sweeps were trained on. With True we sample each
        # agent's goal uniformly from interior tiles on the partner's side of
        # the corridor, skipping walls, the corridor zone, and the two agent
        # start tiles.
        if self.randomize_goals:
            half = self.corridor_width // 2
            def _sample_goal(x_lo, x_hi):
                for _ in range(200):
                    gx = self._rand_int(x_lo, x_hi)
                    gy = self._rand_int(1, height - 1)
                    if (gx, gy) == tuple(self.agent_positions[0]):
                        continue
                    if (gx, gy) == tuple(self.agent_positions[1]):
                        continue
                    if corridor_start <= gx < corridor_end and \
                            mid_y - half <= gy <= mid_y + half:
                        continue
                    if self.grid.get(gx, gy) is not None:
                        continue
                    return gx, gy
                # Fall back to the deterministic tile if sampling fails.
                return None

            picked_a = _sample_goal(corridor_end, width - 1)
            gx_a, gy_a = picked_a if picked_a is not None else (width - 3, height - 4)
            picked_b = _sample_goal(1, corridor_start)
            gx_b, gy_b = picked_b if picked_b is not None else (2, 3)
        else:
            gx_a, gy_a = width - 3, height - 4
            gx_b, gy_b = 2, 3

        self.grid.set(gx_a, gy_a, Goal())
        self.goal_positions[0] = (gx_a, gy_a)
        self.grid.set(gx_b, gy_b, Goal())
        self.goal_positions[1] = (gx_b, gy_b)

        # Place random obstacles in the rooms
        if self.num_obstacles > 0:
            reserved = set()
            reserved.add(tuple(self.agent_positions[0]))
            reserved.add(tuple(self.agent_positions[1]))
            reserved.add(self.goal_positions[0])
            reserved.add(self.goal_positions[1])
            for x in range(corridor_start, corridor_end):
                for y in range(mid_y - half, mid_y + half + 1):
                    reserved.add((x, y))

            placed = 0
            attempts = 0
            while placed < self.num_obstacles and attempts < 200:
                ox = self._rand_int(1, width - 1)
                oy = self._rand_int(1, height - 1)
                if (ox, oy) not in reserved and self.grid.get(ox, oy) is None:
                    self.grid.set(ox, oy, Wall())
                    reserved.add((ox, oy))
                    placed += 1
                attempts += 1

        # Set the default agent position for MiniGridEnv
        self.agent_pos = np.array(self.agent_positions[0])
        self.agent_dir = self.agent_dirs[0]

    def _in_corridor(self, pos):
        """Check if a position is within the corridor zone."""
        x, y = pos
        mid_y = self._size // 2
        half = self.corridor_width // 2
        corridor_start = self._size // 2 - self.corridor_length // 2
        corridor_end = corridor_start + self.corridor_length
        return (corridor_start <= x < corridor_end and
                mid_y - half <= y <= mid_y + half)

    def get_obs_for_agent(self, agent_idx):
        """Get the observation for a specific agent."""
        old_pos = self.agent_pos.copy()
        old_dir = self.agent_dir

        self.agent_pos = np.array(self.agent_positions[agent_idx])
        self.agent_dir = self.agent_dirs[agent_idx]

        # Under mutual_goal_blind, temporarily remove *this* agent's own goal
        # tile from the grid so gen_obs renders a view with only the partner's
        # goal visible. The partner's message is then the only source of the
        # agent's own goal location.
        swapped_own_goal = None
        if self.mutual_goal_blind and self.goal_positions[agent_idx] is not None:
            ogx, ogy = self.goal_positions[agent_idx]
            swapped_own_goal = (ogx, ogy, self.grid.get(ogx, ogy))
            self.grid.set(ogx, ogy, None)

        obs = self.gen_obs()

        if swapped_own_goal is not None:
            ogx, ogy, orig = swapped_own_goal
            self.grid.set(ogx, ogy, orig)

        self.agent_pos = old_pos
        self.agent_dir = old_dir

        image = obs['image']

        # Asymmetric info: agent 1 cannot see goal tiles in the corridor task.
        if self.asymmetric_info and agent_idx == 1:
            mask = image[:, :, 0] == OBJECT_TO_IDX['goal']
            image[mask, 0] = 1  # empty
            image[mask, 1] = 0  # no color
            image[mask, 2] = 0  # no state

        return image

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.agent_dones = [False, False]
        self.current_agent = 0
        self._bus_enter_step = [None, None]

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
        gx, gy = self.goal_positions[agent_idx]
        old_dist = abs(pos[0] - gx) + abs(pos[1] - gy)

        # Direction vectors: 0=right, 1=down, 2=left, 3=up
        dir_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        reward = -0.01  # Small step penalty

        if action is None:
            # True no-op for turn-taking: time advances, but orientation and
            # position do not change.
            pass
        elif action == 0:  # Turn left
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

        # Distance-based reward shaping: bonus for getting closer to goal
        new_pos = self.agent_positions[agent_idx]
        new_dist = abs(new_pos[0] - gx) + abs(new_pos[1] - gy)
        reward += 0.05 * (old_dist - new_dist)

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

        # Turn-taking: only one agent acts per global step (alternating).
        # The other agent receives a true no-op so its orientation does not
        # drift while it waits.
        if self.turn_taking:
            active = (self.step_count - 1) % 2
            effective_actions = [None, None]
            effective_actions[active] = actions[active]
            order = [active, 1 - active]
        else:
            # Randomly decide who moves first to avoid bias
            order = [0, 1] if self.np_random.random() > 0.5 else [1, 0]
            effective_actions = actions

        # Capture pre-step done state for coordination bonus detection
        prev_both_done = all(self.agent_dones)

        obs_list = [None, None]
        reward_list = [0.0, 0.0]
        done_list = [False, False]

        for idx in order:
            obs, rew, done = self.step_agent(idx, effective_actions[idx])
            obs_list[idx] = obs
            reward_list[idx] = rew
            done_list[idx] = done

        # Coordination bonus: only on the exact step both become done
        if all(done_list) and not prev_both_done:
            reward_list = [r + 0.5 for r in reward_list]

        # Rendezvous bonus: both agents in the corridor simultaneously
        a_in = self._in_corridor(self.agent_positions[0])
        b_in = self._in_corridor(self.agent_positions[1])
        both_in_corridor = a_in and b_in
        if self.rendezvous_bonus > 0 and both_in_corridor:
            reward_list = [r + self.rendezvous_bonus for r in reward_list]

        # Track corridor-entry step per agent for bus_window checks.
        for i, in_now in enumerate([a_in, b_in]):
            if in_now and self._bus_enter_step[i] is None:
                self._bus_enter_step[i] = self.step_count
            elif not in_now:
                self._bus_enter_step[i] = None

        # Shared-bus cost model (suggested-approach "cheaper when simultaneous").
        # Only charge agents that are actually in the bus zone. Shared cost is
        # applied to both; solo cost is applied only to the one agent in.
        # bus_window > 0: the shared discount only fires if both agents entered
        # the corridor within that many steps of each other; otherwise solo
        # cost is charged to each -- enforcing "coordinate arrival times."
        if self.bus_cost_solo != 0.0 or self.bus_cost_shared != 0.0:
            if both_in_corridor:
                window_ok = True
                if self.bus_window > 0:
                    ea, eb = self._bus_enter_step
                    if ea is None or eb is None:
                        window_ok = False
                    else:
                        window_ok = abs(ea - eb) <= self.bus_window
                if window_ok:
                    reward_list = [r - self.bus_cost_shared for r in reward_list]
                else:
                    reward_list = [r - self.bus_cost_solo for r in reward_list]
            else:
                if a_in:
                    reward_list[0] -= self.bus_cost_solo
                if b_in:
                    reward_list[1] -= self.bus_cost_solo

        terminated = all(done_list)
        truncated = self.step_count >= self.max_steps

        info = {
            'agent_dones': done_list,
            'agent_successes': list(self.agent_dones),
            'agent_positions': list(self.agent_positions),
            'both_in_corridor': both_in_corridor,
            'success': all(self.agent_dones),
        }

        return tuple(obs_list), tuple(reward_list), terminated, truncated, info


class SingleAgentCorridorEnv(gym.Wrapper):
    """Single-agent version of the corridor env for fair comparison.

    Same layout as TwoAgentCorridorEnv but with only agent A.
    Agent must navigate from top-left to goal in bottom-right through corridor.
    Returns gym-compatible (obs, reward, terminated, truncated, info).
    """

    def __init__(self, size=11, corridor_length=3, max_steps=200, corridor_width=3,
                 randomize_goals=False):
        env = TwoAgentCorridorEnv(
            size=size, corridor_length=corridor_length, max_steps=max_steps,
            corridor_width=corridor_width,
            randomize_goals=randomize_goals,
        )
        super().__init__(env)
        self.observation_space = env.observation_space['image']

    def reset(self, **kwargs):
        (obs_a, _obs_b), info = self.env.reset(**kwargs)
        return obs_a, info

    def step(self, action):
        # Step agent A only; agent B receives a true no-op inside the env.
        (obs_a, _), (rew_a, _), _terminated, truncated, info = self.env.step((action, None))
        agent_a_done = self.env.agent_dones[0]
        info = dict(info)
        info['success'] = bool(agent_a_done)
        return obs_a, rew_a, agent_a_done, truncated, info


def make_corridor_vec_env(num_envs, seed=0, max_steps=200, single_agent=True,
                          corridor_width=3, corridor_size=11,
                          randomize_goals=False):
    """Create vectorized corridor environments.

    Args:
        num_envs: Number of parallel envs.
        seed: Base seed.
        max_steps: Max steps per episode.
        single_agent: If True, single-agent version for discrete mode comparison.
        corridor_width: Width of the corridor passage (1=narrow, 3=wide).
        randomize_goals: Resample goal tiles each episode instead of using the
            fixed (2,3) / (W-3, H-4) layout.
    """
    def _make(i):
        def _init():
            env = SingleAgentCorridorEnv(
                size=corridor_size, corridor_length=3, max_steps=max_steps,
                corridor_width=corridor_width,
                randomize_goals=randomize_goals,
            )
            env.reset(seed=seed + i)
            return env
        return _init

    return gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])


class MultiAgentWrapper:
    """Wrapper that presents TwoAgentCorridorEnv with a gym-like interface.

    Manages observations and actions for two agents, providing a clean
    interface for the multi-agent trainer.
    """

    def __init__(self, size=11, corridor_length=3, max_steps=200, seed=0,
                 corridor_width=3, asymmetric_info=False,
                 rendezvous_bonus=0.0, num_obstacles=0,
                 bus_cost_solo=0.0, bus_cost_shared=0.0,
                 bus_window=0, turn_taking=False,
                 randomize_goals=False, mutual_goal_blind=False):
        self.env = TwoAgentCorridorEnv(
            size=size,
            corridor_length=corridor_length,
            max_steps=max_steps,
            corridor_width=corridor_width,
            asymmetric_info=asymmetric_info,
            rendezvous_bonus=rendezvous_bonus,
            num_obstacles=num_obstacles,
            bus_cost_solo=bus_cost_solo,
            bus_cost_shared=bus_cost_shared,
            bus_window=bus_window,
            turn_taking=turn_taking,
            randomize_goals=randomize_goals,
            mutual_goal_blind=mutual_goal_blind,
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
