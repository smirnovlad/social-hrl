import unittest

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

from envs.multi_agent_env import (
    SingleAgentCorridorEnv,
    TwoAgentCorridorEnv,
    make_corridor_vec_env,
)


class CorridorEnvTests(unittest.TestCase):
    def test_make_corridor_vec_env_threads_corridor_size(self):
        envs = make_corridor_vec_env(num_envs=1, seed=0, corridor_size=15)
        try:
            self.assertEqual(envs.envs[0].unwrapped._size, 15)
        finally:
            envs.close()

    def test_asymmetric_info_masks_goal_tiles_for_agent_one(self):
        plain_env = TwoAgentCorridorEnv(asymmetric_info=False)
        masked_env = TwoAgentCorridorEnv(asymmetric_info=True)
        try:
            plain_env.reset(seed=123)
            masked_env.reset(seed=123)

            goal_x, goal_y = plain_env.goal_positions[0]
            for env in (plain_env, masked_env):
                env.agent_positions[1] = (goal_x, goal_y + 1)
                env.agent_dirs[1] = 3  # face upward toward the goal

            plain_obs = plain_env.get_obs_for_agent(1)
            masked_obs = masked_env.get_obs_for_agent(1)

            self.assertTrue(np.any(plain_obs[:, :, 0] == OBJECT_TO_IDX["goal"]))
            self.assertFalse(np.any(masked_obs[:, :, 0] == OBJECT_TO_IDX["goal"]))
        finally:
            plain_env.close()
            masked_env.close()

    def test_move_order_is_reproducible_under_fixed_seed(self):
        env_a = TwoAgentCorridorEnv()
        env_b = TwoAgentCorridorEnv()
        actions = [(2, 2), (0, 2), (2, 0), (1, 2), (2, 2)]

        try:
            env_a.reset(seed=321)
            env_b.reset(seed=321)

            trajectory_a = []
            trajectory_b = []

            for action in actions:
                step_a = env_a.step(action)
                step_b = env_b.step(action)

                trajectory_a.append((step_a[1], step_a[2], step_a[3], list(env_a.agent_positions)))
                trajectory_b.append((step_b[1], step_b[2], step_b[3], list(env_b.agent_positions)))

            self.assertEqual(trajectory_a, trajectory_b)
        finally:
            env_a.close()
            env_b.close()

    def test_turn_taking_uses_true_noop_for_inactive_agent(self):
        env = TwoAgentCorridorEnv(turn_taking=True)
        try:
            env.reset(seed=7)
            initial_dir = env.agent_dirs[1]
            initial_pos = env.agent_positions[1]

            env.step((2, 1))

            self.assertEqual(env.agent_dirs[1], initial_dir)
            self.assertEqual(env.agent_positions[1], initial_pos)
        finally:
            env.close()

    def test_single_agent_corridor_reports_success_in_info(self):
        env = SingleAgentCorridorEnv()
        try:
            env.reset(seed=11)
            goal_x, goal_y = env.env.goal_positions[0]
            env.env.agent_positions[0] = (goal_x - 1, goal_y)
            env.env.agent_dirs[0] = 0  # face right toward the goal

            _, reward, terminated, truncated, info = env.step(2)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
            self.assertGreater(reward, 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
