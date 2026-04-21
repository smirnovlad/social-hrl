import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiment_utils import build_run_metadata, discover_runs


class ExperimentUtilsTests(unittest.TestCase):
    def test_build_run_metadata_distinguishes_envs_for_same_mode_and_seed(self):
        keycorridor = build_run_metadata(
            mode="discrete",
            seed=42,
            env_name="MiniGrid-KeyCorridorS6R3-v0",
        )
        multiroom = build_run_metadata(
            mode="discrete",
            seed=42,
            env_name="MiniGrid-MultiRoom-N6-v0",
        )

        self.assertNotEqual(keycorridor["condition_id"], multiroom["condition_id"])
        self.assertNotEqual(keycorridor["run_slug"], multiroom["run_slug"])
        self.assertEqual(keycorridor["task_family"], "keycorridor")
        self.assertEqual(multiroom["task_family"], "multiroom")

    def test_discover_runs_prefers_metadata_backed_runs_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            metadata_run = root / "runs" / "mode-discrete__task-keycorridor__env-keycorridors6r3__seed-42" / "10-00-00"
            metadata_run.mkdir(parents=True)
            (metadata_run / "run_info.json").write_text(json.dumps({
                "mode": "discrete",
                "seed": 42,
                "env_name": "MiniGrid-KeyCorridorS6R3-v0",
                "source_env_name": "MiniGrid-KeyCorridorS6R3-v0",
                "task_family": "keycorridor",
                "use_corridor": False,
                "corridor_size": 11,
                "corridor_width": 3,
                "listener_reward_coef": 0.0,
                "intrinsic_anneal": False,
                "asymmetric_info": False,
                "use_sac": False,
                "use_option_critic": False,
                "condition_id": "mode-discrete__task-keycorridor__env-keycorridors6r3",
                "condition_label": "HRL Discrete (KeyCorridorS6R3)",
                "run_slug": "mode-discrete__task-keycorridor__env-keycorridors6r3__seed-42",
            }))

            legacy_run = root / "legacy" / "discrete_seed42" / "09-00-00"
            legacy_run.mkdir(parents=True)
            (legacy_run / "config.yaml").write_text(yaml.safe_dump({
                "env": {"name": "MiniGrid-KeyCorridorS6R3-v0"},
                "communication": {"listener_reward_coef": 0.0},
                "worker": {"intrinsic_anneal": False},
                "manager": {"use_option_critic": False},
                "sac": {"enabled": False},
            }))
            np.save(legacy_run / "returns.npy", np.array([0.0, 1.0]))

            records = discover_runs(root, allow_legacy=True)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["seed"], 42)
            self.assertEqual(records[0]["condition_id"], "mode-discrete__task-keycorridor__env-keycorridors6r3")

    def test_discover_runs_falls_back_to_legacy_outputs_when_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_run = root / "discrete_seed42" / "09-00-00"
            legacy_run.mkdir(parents=True)
            (legacy_run / "config.yaml").write_text(yaml.safe_dump({
                "env": {
                    "name": "MiniGrid-MultiRoom-N6-v0",
                    "corridor_size": 11,
                    "corridor_width": 3,
                },
                "communication": {"listener_reward_coef": 0.0},
                "worker": {"intrinsic_anneal": False},
                "manager": {"use_option_critic": False},
                "sac": {"enabled": False},
            }))
            np.save(legacy_run / "returns.npy", np.array([0.0, 1.0]))

            records = discover_runs(root, allow_legacy=True)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["mode"], "discrete")
            self.assertEqual(records[0]["seed"], 42)
            self.assertEqual(records[0]["task_family"], "multiroom")


if __name__ == "__main__":
    unittest.main()
