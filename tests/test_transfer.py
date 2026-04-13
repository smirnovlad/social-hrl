import json
import tempfile
import unittest
from pathlib import Path

from transfer_utils import discover_source_runs, validate_transfer_request


class TransferDiscoveryTests(unittest.TestCase):
    def _write_run(self, root, seed, mode, task_family, source_env_name, eval_success_rate):
        run_dir = root / f"{mode}-{seed}" / "10-00-00"
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text(json.dumps({
            "mode": mode,
            "seed": seed,
            "env_name": source_env_name,
            "source_env_name": source_env_name,
            "task_family": task_family,
            "use_corridor": False,
            "corridor_size": 11,
            "corridor_width": 3,
            "listener_reward_coef": 0.0,
            "intrinsic_anneal": False,
            "asymmetric_info": False,
            "use_sac": False,
            "use_option_critic": False,
            "condition_id": f"{mode}-{task_family}",
            "condition_label": f"{mode} {task_family}",
            "run_slug": f"{mode}-{task_family}__seed-{seed}",
        }))
        (run_dir / "metrics.json").write_text(json.dumps({
            "eval_success_rate": eval_success_rate,
        }))
        (run_dir / "final.pt").write_bytes(b"checkpoint")
        return run_dir

    def test_discover_source_runs_filters_by_family_env_and_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_run(
                root,
                seed=42,
                mode="discrete",
                task_family="keycorridor",
                source_env_name="MiniGrid-KeyCorridorS3R2-v0",
                eval_success_rate=0.8,
            )
            self._write_run(
                root,
                seed=7,
                mode="discrete",
                task_family="keycorridor",
                source_env_name="MiniGrid-KeyCorridorS3R2-v0",
                eval_success_rate=0.2,
            )
            self._write_run(
                root,
                seed=123,
                mode="discrete",
                task_family="multiroom",
                source_env_name="MiniGrid-MultiRoom-N6-v0",
                eval_success_rate=0.9,
            )

            selected, skipped = discover_source_runs(
                root,
                "discrete",
                source_task_family="keycorridor",
                source_env="MiniGrid-KeyCorridorS3R2-v0",
                min_source_success=0.5,
            )

            self.assertEqual(sorted(selected.keys()), [42])
            self.assertEqual(selected[42]["task_family"], "keycorridor")
            self.assertEqual(len(skipped), 1)
            self.assertIn("below_threshold", skipped[0]["reason"])

    def test_validate_transfer_request_blocks_social_sources(self):
        with self.assertRaises(ValueError):
            validate_transfer_request("social")


if __name__ == "__main__":
    unittest.main()
