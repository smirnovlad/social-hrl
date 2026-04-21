import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[1]
PLOT_RESULTS_PATH = ROOT / "scripts" / "plot_results.py"
PLOT_RESULTS_SPEC = importlib.util.spec_from_file_location("plot_results", PLOT_RESULTS_PATH)
PLOT_RESULTS_MODULE = importlib.util.module_from_spec(PLOT_RESULTS_SPEC)
PLOT_RESULTS_SPEC.loader.exec_module(PLOT_RESULTS_MODULE)

find_latest_runs = PLOT_RESULTS_MODULE.find_latest_runs


class PlotDiscoveryTests(unittest.TestCase):
    def test_find_latest_runs_uses_latest_suite_run_per_condition_and_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            older = run_root / "condition-a" / "09-00-00"
            newer = run_root / "condition-a" / "10-00-00"

            for run_dir, reward in ((older, -1.0), (newer, 1.0)):
                run_dir.mkdir(parents=True)
                (run_dir / "run_info.json").write_text(json.dumps({
                    "mode": "discrete",
                    "seed": 42,
                    "env_name": "MiniGrid-KeyCorridorS3R2-v0",
                    "source_env_name": "MiniGrid-KeyCorridorS3R2-v0",
                    "task_family": "keycorridor",
                    "use_corridor": False,
                    "corridor_size": 11,
                    "corridor_width": 3,
                    "listener_reward_coef": 0.0,
                    "intrinsic_anneal": False,
                    "asymmetric_info": False,
                    "use_sac": False,
                    "use_option_critic": False,
                    "condition_id": "condition-a",
                    "condition_label": "Condition A",
                    "run_slug": "condition-a__seed-42",
                }))
                np.save(run_dir / "returns.npy", np.array([reward]))

            latest = find_latest_runs(run_root)

            self.assertEqual(len(latest), 1)
            record = list(latest.values())[0]
            self.assertTrue(record["run_dir"].endswith("10-00-00"))


if __name__ == "__main__":
    unittest.main()
