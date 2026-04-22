import unittest

import numpy as np

from analysis.goal_metrics import compute_all_metrics, goal_coverage, message_novelty


class GoalMetricTests(unittest.TestCase):
    def test_goal_coverage_measures_fraction_of_vocabulary(self):
        messages = [
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
        ]

        self.assertAlmostEqual(goal_coverage(messages, vocab_size=10, message_length=3), 2 / 1000)

    def test_message_novelty_measures_unique_fraction_of_sample(self):
        messages = [
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
        ]

        self.assertAlmostEqual(message_novelty(messages), 0.5)

    def test_compute_all_metrics_reports_both_coverage_and_novelty(self):
        messages = [
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
        ]

        metrics = compute_all_metrics(messages=messages, vocab_size=10, message_length=3)

        self.assertAlmostEqual(metrics["coverage"], 2 / 1000)
        self.assertAlmostEqual(metrics["message_novelty"], 0.5)


if __name__ == "__main__":
    unittest.main()
