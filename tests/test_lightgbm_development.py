from __future__ import annotations

import inspect
import unittest

from training.experiments.run_lightgbm_development import CONFIG, NAME


class LightGBMDevelopmentTests(unittest.TestCase):
    def test_fixed_authorized_configuration(self):
        self.assertEqual(CONFIG, {
            "objective": "binary", "n_estimators": 100, "learning_rate": 0.05,
            "max_depth": 3, "num_leaves": 7, "min_child_samples": 20,
            "reg_lambda": 1.0, "n_jobs": 2, "random_state": 42, "verbosity": -1,
        })

    def test_runner_is_development_only(self):
        module = __import__("training.experiments.run_lightgbm_development", fromlist=["x"])
        source = inspect.getsource(module)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertIn(NAME, source)


if __name__ == "__main__":
    unittest.main()
