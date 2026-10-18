from __future__ import annotations

import inspect
import unittest

from training.experiments.run_figs_development import CONFIG, NAME


class FIGSDevelopmentTests(unittest.TestCase):
    def test_fixed_authorized_configuration(self):
        self.assertEqual(CONFIG, {"max_rules": 8, "max_trees": 3, "max_depth": 3,
                                  "min_impurity_decrease": 0.0, "class_weight": "balanced",
                                  "random_state": 42, "n_jobs": 2, "verbose": 0})

    def test_runner_is_development_only(self):
        module = __import__("training.experiments.run_figs_development", fromlist=["x"])
        self.assertNotIn("load_final_test_dataset", inspect.getsource(module))
        self.assertIn(NAME, inspect.getsource(module))


if __name__ == "__main__":
    unittest.main()
