import numpy as np

from training.experiments.run_tabiclv2_sensitivity_threshold_study import select_sensitivity_constrained_threshold


def test_constrained_threshold_meets_fixed_target_and_prefers_specificity():
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    p = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    result = select_sensitivity_constrained_threshold(y, p, target=2 / 3)
    assert result["metrics"]["sensitivity"] >= 2 / 3
    assert result["metrics"]["specificity"] == 1.0


def test_constrained_threshold_rejects_invalid_target():
    y = np.array([0, 1], dtype=np.int8); p = np.array([0.1, 0.9])
    try:
        select_sensitivity_constrained_threshold(y, p, target=0.0)
    except ValueError:
        return
    raise AssertionError("Expected invalid target to raise ValueError")
