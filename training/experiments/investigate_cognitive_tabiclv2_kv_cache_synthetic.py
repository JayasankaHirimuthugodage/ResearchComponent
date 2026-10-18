"""Synthetic-only numerical investigation for the Cognitive TabICLv2 KV cache.

This module deliberately has no protected-context paths and never writes a cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS
from training.tabiclv2.candidate import tabiclv2_frame
from training.experiments.prepare_cognitive_tabiclv2_kv_cache import (
    CHECKPOINT,
    TOLERANCE,
    comparison,
    model,
    sigmoid,
)

FROZEN_SUITE_SIZE = 128
THRESHOLD = 0.1503614235173848
ADJACENCY_BAND = 0.01

# Eight predeclared answer profiles: low, high, alternating, and mixed patterns.
ANSWER_PROFILES = (
    ("0",) * 15,
    ("1",) * 15,
    ("2",) * 15,
    tuple(str(index % 3) for index in range(15)),
    tuple(str((index + 1) % 3) for index in range(15)),
    tuple(str((index + 2) % 3) for index in range(15)),
    ("2", "2", "1", "0", "0") * 3,
    ("0", "1", "2", "1", "0") * 3,
)


def frozen_rows() -> tuple[tuple[str, ...], ...]:
    """Return the predefined 128 valid 16-feature synthetic requests."""
    rows = [
        (checkpoint, *profile)
        for checkpoint in CHECKPOINT_IDS
        for profile in ANSWER_PROFILES
    ]
    # Fixed mixed profiles add 32 non-uniform requests, without data-dependent choice.
    for index in range(32):
        checkpoint = CHECKPOINT_IDS[index % len(CHECKPOINT_IDS)]
        profile = tuple(str((index * 2 + column * 5 + column // 3) % 3) for column in range(15))
        rows.append((checkpoint, *profile))
    assert len(rows) == FROZEN_SUITE_SIZE
    return tuple(rows)


def suite_sha256(rows: tuple[tuple[str, ...], ...] | None = None) -> str:
    value = rows if rows is not None else frozen_rows()
    canonical = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(canonical).hexdigest().upper()


FROZEN_SUITE_SHA256 = "78F302A0AEB5E2FF4809EA1316D69EE2187EABC748B41AB8EA9F5BF4636006E3"


def synthetic_training_data() -> tuple[np.ndarray, np.ndarray]:
    """A balanced, entirely synthetic context; no research records are used."""
    rows = frozen_rows()[:96]
    labels = np.asarray([int(sum(int(value) for value in row[1:]) >= 15) for row in rows], dtype=np.int8)
    return np.asarray(rows, dtype=object), labels


def threshold_distance(raw: np.ndarray) -> dict:
    calibrated = sigmoid(raw)
    distance = np.abs(calibrated - THRESHOLD)
    return {
        "minimum_calibrated_distance_to_threshold": float(distance.min()),
        "requests_within_predefined_0.01_band": int(np.count_nonzero(distance <= ADJACENCY_BAND)),
    }


def execute_synthetic() -> dict:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Verified local checkpoint is required for the synthetic-only investigation.")
    rows = frozen_rows()
    if suite_sha256(rows) != FROZEN_SUITE_SHA256:
        raise RuntimeError("Frozen synthetic verification suite hash mismatch.")
    train_rows, labels = synthetic_training_data()
    X_train = tabiclv2_frame(train_rows)
    X_test = tabiclv2_frame(np.asarray(rows, dtype=object))
    direct = model(False)
    direct.fit(X_train, labels)
    direct_first = direct.predict_proba(X_test)[:, 1]
    direct_second = direct.predict_proba(X_test)[:, 1]
    cached = model(True)
    cached.fit(X_train, labels)
    cached_first = cached.predict_proba(X_test)[:, 1]
    cached_second = cached.predict_proba(X_test)[:, 1]
    return {
        "schema": "synthetic-cognitive-tabiclv2-kv-cache-numerical-investigation-v1",
        "synthetic_only": True,
        "suite": {
            "requests": len(rows),
            "feature_count": len(FEATURE_COLUMNS),
            "checkpoint_ids": list(CHECKPOINT_IDS),
            "sha256": FROZEN_SUITE_SHA256,
            "predefined_adjacency_band": ADJACENCY_BAND,
        },
        "configuration": {
            "n_estimators": 8,
            "device": "cpu",
            "random_state": 20260822,
            "n_jobs": 4,
            "use_amp": False,
            "use_fa3": False,
            "offload_mode": False,
        },
        "comparisons": {
            "direct_repeat": comparison(direct_first, direct_second, THRESHOLD),
            "cache_repeat": comparison(cached_first, cached_second, THRESHOLD),
            "direct_vs_cache": comparison(direct_first, cached_first, THRESHOLD),
        },
        "threshold_proximity": {
            "direct": threshold_distance(direct_first),
            "cache": threshold_distance(cached_first),
        },
        "original_1e-8_tolerance": TOLERANCE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-synthetic", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_synthetic:
        print(json.dumps({"status": "blocked", "reason": "--execute-synthetic is required; no model was loaded."}))
        return 2
    print(json.dumps(execute_synthetic(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
