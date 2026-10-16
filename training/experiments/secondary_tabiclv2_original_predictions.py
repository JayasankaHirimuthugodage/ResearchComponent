"""Isolated bridge for frozen original-model verification and in-memory comparison."""

from __future__ import annotations

import argparse
import json

from training.contracts import TARGET_COLUMNS, repository_root
from training.data.development_data import load_final_test_dataset
from training.experiments.run_locked_test_evaluation import _load_and_verify_artifacts
from training.models.common import modeling_arrays
from inference.contracts import DOMAIN_IDS


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--emit-aggregate-inputs", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    artifacts, release = _load_and_verify_artifacts(root)
    if args.preflight:
        print(json.dumps({
            "artifacts_verified": True,
            "release_manifest_sha256": release["release_manifest_sha256"],
            "test_outcomes_accessed": False,
        }))
        return 0

    final_test = load_final_test_dataset()
    features, targets = modeling_arrays(final_test.records)
    domains = {}
    for position, (target, domain) in enumerate(zip(TARGET_COLUMNS, DOMAIN_IDS, strict=True)):
        artifact = artifacts[domain]
        domains[target] = {
            "model_configuration_version": artifact.model_configuration_version,
            "threshold": artifact.threshold,
            "targets": targets[:, position].astype(int).tolist(),
            "probabilities": [
                float(artifact.positive_probability(tuple(row))) for row in features
            ],
        }
    print(json.dumps({
        "participants": len(final_test.records),
        "dataset_sha256": final_test.dataset_sha256,
        "split_fingerprint_sha256": final_test.split_fingerprint_sha256,
        "domains": domains,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
