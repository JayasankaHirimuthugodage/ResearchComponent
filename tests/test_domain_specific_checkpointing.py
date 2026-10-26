"""Synthetic-only tests for resumable domain-specific training checkpoints."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.experiments.domain_specific_checkpointing import (
    CheckpointError,
    atomic_json,
    fingerprint,
    load_json,
    load_result,
    validate_provenance,
    write_result,
)


class DomainSpecificCheckpointingTests(unittest.TestCase):
    def test_atomic_json_round_trip_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            digest = atomic_json(path, {"schema": "synthetic", "seed": 7})
            self.assertEqual({"schema": "synthetic", "seed": 7}, load_json(path, expected_sha256=digest))
            with self.assertRaises(FileExistsError):
                atomic_json(path, {"schema": "replacement"})

    def test_fold_round_trip_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value) / "individual" / "logistic" / "communication" / "repeat-00-fold-00"
            metadata = {"candidate": "logistic", "domain": "communication", "repeat": 0,
                        "fold": 0, "fold_plan_sha256": "SYNTHETIC"}
            write_result(directory, np.array([0.2, 0.8]), metadata)
            loaded = load_result(directory, {**metadata})
            np.testing.assert_allclose(loaded, [0.2, 0.8])
            self.assertIsNone(load_result(directory.parent / "repeat-00-fold-01", metadata))

    def test_corruption_and_partial_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value) / "fold"
            metadata = {"candidate": "x", "domain": "cognitive", "repeat": 0, "fold": 0}
            write_result(directory, np.array([0.4]), metadata)
            (directory / "result.npy").write_bytes(b"corrupt")
            with self.assertRaisesRegex(CheckpointError, "checksum"):
                load_result(directory, metadata)
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value) / "partial"
            directory.mkdir()
            (directory / "result.npy").write_bytes(b"orphan")
            with self.assertRaisesRegex(CheckpointError, "Incomplete"):
                load_result(directory, {})

    def test_provenance_mismatch_identifies_changed_contract(self):
        base = {"dataset_sha256": "A", "split_sha256": "B", "features": ["checkpoint", "q1"]}
        validate_provenance(base, dict(base))
        changed = {**base, "features": ["checkpoint", "q1", "q2"]}
        with self.assertRaisesRegex(CheckpointError, "features"):
            validate_provenance(changed, base)
        self.assertNotEqual(fingerprint(changed), fingerprint(base))

    def test_privacy_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            with self.assertRaisesRegex(CheckpointError, "Prohibited"):
                atomic_json(Path(value) / "bad.json", {"participant_number": ["P001"]})
            with self.assertRaisesRegex(CheckpointError, "Prohibited"):
                write_result(Path(value) / "bad-fold", np.array([0.5]), {"locked_test": True})

    def test_hybrid_and_individual_namespaces_are_independent(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            metadata = {"domain": "social_emotional", "repeat": 1, "fold": 2}
            write_result(root / "individual" / "candidate" / "fold", np.array([0.1]),
                         {**metadata, "candidate": "candidate"})
            write_result(root / "hybrid" / "fold", np.array([0.9]),
                         {**metadata, "candidate": "hybrid"})
            self.assertAlmostEqual(0.1, load_result(root / "individual" / "candidate" / "fold",
                                                   {**metadata, "candidate": "candidate"})[0])
            self.assertAlmostEqual(0.9, load_result(root / "hybrid" / "fold",
                                                   {**metadata, "candidate": "hybrid"})[0])


if __name__ == "__main__":
    unittest.main()
