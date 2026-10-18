from __future__ import annotations

import unittest

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS
from training.experiments.investigate_cognitive_tabiclv2_kv_cache_synthetic import (
    FROZEN_SUITE_SHA256,
    FROZEN_SUITE_SIZE,
    frozen_rows,
    suite_sha256,
    synthetic_training_data,
)


class SyntheticKVCacheInvestigationTests(unittest.TestCase):
    def test_frozen_suite_has_expected_shape_and_hash(self):
        rows = frozen_rows()
        self.assertEqual(FROZEN_SUITE_SIZE, len(rows))
        self.assertEqual(FROZEN_SUITE_SHA256, suite_sha256(rows))
        self.assertTrue(all(len(row) == len(FEATURE_COLUMNS) for row in rows))

    def test_every_valid_checkpoint_is_covered(self):
        rows = frozen_rows()
        self.assertEqual(set(CHECKPOINT_IDS), {row[0] for row in rows})

    def test_answers_and_synthetic_training_context_are_valid(self):
        rows = frozen_rows()
        self.assertTrue(all(value in {"0", "1", "2"} for row in rows for value in row[1:]))
        training_rows, labels = synthetic_training_data()
        self.assertEqual((96, 16), training_rows.shape)
        self.assertEqual(96, labels.size)
        self.assertEqual({0, 1}, set(labels.tolist()))
