from __future__ import annotations

import json
import unittest

import numpy as np

from training.contracts import (
    EXTRA_TREES_MODEL_CONFIGURATION_VERSION,
    EXPERIMENTAL_DECISION_THRESHOLD_STATUS,
    DECISION_THRESHOLD_APPROVAL_STATEMENT,
    DECISION_THRESHOLD_APPROVAL_STATUS,
    DECISION_THRESHOLD_VERSION,
    BASELINE_CHECKPOINT_COLUMN,
    CHECKPOINT_IDS,
    PARTICIPANT_COLUMN,
    QUESTION_COLUMNS,
    TARGET_COLUMNS,
)
from training.data.development_data import ModelingRecord, select_development_records
from training.experiments.cross_validation import create_shared_folds
from training.experiments.evaluation import (
    binary_probability_metrics,
    decision_threshold_metadata,
    run_development_cross_validation,
)
from training.models.common import modeling_arrays
from training.models.extra_trees import REGULARIZED_EXTRA_TREES
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.registry import DEFAULT_CANDIDATES


def synthetic_modeling_records(
    records_per_combination: int = 2,
) -> tuple[ModelingRecord, ...]:
    records: list[ModelingRecord] = []
    combinations = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )
    for combination_index, combination in enumerate(combinations):
        for repetition in range(records_per_combination):
            position = combination_index * records_per_combination + repetition
            questionnaire = tuple(
                str((position + question_index) % 3)
                for question_index in range(len(QUESTION_COLUMNS))
            )
            records.append(
                ModelingRecord(
                    participant_key=f"synthetic-{position:03d}",
                    feature_values=(
                        CHECKPOINT_IDS[position % len(CHECKPOINT_IDS)],
                        *questionnaire,
                    ),
                    target_values=combination,
                )
            )
    return tuple(records)


class DevelopmentModelingContractTests(unittest.TestCase):
    def test_selection_projects_allowlisted_inputs_and_excludes_final_test(self) -> None:
        source_records: list[dict[str, str]] = []
        manifest: list[dict[str, str]] = []
        for index, partition in enumerate(("development", "final_test")):
            participant = f"private-fixture-{index}"
            row = {
                PARTICIPANT_COLUMN: participant,
                BASELINE_CHECKPOINT_COLUMN: CHECKPOINT_IDS[index],
                **{column: str((index + offset) % 3) for offset, column in enumerate(QUESTION_COLUMNS)},
                **{column: str(index) for column in TARGET_COLUMNS},
                "followup_communication_score": "PRIVATE_FUTURE_SENTINEL",
            }
            source_records.append(row)
            manifest.append(
                {PARTICIPANT_COLUMN: participant, "partition": partition}
            )

        selected = select_development_records(source_records, manifest)

        self.assertEqual(1, len(selected))
        self.assertEqual("private-fixture-0", selected[0].participant_key)
        self.assertEqual(16, len(selected[0].feature_values))
        self.assertNotIn("PRIVATE_FUTURE_SENTINEL", selected[0].feature_values)

    def test_shared_folds_are_deterministic_disjoint_and_cover_each_repeat(self) -> None:
        records = synthetic_modeling_records(records_per_combination=3)
        first = create_shared_folds(records, n_splits=3, n_repeats=2, seed=17)
        second = create_shared_folds(records, n_splits=3, n_repeats=2, seed=17)

        self.assertEqual(6, len(first))
        for left, right in zip(first, second, strict=True):
            self.assertTrue(np.array_equal(left.train_indices, right.train_indices))
            self.assertTrue(
                np.array_equal(left.validation_indices, right.validation_indices)
            )
            self.assertFalse(
                set(left.train_indices.tolist()).intersection(
                    left.validation_indices.tolist()
                )
            )
        for repeat in range(2):
            validation_indices = np.concatenate(
                [fold.validation_indices for fold in first if fold.repeat_index == repeat]
            )
            self.assertEqual(list(range(len(records))), sorted(validation_indices.tolist()))

    def test_probability_half_cutoff_is_separate_from_observed_target(self) -> None:
        metrics = binary_probability_metrics(
            np.asarray([1, 0], dtype=np.int8),
            np.asarray([0.50, 0.499], dtype=float),
        )

        self.assertEqual(1, metrics["confusion_matrix"]["true_positive"])
        self.assertEqual(1, metrics["confusion_matrix"]["true_negative"])

    def test_direct_modeling_helper_rejects_an_unscorable_response(self) -> None:
        records = list(synthetic_modeling_records())
        first = records[0]
        records[0] = ModelingRecord(
            participant_key=first.participant_key,
            feature_values=(first.feature_values[0], "NOT_SURE", *first.feature_values[2:]),
            target_values=first.target_values,
        )

        with self.assertRaisesRegex(ValueError, "scoreable"):
            modeling_arrays(records)

    def test_different_probability_cutoff_does_not_inherit_approval(self) -> None:
        metadata = decision_threshold_metadata(
            0.40,
            model_configuration_version=REGULARIZED_LOGISTIC.version,
        )

        self.assertEqual("unapproved_experimental_override", metadata["status"])
        self.assertIsNone(metadata["version"])
        self.assertIsNone(metadata["approval_statement"])

    def test_same_probability_cutoff_is_experimental_for_extra_trees(self) -> None:
        metadata = decision_threshold_metadata(
            0.50,
            model_configuration_version=REGULARIZED_EXTRA_TREES.version,
        )

        self.assertEqual(EXPERIMENTAL_DECISION_THRESHOLD_STATUS, metadata["status"])
        self.assertIsNone(metadata["version"])
        self.assertIsNone(metadata["approval_statement"])

    def test_two_candidate_experiment_emits_aggregate_explanations(self) -> None:
        records = synthetic_modeling_records(records_per_combination=2)

        report = run_development_cross_validation(
            records,
            candidates=DEFAULT_CANDIDATES,
            n_splits=2,
            n_repeats=1,
            cv_seed=23,
            model_seed=29,
        )

        self.assertEqual("development_only", report["scope"])
        self.assertTrue(
            report["cross_validation"]["same_folds_for_all_domains_and_candidates"]
        )
        self.assertEqual(0.50, report["comparison_threshold"]["value"])
        self.assertEqual(
            {REGULARIZED_LOGISTIC.name, REGULARIZED_EXTRA_TREES.name},
            set(report["candidates"]),
        )
        candidate = report["candidates"][REGULARIZED_LOGISTIC.name]
        self.assertEqual(REGULARIZED_LOGISTIC.version, candidate["model_configuration_version"])
        self.assertEqual(
            DECISION_THRESHOLD_APPROVAL_STATUS,
            candidate["decision_threshold"]["status"],
        )
        self.assertEqual(
            DECISION_THRESHOLD_VERSION,
            candidate["decision_threshold"]["version"],
        )
        self.assertEqual(
            DECISION_THRESHOLD_APPROVAL_STATEMENT,
            candidate["decision_threshold"]["approval_statement"],
        )
        self.assertEqual(3, candidate["models_per_fold"])
        self.assertEqual(set(TARGET_COLUMNS), set(candidate["domains"]))
        for target in TARGET_COLUMNS:
            domain = candidate["domains"][target]
            self.assertEqual(2, len(domain["fold_results"]))
            self.assertEqual(
                len(CHECKPOINT_IDS) - 1 + len(QUESTION_COLUMNS),
                len(domain["explanation"]["values"]),
            )
            self.assertEqual(
                "standardized_logistic_coefficient_stability",
                domain["explanation"]["method"],
            )
            self.assertIn("brier_score", domain["averaged_participant_oof_metrics"])

        extra_trees = report["candidates"][REGULARIZED_EXTRA_TREES.name]
        self.assertEqual(
            EXTRA_TREES_MODEL_CONFIGURATION_VERSION,
            extra_trees["model_configuration_version"],
        )
        self.assertEqual(
            EXPERIMENTAL_DECISION_THRESHOLD_STATUS,
            extra_trees["decision_threshold"]["status"],
        )
        self.assertIsNone(extra_trees["decision_threshold"]["approval_statement"])
        for target in TARGET_COLUMNS:
            explanation = extra_trees["domains"][target]["explanation"]
            self.assertEqual(
                "held_out_permutation_importance_average_precision",
                explanation["method"],
            )
            self.assertEqual("average_precision", explanation["scoring"])
            self.assertEqual(len(QUESTION_COLUMNS) + 1, len(explanation["values"]))

        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("synthetic-", serialized)
        self.assertNotIn("participant_key", serialized)
        self.assertEqual(
            "OR of the three domain decisions; no fourth model",
            candidate["derived_any_concern"]["definition"],
        )
        self.assertEqual(
            "OR of the three domain decisions; no fourth model",
            extra_trees["derived_any_concern"]["definition"],
        )


if __name__ == "__main__":
    unittest.main()
