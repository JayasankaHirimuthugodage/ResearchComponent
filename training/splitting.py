"""Deterministic participant-level development/final-test allocation."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from training.contracts import (
    BASELINE_CHECKPOINT_COLUMN,
    DERIVED_TARGET_COLUMN,
    PARTICIPANT_COLUMN,
    TARGET_COLUMNS,
)


DEFAULT_SPLIT_SEED = 20260820
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_SEARCH_ITERATIONS = 25_000


@dataclass(frozen=True)
class ParticipantLabel:
    participant_number: str
    checkpoint: str
    targets: tuple[int, int, int]

    @property
    def combination(self) -> str:
        return "".join(str(value) for value in self.targets)


def _participant_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def participant_labels(
    records: Iterable[Mapping[str, str]],
) -> list[ParticipantLabel]:
    """Extract one aggregate-safe split label per participant."""

    labels: list[ParticipantLabel] = []
    seen: set[str] = set()
    for record in records:
        participant = record[PARTICIPANT_COLUMN].strip()
        if not participant:
            raise ValueError("Participant number is required for grouped splitting.")
        if participant in seen:
            raise ValueError(
                "The current split implementation expects one validated modelling row per participant."
            )
        seen.add(participant)
        labels.append(
            ParticipantLabel(
                participant_number=participant,
                checkpoint=record[BASELINE_CHECKPOINT_COLUMN].strip(),
                targets=tuple(int(record[target]) for target in TARGET_COLUMNS),
            )
        )
    return sorted(labels, key=lambda label: _participant_sort_key(label.participant_number))


def _objective(
    test_indices: Sequence[int],
    labels: Sequence[ParticipantLabel],
    target_totals: Sequence[int],
    checkpoint_totals: Counter[str],
    combination_totals: Counter[str],
) -> float:
    total = len(labels)
    test_size = len(test_indices)
    fraction = test_size / total
    test_targets = [0, 0, 0]
    test_checkpoints: Counter[str] = Counter()
    test_combinations: Counter[str] = Counter()
    for index in test_indices:
        label = labels[index]
        for position, value in enumerate(label.targets):
            test_targets[position] += value
        test_checkpoints[label.checkpoint] += 1
        test_combinations[label.combination] += 1

    score = 0.0
    for position, total_positive in enumerate(target_totals):
        expected = total_positive * fraction
        score += 5.0 * abs(test_targets[position] - expected) / max(expected, 1.0)
        if test_targets[position] == 0 or test_targets[position] == total_positive:
            score += 1_000.0

    for checkpoint, category_total in checkpoint_totals.items():
        expected = category_total * fraction
        observed = test_checkpoints[checkpoint]
        score += 0.5 * abs(observed - expected) / max(expected, 1.0)
        if observed == 0 or observed == category_total:
            score += 100.0

    for combination, category_total in combination_totals.items():
        expected = category_total * fraction
        observed = test_combinations[combination]
        score += abs(observed - expected) / max(expected, 1.0)
        if observed == 0 or observed == category_total:
            score += 250.0

    return score


def _partition_distribution(
    labels: Sequence[ParticipantLabel], participant_ids: set[str]
) -> dict[str, object]:
    selected = [label for label in labels if label.participant_number in participant_ids]
    count = len(selected)
    target_summary: dict[str, dict[str, float | int]] = {}
    for position, target in enumerate(TARGET_COLUMNS):
        positives = sum(label.targets[position] for label in selected)
        target_summary[target] = {
            "negative": count - positives,
            "positive": positives,
            "positive_rate": round(positives / count, 6) if count else 0.0,
        }
    overall_positives = sum(max(label.targets) for label in selected)
    target_summary[DERIVED_TARGET_COLUMN] = {
        "negative": count - overall_positives,
        "positive": overall_positives,
        "positive_rate": round(overall_positives / count, 6) if count else 0.0,
    }
    return {
        "participants": count,
        "targets": target_summary,
        "baseline_checkpoints": dict(
            sorted(Counter(label.checkpoint for label in selected).items())
        ),
        "label_combinations": dict(
            sorted(Counter(label.combination for label in selected).items())
        ),
    }


def create_participant_split(
    records: Iterable[Mapping[str, str]],
    *,
    dataset_sha256: str,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Create a reproducible split while balancing targets, checkpoints and combinations."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if search_iterations < 1:
        raise ValueError("search_iterations must be positive.")

    labels = participant_labels(records)
    participant_count = len(labels)
    if participant_count < 2:
        raise ValueError("At least two participants are required for splitting.")

    test_size = math.ceil(participant_count * test_fraction)
    all_indices = tuple(range(participant_count))
    target_totals = [
        sum(label.targets[position] for label in labels)
        for position in range(len(TARGET_COLUMNS))
    ]
    checkpoint_totals = Counter(label.checkpoint for label in labels)
    combination_totals = Counter(label.combination for label in labels)

    rng = random.Random(seed)
    best_indices: tuple[int, ...] | None = None
    best_score = math.inf
    for _ in range(search_iterations):
        candidate = tuple(sorted(rng.sample(all_indices, test_size)))
        score = _objective(
            candidate,
            labels,
            target_totals,
            checkpoint_totals,
            combination_totals,
        )
        if score < best_score:
            best_indices = candidate
            best_score = score

    if best_indices is None:
        raise RuntimeError("Split search did not produce a candidate allocation.")

    test_ids = {labels[index].participant_number for index in best_indices}
    development_ids = {
        label.participant_number for label in labels if label.participant_number not in test_ids
    }
    manifest = [
        {
            PARTICIPANT_COLUMN: label.participant_number,
            "partition": (
                "final_test" if label.participant_number in test_ids else "development"
            ),
        }
        for label in labels
    ]

    fingerprint_source = "\n".join(
        f"{row[PARTICIPANT_COLUMN]}:{row['partition']}" for row in manifest
    ).encode("utf-8")
    split_fingerprint = hashlib.sha256(fingerprint_source).hexdigest().upper()

    all_ids = {label.participant_number for label in labels}
    summary = {
        "dataset_sha256": dataset_sha256.upper(),
        "split_seed": seed,
        "requested_test_fraction": test_fraction,
        "search_iterations": search_iterations,
        "objective_score": round(best_score, 8),
        "split_fingerprint_sha256": split_fingerprint,
        "privacy_note": (
            "This aggregate summary contains no participant identifiers. "
            "The ignored local manifest is privacy-sensitive."
        ),
        "constraints": {
            "participant_overlap": len(test_ids.intersection(development_ids)),
            "all_participants_assigned_once": len(test_ids | development_ids)
            == participant_count,
            "all_target_classes_in_both_partitions": all(
                0 < sum(label.targets[position] for label in labels if label.participant_number in test_ids)
                < target_totals[position]
                for position in range(len(TARGET_COLUMNS))
            ),
            "all_checkpoints_in_both_partitions": all(
                0
                < sum(
                    label.checkpoint == checkpoint
                    for label in labels
                    if label.participant_number in test_ids
                )
                < total
                for checkpoint, total in checkpoint_totals.items()
            ),
            "all_label_combinations_in_both_partitions": all(
                0
                < sum(
                    label.combination == combination
                    for label in labels
                    if label.participant_number in test_ids
                )
                < total
                for combination, total in combination_totals.items()
            ),
        },
        "overall": _partition_distribution(labels, all_ids),
        "development": _partition_distribution(labels, development_ids),
        "final_test": _partition_distribution(labels, test_ids),
    }
    return manifest, summary

