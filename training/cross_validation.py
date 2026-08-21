"""Deterministic shared development folds for all domains and candidates."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from training.development_data import ModelingRecord


DEFAULT_CV_SEED = 20260821
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
DEFAULT_FOLD_SEARCH_ITERATIONS = 250


@dataclass(frozen=True)
class SharedFold:
    repeat_index: int
    fold_index: int
    allocation_seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


def target_combination(record: ModelingRecord) -> str:
    return "".join(str(value) for value in record.target_values)


def _checkpoint_balance_objective(
    validation_folds: Sequence[np.ndarray], checkpoints: Sequence[str]
) -> float:
    checkpoint_totals = Counter(checkpoints)
    n_splits = len(validation_folds)
    score = 0.0
    for validation_indices in validation_folds:
        observed = Counter(checkpoints[index] for index in validation_indices)
        for checkpoint, total in checkpoint_totals.items():
            expected = total / n_splits
            count = observed[checkpoint]
            score += abs(count - expected) / max(expected, 1.0)
            if count == 0:
                score += 100.0
    return score


def _best_repeat_split(
    combinations: Sequence[str],
    checkpoints: Sequence[str],
    *,
    n_splits: int,
    base_seed: int,
    search_iterations: int,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    placeholder = np.zeros(len(combinations), dtype=np.int8)
    best_seed: int | None = None
    best_score = math.inf
    best_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    for offset in range(search_iterations):
        candidate_seed = base_seed + offset
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=candidate_seed,
        )
        candidate_splits = list(splitter.split(placeholder, combinations))
        score = _checkpoint_balance_objective(
            [validation for _, validation in candidate_splits], checkpoints
        )
        if score < best_score:
            best_seed = candidate_seed
            best_score = score
            best_splits = candidate_splits
    if best_seed is None or best_splits is None:
        raise RuntimeError("Shared-fold search did not produce an allocation.")
    return best_seed, best_splits


def create_shared_folds(
    records: Sequence[ModelingRecord],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = DEFAULT_CV_SEED,
    search_iterations: int = DEFAULT_FOLD_SEARCH_ITERATIONS,
) -> tuple[SharedFold, ...]:
    """Create participant-safe folds balancing labels first and checkpoints second."""

    if n_splits < 2 or n_repeats < 1 or search_iterations < 1:
        raise ValueError(
            "Cross-validation requires at least two folds, one repeat and one search iteration."
        )
    participant_keys = [record.participant_key for record in records]
    if len(participant_keys) != len(set(participant_keys)):
        raise ValueError("Shared folds require one modelling row per participant.")

    combinations = [target_combination(record) for record in records]
    checkpoints = [record.feature_values[0] for record in records]
    combination_counts = Counter(combinations)
    if not combination_counts or min(combination_counts.values()) < n_splits:
        raise ValueError(
            "Every observed three-label combination must have at least one record per fold."
        )

    targets = np.asarray([record.target_values for record in records], dtype=np.int8)
    folds: list[SharedFold] = []
    validation_coverage = np.zeros((n_repeats, len(records)), dtype=np.int8)

    for repeat_index in range(n_repeats):
        base_seed = seed + repeat_index * search_iterations
        allocation_seed, repeat_splits = _best_repeat_split(
            combinations,
            checkpoints,
            n_splits=n_splits,
            base_seed=base_seed,
            search_iterations=search_iterations,
        )
        for fold_index, (train_indices, validation_indices) in enumerate(repeat_splits):
            if np.intersect1d(train_indices, validation_indices).size:
                raise RuntimeError("A participant appears in both sides of a shared fold.")
            for target_position in range(targets.shape[1]):
                if np.unique(targets[train_indices, target_position]).size != 2:
                    raise ValueError("A training fold is missing a class for a domain target.")
                if np.unique(targets[validation_indices, target_position]).size != 2:
                    raise ValueError("A validation fold is missing a class for a domain target.")
            validation_coverage[repeat_index, validation_indices] += 1
            folds.append(
                SharedFold(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    allocation_seed=allocation_seed,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                )
            )

    if len(folds) != n_splits * n_repeats:
        raise RuntimeError("Cross-validation did not produce the requested fold count.")
    if not np.all(validation_coverage == 1):
        raise RuntimeError("Each participant must be validated exactly once per repeat.")
    return tuple(folds)
