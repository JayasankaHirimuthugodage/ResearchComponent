"""Create a temporary, identifier-free development prediction bundle for TabICLv2 work.

This helper runs in the primary research environment because the isolated TabICLv2
environment intentionally does not contain XGBoost.  Its output is an intermediate
file, not a research report, and must be deleted after aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from training.contracts import TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, modeling_arrays
from training.models.common import build_candidate_pipeline, fit_candidate_pipeline
from training.models.logistic import REGULARIZED_LOGISTIC
from training.models.xgboost import REGULARIZED_XGBOOST


CV_SEED = 20260821
MODEL_SEED = 20260822
INNER_SPLITS = 3
SELECTIONS = {
    TARGET_COLUMNS[0]: REGULARIZED_XGBOOST,
    TARGET_COLUMNS[1]: REGULARIZED_LOGISTIC,
    TARGET_COLUMNS[2]: REGULARIZED_XGBOOST,
}


def _predict(candidate, train_x, train_y, validation_x, seed):
    pipeline = build_candidate_pipeline(candidate, seed=seed)
    fit_candidate_pipeline(pipeline, candidate, train_x, train_y)
    return pipeline.predict_proba(validation_x)[:, 1].astype(float)


def build_bundle(records) -> dict[str, object]:
    features, targets = modeling_arrays(records)
    folds = create_shared_folds(records, seed=CV_SEED)
    domains: dict[str, object] = {}
    for position, target in enumerate(TARGET_COLUMNS):
        candidate = SELECTIONS[target]
        rows = []
        for fold in folds:
            seed = MODEL_SEED + position * 1_000_000 + fold.repeat_index * 10_000 + fold.fold_index * 100
            train_x, train_y = features[fold.train_indices], targets[fold.train_indices, position]
            splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed + 1)
            inner = np.full(len(train_y), np.nan)
            for inner_index, (fit_indices, validation_indices) in enumerate(splitter.split(train_x, train_y)):
                inner[validation_indices] = _predict(
                    candidate, train_x[fit_indices], train_y[fit_indices],
                    train_x[validation_indices], seed + 10 + inner_index,
                )
            outer = _predict(candidate, train_x, train_y, features[fold.validation_indices], seed + 75)
            rows.append({
                "repeat": fold.repeat_index + 1,
                "fold": fold.fold_index + 1,
                "inner_raw_probabilities": inner.tolist(),
                "outer_raw_probabilities": outer.tolist(),
            })
        domains[target] = {
            "candidate": candidate.name,
            "configuration_version": candidate.version,
            "folds": rows,
        }
    return {
        "schema": "tabiclv2-hybrid-reference-temporary-v1",
        "development_participants": len(records),
        "shared_fold_plan_sha256": fold_plan_fingerprint(folds),
        "contains_participant_identifiers": False,
        "contains_test_predictions": False,
        "domains": domains,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = repository_root().resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Temporary output must be a new file inside the repository.")
    development = load_development_dataset()
    bundle = build_bundle(development.records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle), encoding="utf-8")
    print(json.dumps({"status": "complete", "domains": list(bundle["domains"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
