"""One isolated, development-only LightGBM comparison across three domains."""
from __future__ import annotations

import argparse, json, os, platform, time
from pathlib import Path

import lightgbm
import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.development_decision_report import sha256_file
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, run_tabiclv2_evaluation

REPORT = Path("reports/generated/development-lightgbm-comparison-report.json")
CONFIG = {
    "objective": "binary", "n_estimators": 100, "learning_rate": 0.05,
    "max_depth": 3, "num_leaves": 7, "min_child_samples": 20,
    "reg_lambda": 1.0, "n_jobs": 2, "random_state": 42, "verbosity": -1,
}
NAME = "lightgbm_fixed_small_tree_onehot_checkpoint_v1"


def predict(train_x, train_y, validation_x, seed):
    """Fit only on the supplied fold; seed is intentionally fixed by approved config."""
    del seed
    preprocessor = ColumnTransformer((
        ("checkpoint", OneHotEncoder(categories=[list(CHECKPOINT_IDS)], drop="first",
         handle_unknown="ignore", sparse_output=False), [0]),
        ("questionnaire", "passthrough", list(range(1, len(FEATURE_COLUMNS)))),
    ), remainder="drop")
    model = LGBMClassifier(**CONFIG)
    return Pipeline((("preprocess", preprocessor), ("estimator", model))).fit(
        train_x, train_y
    ).predict_proba(validation_x)[:, 1].astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    output = root / REPORT
    if output.exists():
        raise FileExistsError("Refusing to overwrite completed report")
    development = load_development_dataset()
    historical_path = root / "reports/generated/development-additional-model-comparison-report.json"
    historical = json.loads(historical_path.read_text())
    folds = create_shared_folds(development.records, seed=20260821)
    if len(development.records) != 500 or len(FEATURE_COLUMNS) != 16:
        raise ValueError("Development participant or frozen feature contract mismatch")
    if fold_plan_fingerprint(folds) != historical["cross_validation"]["shared_fold_plan_sha256"]:
        raise ValueError("Development provenance or fold fingerprint mismatch")
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_passed", "development_participants": len(development.records),
            "feature_count": len(FEATURE_COLUMNS), "targets": 3, "outer_folds": 75,
            "configuration": CONFIG, "test_outcomes_accessed": False,
        }, indent=2))
        return
    started = time.perf_counter()

    def progress(event):
        if event["event"] == "fold_completed":
            print("[{}/75] {} repeat={} fold={} elapsed={:.1f}s ETA={:.1f}s".format(
                event["completed_outer_folds"], event["domain"], event["repeat"], event["fold"],
                event["elapsed_seconds"], event["estimated_remaining_seconds"]), flush=True)

    result = run_tabiclv2_evaluation(
        development.records, fit_predict=predict, historical_report=historical,
        progress_callback=progress,
    )
    result["artifact_schema_version"] = "lightgbm-development-comparison-v1"
    result["status"] = "completed_development_only_lightgbm_evaluation"
    result["scope"] = "development_only_exploratory_lightgbm_evaluation"
    result["candidate_names"] = [NAME]
    result["lightgbm_configuration"] = CONFIG
    result["lightgbm_version"] = lightgbm.__version__
    result["class_imbalance_note"] = "No class weighting was added: the pre-authorized fixed configuration was used unchanged."
    result["early_stopping_leakage_note"] = "No early stopping or validation-set-based iteration selection was used."
    result["test_cohort_loaded"] = False
    result["runtime_seconds"] = time.perf_counter() - started
    result["historical_additional_report_sha256"] = sha256_file(historical_path)
    result["runtime_versions"] = {"python": platform.python_version(), "scikit_learn": sklearn.__version__}
    for domain in result["domains"].values():
        domain["candidate"] = NAME
        for ranking in domain["eight_model_pr_auc_ranking"]:
            if ranking["candidate"] == "tabiclv2_default_eight_view":
                ranking["candidate"] = NAME
                ranking["source"] = "current_lightgbm_evaluation"
        comparison = domain["numerical_comparison_to_historical_selected_model"]
        for values in comparison.values():
            values["lightgbm_mean"] = values.pop("tabiclv2_mean")
            values["lightgbm_minus_reference"] = values.pop("tabiclv2_minus_reference")
    temporary = output.with_name("." + output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(result, indent=2) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"status": result["status"], "outer_folds": 75,
                      "runtime_seconds": result["runtime_seconds"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
