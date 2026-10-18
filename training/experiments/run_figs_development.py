"""Fixed-complexity FIGS comparison on the development cohort only."""
from __future__ import annotations

import argparse, json, os, platform, time
from importlib.metadata import version
from pathlib import Path

import imodels, numpy as np, pandas as pd, sklearn
from imodels import FIGSClassifier

from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.development_decision_report import sha256_file
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, run_tabiclv2_evaluation

REPORT = Path("reports/generated/development-figs-comparison-report.json")
TABIC_REPORT = Path("reports/generated/development-tabiclv2-model-comparison-report.json")
CONFIG = {"max_rules": 8, "max_trees": 3, "max_depth": 3, "min_impurity_decrease": 0.0,
          "class_weight": "balanced", "random_state": 42, "n_jobs": 2, "verbose": 0}
NAME = "figs_balanced_rule_budget8_tree_budget3_depth3_v1"
OUTER_COMPLEXITIES: list[dict[str, int]] = []


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def predict(train_x, train_y, validation_x, seed):
    """Fold-local categorical processing and fixed-complexity FIGS fitting."""
    del seed
    model = FIGSClassifier(**CONFIG)
    model.fit(_frame(train_x), train_y, categorical_features=[FEATURE_COLUMNS[0]])
    probabilities = model.predict_proba(_frame(validation_x))[:, 1].astype(float)
    if not (np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))):
        raise RuntimeError("FIGS produced invalid probabilities")
    # The shared evaluator calls this once per outer fold on its 100-person validation set.
    if len(validation_x) == 100:
        OUTER_COMPLEXITIES.append({"rule_count": int(len(model.get_rules())),
                                   "tree_count": int(len(model.trees_))})
    return probabilities


def _complexity_summary(rows: list[dict[str, int]]) -> dict[str, object]:
    if len(rows) != 25:
        raise RuntimeError("Expected one FIGS complexity record per outer fold.")
    return {"outer_fitted_models": len(rows),
            "rule_count": {"mean": float(np.mean([r["rule_count"] for r in rows])),
                           "minimum": int(min(r["rule_count"] for r in rows)),
                           "maximum": int(max(r["rule_count"] for r in rows))},
            "tree_count": {"mean": float(np.mean([r["tree_count"] for r in rows])),
                           "minimum": int(min(r["tree_count"] for r in rows)),
                           "maximum": int(max(r["tree_count"] for r in rows))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    output = root / REPORT
    if output.exists():
        raise FileExistsError("Refusing to overwrite completed FIGS report")
    development = load_development_dataset()
    historical_path = root / "reports/generated/development-additional-model-comparison-report.json"
    historical = json.loads(historical_path.read_text())
    tabic_path = root / TABIC_REPORT
    tabic = json.loads(tabic_path.read_text())
    folds = create_shared_folds(development.records, seed=20260821)
    fingerprint = fold_plan_fingerprint(folds)
    if len(development.records) != 500 or len(FEATURE_COLUMNS) != 16 or len(TARGET_COLUMNS) != 3:
        raise ValueError("Frozen development contract mismatch")
    if fingerprint != historical["cross_validation"]["shared_fold_plan_sha256"]:
        raise ValueError("Historical fold fingerprint mismatch")
    if fingerprint != tabic["cross_validation"]["shared_fold_plan_sha256"]:
        raise ValueError("TabICLv2 fold fingerprint mismatch")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "development_participants": 500,
                          "feature_count": 16, "targets": list(TARGET_COLUMNS), "outer_folds": 75,
                          "configuration": CONFIG, "test_outcomes_accessed": False}, indent=2))
        return
    OUTER_COMPLEXITIES.clear()
    started = time.perf_counter()
    def progress(event):
        if event["event"] == "fold_completed":
            print("[{}/75] {} repeat={} fold={} elapsed={:.1f}s ETA={:.1f}s".format(
                event["completed_outer_folds"], event["domain"], event["repeat"], event["fold"],
                event["elapsed_seconds"], event["estimated_remaining_seconds"]), flush=True)
    result = run_tabiclv2_evaluation(development.records, fit_predict=predict,
                                     historical_report=historical, progress_callback=progress)
    if len(OUTER_COMPLEXITIES) != 75:
        raise RuntimeError("FIGS complexity capture is incomplete")
    result.update({"artifact_schema_version": "figs-development-comparison-v1",
                   "status": "completed_development_only_figs_evaluation",
                   "scope": "development_only_exploratory_figs_evaluation",
                   "candidate_names": [NAME], "figs_configuration": CONFIG,
                   "imodels_version": version("imodels"), "test_cohort_loaded": False,
                   "runtime_seconds": time.perf_counter() - started,
                   "historical_additional_report_sha256": sha256_file(historical_path),
                   "tabiclv2_report_sha256": sha256_file(tabic_path),
                   "runtime_versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
                   "model_complexity_interpretation": "Rule and tree counts describe the final outer-fitted FIGS models; no learned rules or participant-level data are written."})
    for index, (domain_name, domain) in enumerate(result["domains"].items()):
        domain["candidate"] = NAME
        domain["outer_model_complexity"] = _complexity_summary(OUTER_COMPLEXITIES[index * 25:(index + 1) * 25])
        for ranking in domain["eight_model_pr_auc_ranking"]:
            if ranking["candidate"] == "tabiclv2_default_eight_view":
                ranking["candidate"] = NAME
                ranking["source"] = "current_figs_evaluation"
        tabic_domain = tabic["domains"][domain_name]
        ranking = [*domain["eight_model_pr_auc_ranking"],
                   {"candidate": "tabiclv2_default_eight_view",
                    "mean_average_precision": tabic_domain["repeat_metric_summary"]["average_precision_pr_auc"]["mean"],
                    "source": "saved_tabiclv2_development_report"}]
        domain["nine_model_pr_auc_ranking"] = sorted(ranking, key=lambda row: (-float(row["mean_average_precision"]), str(row["candidate"])))
        domain["tabiclv2_numerical_comparison"] = {
            metric: {"figs_mean": domain["repeat_metric_summary"][metric]["mean"],
                     "tabiclv2_mean": tabic_domain["repeat_metric_summary"][metric]["mean"],
                     "figs_minus_tabiclv2": domain["repeat_metric_summary"][metric]["mean"] - tabic_domain["repeat_metric_summary"][metric]["mean"]}
            for metric in ("average_precision_pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "precision", "negative_predictive_value", "f1", "brier_score")}
        for values in domain["numerical_comparison_to_historical_selected_model"].values():
            values["figs_mean"] = values.pop("tabiclv2_mean")
            values["figs_minus_reference"] = values.pop("tabiclv2_minus_reference")
    temporary = output.with_name("." + output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(result, indent=2) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
    print(json.dumps({"status": result["status"], "outer_folds": 75,
                      "runtime_seconds": result["runtime_seconds"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
