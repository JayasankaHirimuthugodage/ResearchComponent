"""Finalize TabICLv2 on development data and assess narrow domain hybrids."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, modeling_arrays, select_balanced_accuracy_threshold
from training.tabiclv2.candidate import (TABICLV2_CHECKPOINT_NAME, TABICLV2_CHECKPOINT_SHA256,
    TABICLV2_CONFIGURATION_VERSION, TABICLV2_PACKAGE_VERSION, tabiclv2_frame, tabiclv2_probabilities)


CV_SEED, MODEL_SEED, INNER_SPLITS = 20260821, 20260822, 3
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _calibrator(raw, y, seed):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    model.fit(_logit(raw), y)
    return model


def _metrics(y, p, d):
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {
        "average_precision_pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, d)),
        "balanced_accuracy": float(balanced_accuracy_score(y, d)),
        "sensitivity": float(recall_score(y, d, zero_division=0)),
        "specificity": float(tn / (tn + fp)), "precision": float(precision_score(y, d, zero_division=0)),
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else 0.0,
        "f1": float(f1_score(y, d, zero_division=0)), "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def _summary(rows):
    names = [k for k, v in rows[0].items() if isinstance(v, float)]
    return {name: {"mean": float(np.mean([r[name] for r in rows])),
        "standard_deviation": float(np.std([r[name] for r in rows], ddof=1)),
        "minimum": float(np.min([r[name] for r in rows])), "maximum": float(np.max([r[name] for r in rows]))}
        for name in names}


def _bootstrap(y, methods, reference, replicates=2000):
    rng = np.random.default_rng(CV_SEED + 90000)
    indices = [np.flatnonzero(y == value) for value in (0, 1)]
    values = defaultdict(list)
    for _ in range(replicates):
        sample = np.concatenate([rng.choice(i, len(i), replace=True) for i in indices])
        for method in methods:
            values[method].append(average_precision_score(y[sample], methods[method][sample]) - average_precision_score(y[sample], reference[sample]))
    return {method: {"mean_difference": float(np.mean(v)), "confidence_interval_95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for method, v in values.items()}


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("TABICLV2_CHECKPOINT", Path(os.sys.prefix) / "hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt")))
    parser.add_argument("--report", type=Path, default=repository_root()/"reports/generated/development-tabiclv2-finalization-report.json")
    parser.add_argument("--release", type=Path, default=repository_root()/"models/experimental-tabiclv2-secondary-v1/release-manifest.json")
    args = parser.parse_args()
    for path in (args.report, args.release):
        if path.exists(): raise FileExistsError(f"Refusing to overwrite {path}")
    checkpoint = args.checkpoint.resolve()
    if _sha(checkpoint) != TABICLV2_CHECKPOINT_SHA256: raise ValueError("Checkpoint checksum mismatch")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION: raise ValueError("Unexpected tabicl version")
    development = load_development_dataset()
    features, targets = modeling_arrays(development.records)
    folds = create_shared_folds(development.records, seed=CV_SEED)
    bundle = json.loads(args.reference_bundle.read_text(encoding="utf-8"))
    if bundle["shared_fold_plan_sha256"] != fold_plan_fingerprint(folds): raise ValueError("Fold fingerprint mismatch")
    started = time.perf_counter(); domains = {}; release_domains = {}
    for position, target in enumerate(TARGET_COLUMNS):
        raw = {"tabiclv2": {r: np.zeros(len(features)) for r in range(5)}, "original_selected": {r: np.zeros(len(features)) for r in range(5)}}
        calibrated = {name: {r: np.zeros(len(features)) for r in range(5)} for name in (*raw, "equal_weight_hybrid")}
        decisions = {name: {r: np.zeros(len(features), dtype=np.int8) for r in range(5)} for name in calibrated}
        fold_rows = []; reference_rows = bundle["domains"][target]["folds"]
        for index, fold in enumerate(folds):
            seed = MODEL_SEED + position * 1_000_000 + fold.repeat_index * 10_000 + fold.fold_index * 100
            train_x, train_y = features[fold.train_indices], targets[fold.train_indices, position]
            splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed + 1)
            tab_inner = np.full(len(train_y), np.nan)
            for inner_index, (fit_i, val_i) in enumerate(splitter.split(train_x, train_y)):
                tab_inner[val_i] = tabiclv2_probabilities(train_x[fit_i], train_y[fit_i], train_x[val_i], checkpoint_path=checkpoint, seed=seed + 10 + inner_index)
            tab_outer = tabiclv2_probabilities(train_x, train_y, features[fold.validation_indices], checkpoint_path=checkpoint, seed=seed + 75)
            ref_inner = np.asarray(reference_rows[index]["inner_raw_probabilities"]); ref_outer = np.asarray(reference_rows[index]["outer_raw_probabilities"])
            inner_cal = {}; outer_cal = {}
            for name, inner_values, outer_values in (("tabiclv2", tab_inner, tab_outer), ("original_selected", ref_inner, ref_outer)):
                cal = _calibrator(inner_values, train_y, seed + (50 if name == "tabiclv2" else 60))
                inner_cal[name] = cal.predict_proba(_logit(inner_values))[:, 1]
                outer_cal[name] = cal.predict_proba(_logit(outer_values))[:, 1]
                raw[name][fold.repeat_index][fold.validation_indices] = outer_values
            inner_cal["equal_weight_hybrid"] = (inner_cal["tabiclv2"] + inner_cal["original_selected"]) / 2
            outer_cal["equal_weight_hybrid"] = (outer_cal["tabiclv2"] + outer_cal["original_selected"]) / 2
            thresholds = {}
            for name in calibrated:
                thresholds[name] = float(select_balanced_accuracy_threshold(train_y, inner_cal[name])["threshold"])
                calibrated[name][fold.repeat_index][fold.validation_indices] = outer_cal[name]
                decisions[name][fold.repeat_index][fold.validation_indices] = outer_cal[name] >= thresholds[name]
            fold_rows.append({"repeat": fold.repeat_index + 1, "fold": fold.fold_index + 1, "thresholds": thresholds})
            print(f"[{position * 25 + index + 1}/75] {target} repeat={fold.repeat_index+1} fold={fold.fold_index+1}", flush=True)
        y = targets[:, position]; methods = {}; avg_prob = {}; avg_dec = {}
        for name in calibrated:
            repeat_metrics = [_metrics(y, calibrated[name][r], decisions[name][r]) for r in range(5)]
            avg_prob[name] = np.mean(list(calibrated[name].values()), axis=0)
            avg_dec[name] = (np.mean(list(decisions[name].values()), axis=0) >= .5).astype(np.int8)
            methods[name] = {"repeat_metric_summary": _summary(repeat_metrics), "participant_averaged_oof_metrics": _metrics(y, avg_prob[name], avg_dec[name])}
        corr = float(np.corrcoef(avg_prob["tabiclv2"], avg_prob["original_selected"])[0, 1])
        tab_correct = avg_dec["tabiclv2"] == y; ref_correct = avg_dec["original_selected"] == y
        comp = {"probability_pearson_correlation": corr, "decision_disagreement_rate": float(np.mean(avg_dec["tabiclv2"] != avg_dec["original_selected"])),
            "tabiclv2_only_correct_rate": float(np.mean(tab_correct & ~ref_correct)), "original_only_correct_rate": float(np.mean(~tab_correct & ref_correct))}
        justified = bool(corr < .95 and comp["tabiclv2_only_correct_rate"] >= .02 and comp["original_only_correct_rate"] >= .02)
        final_tab_raw = np.mean(list(raw["tabiclv2"].values()), axis=0)
        final_cal = _calibrator(final_tab_raw, y, MODEL_SEED + position + 8_000_000)
        final_p = final_cal.predict_proba(_logit(final_tab_raw))[:, 1]
        selected = select_balanced_accuracy_threshold(y, final_p)
        domains[target] = {"original_selected_candidate": bundle["domains"][target]["candidate"], "complementarity": comp,
            "hybrid_justified_by_predefined_rule": justified, "hybrid_rule": "correlation_below_0.95_and_each_model_uniquely_correct_on_at_least_2_percent",
            "methods": methods, "paired_bootstrap_pr_auc_vs_original": _bootstrap(y, {"tabiclv2": avg_prob["tabiclv2"], "equal_weight_hybrid": avg_prob["equal_weight_hybrid"]}, avg_prob["original_selected"]), "fold_results": fold_rows}
        release_domains[target] = {"candidate": "tabiclv2_default_eight_view", "calibration": {"method": "sigmoid_on_logit_of_five_repeat_averaged_raw_oof_probability", "coefficient": float(final_cal.coef_[0,0]), "intercept": float(final_cal.intercept_[0])},
            "threshold": {"value": float(selected["threshold"]), "selection_method": selected["objective"], "tie_breaking": selected["tie_breaking"]},
            # Complementarity permits evaluation, but a hybrid is not frozen unless its
            # improvement is convincing. The current bounded run treats the bootstrap
            # uncertainty review as a separate Phase-3 selection gate.
            "hybrid_candidate_frozen": False,
            "hybrid": None,
            "hybrid_selection_note": (
                "Complementarity screen passed, but the development PR-AUC improvement "
                "requires a confidence interval excluding zero before secondary evaluation."
                if justified
                else "The predefined complementarity screen did not pass."
            )}
        # Prepare each full 500-person context in memory without serializing participant data.
        from tabicl import TabICLClassifier
        context = TabICLClassifier(n_estimators=8, model_path=str(checkpoint), allow_auto_download=False, device="cpu", use_amp=False, use_fa3=False, offload_mode=False, random_state=MODEL_SEED + position, n_jobs=4, verbose=False)
        context.fit(tabiclv2_frame(features), y); del context
    report = {"artifact_schema_version": "tabiclv2-finalization-development-v1", "status": "development_finalization_complete_pending_test_approval", "scope": "development_only",
        "development_participants": 500, "feature_count": len(FEATURE_COLUMNS), "targets": list(TARGET_COLUMNS), "dataset_sha256": development.dataset_sha256, "split_fingerprint_sha256": development.split_fingerprint_sha256,
        "shared_fold_plan_sha256": fold_plan_fingerprint(folds), "final_test_evaluated": False, "domains": domains, "runtime_seconds": time.perf_counter()-started,
        "privacy": {"participant_identifiers_written": False, "row_level_predictions_written": False, "labelled_context_persisted": False}}
    release = {"artifact_schema_version": "tabiclv2-experimental-release-v1", "status": "frozen_experimental_candidates_pending_secondary_test_approval", "deployment_status": "not_deployed",
        "test_cohort_evaluated_for_these_candidates": False, "development_participants": 500, "feature_contract": list(FEATURE_COLUMNS), "targets": list(TARGET_COLUMNS),
        "dataset_sha256": development.dataset_sha256, "split_fingerprint_sha256": development.split_fingerprint_sha256, "shared_fold_plan_sha256": fold_plan_fingerprint(folds),
        "seeds": {"cv": CV_SEED, "model": MODEL_SEED}, "tabiclv2": {"package_version": TABICLV2_PACKAGE_VERSION, "configuration_version": TABICLV2_CONFIGURATION_VERSION,
        "checkpoint_filename": TABICLV2_CHECKPOINT_NAME, "checkpoint_sha256": TABICLV2_CHECKPOINT_SHA256, "n_estimators": 8, "device": "cpu"},
        "final_context": {"participants": 500, "persistence": "reconstructed_in_memory_from_checksum_verified_development_partition", "cross_validation_context_participants": 400,
        "limitation": "Final 500-participant context differs from the 400-participant outer-fold contexts and approximately 266-person inner contexts."}, "domains": release_domains,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "tabicl": importlib.metadata.version("tabicl")}}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.release.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8"); args.release.write_text(json.dumps(release, indent=2)+"\n", encoding="utf-8")
    args.reference_bundle.unlink()
    print(json.dumps({"status": report["status"], "report": str(args.report), "release": str(args.release), "temporary_bundle_removed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
