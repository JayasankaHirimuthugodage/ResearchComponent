"""Isolated, development-only Cognitive Logistic Regression versus TabICLv2 study."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, os, platform, sys, time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, TARGET_COLUMNS, default_dataset_path, repository_root
from training.data.development_data import default_manifest_path, default_split_summary_path, load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.run_tabiclv2_sensitivity_threshold_study import select_sensitivity_constrained_threshold
from training.experiments.tabiclv2_evaluation import _fit_calibrator, _logit, _metric_summary, _metrics, fold_plan_fingerprint, modeling_arrays, select_balanced_accuracy_threshold
from training.tabiclv2.candidate import TABICLV2_CHECKPOINT_NAME, TABICLV2_CHECKPOINT_SHA256, TABICLV2_CONFIGURATION_VERSION, TABICLV2_PACKAGE_VERSION, tabiclv2_probabilities

MODEL_SEED = 20260822
INNER_SPLITS = 3
SENSITIVITY_TARGET = 0.85
TARGET = "target_cognitive_concern"
FINGERPRINT = "7AB8604BB0DD59DA33420D2743730A468D22C06FFD7EE1E78395EB5050B92C39"
REPORT = "development-cognitive-tabiclv2-logistic-head-to-head-report.json"
PROGRESS = "development-cognitive-tabiclv2-logistic-head-to-head-progress.json"
RUNTIME_LIMIT_SECONDS = 60 * 60


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _lr_probability(train_x, train_y, validation_x, seed):
    pre = ColumnTransformer((("checkpoint", OneHotEncoder(categories=[list(CHECKPOINT_IDS)], drop="first", handle_unknown="ignore", sparse_output=False), [0]), ("questionnaire", StandardScaler(), list(range(1, len(FEATURE_COLUMNS))))), remainder="drop", verbose_feature_names_out=False)
    model = Pipeline((("preprocess", pre), ("estimator", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed))))
    return model.fit(train_x, train_y).predict_proba(validation_x)[:, 1].astype(float)


def _inner_probability(predict, train_x, train_y, checkpoint, seed):
    values = np.full(len(train_y), np.nan)
    for i, (fit, val) in enumerate(StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed).split(np.zeros(len(train_y)), train_y)):
        values[val] = predict(train_x[fit], train_y[fit], train_x[val], checkpoint, seed + i)
    if not np.isfinite(values).all(): raise RuntimeError("Incomplete inner OOF predictions")
    return values


def _tabic(train_x, train_y, validation_x, checkpoint, seed):
    return tabiclv2_probabilities(train_x, train_y, validation_x, checkpoint_path=checkpoint, seed=seed)


def _lr(train_x, train_y, validation_x, checkpoint, seed):
    del checkpoint
    return _lr_probability(train_x, train_y, validation_x, seed)


def _threshold_stability(rows, model, policy):
    x = np.asarray([row["thresholds"][model][policy] for row in rows], dtype=float)
    return {"mean": float(x.mean()), "standard_deviation": float(x.std(ddof=1)), "minimum": float(x.min()), "maximum": float(x.max())}


def _paired(y, left_p, left_d, right_p, right_d):
    rng = np.random.default_rng(20261018); names = ("average_precision_pr_auc", "roc_auc", "accuracy", "sensitivity", "specificity", "balanced_accuracy", "precision", "negative_predictive_value", "f1", "brier_score")
    def calc(ix, p, d): return _metrics(y[ix], p[ix], d[ix])
    a, b = calc(np.arange(len(y)), left_p, left_d), calc(np.arange(len(y)), right_p, right_d)
    values = {n: [] for n in names}; values.update({"false_negative": [], "false_positive": []})
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(2000):
        ix = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True))); aa, bb = calc(ix, left_p, left_d), calc(ix, right_p, right_d)
        for n in names: values[n].append(float(bb[n]) - float(aa[n]))
        for n in ("false_negative", "false_positive"): values[n].append(float(bb["confusion_matrix"][n]) - float(aa["confusion_matrix"][n]))
    point = {n: float(b[n]) - float(a[n]) for n in names}; point.update({n: int(b["confusion_matrix"][n]) - int(a["confusion_matrix"][n]) for n in ("false_negative", "false_positive")})
    return {"method": "stratified_participant_level_paired_bootstrap_on_repeat_averaged_oof_outputs", "replicates": 2000, "comparison": "tabiclv2_minus_logistic_regression", "metrics": {n: {"point_difference": point[n], "confidence_interval_95": [float(np.percentile(values[n], 2.5)), float(np.percentile(values[n], 97.5))]} for n in values}}


def _write_once(path, value):
    if path.exists(): raise FileExistsError(f"Refusing to overwrite {path}")
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        with tmp.open("xb") as f: f.write((json.dumps(value, indent=2) + "\n").encode()); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _write_progress(path, value):
    """Replace only this run's non-identifying aggregate progress record."""
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        with tmp.open("wb") as f: f.write((json.dumps(value, indent=2) + "\n").encode()); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _validate_serialization_structure():
    """Fail before fitting if the report's supported value types are not JSON serializable."""
    json.dumps({"artifact_schema_version": "cognitive-head-to-head-development-v1", "models": {"logistic_l2_balanced": {"C": 1.0}, "tabiclv2_default_eight_view": {"n_estimators": 8}}, "methods": {"youdens_j": {"participant_averaged_oof_metrics": {"accuracy": 0.0}}}, "fold_results": [], "paired_uncertainty": {"metrics": {}}, "privacy": {"aggregate_results_only": True}})


def main():
    root = repository_root().resolve(); parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, default=default_dataset_path()); parser.add_argument("--manifest", type=Path, default=default_manifest_path()); parser.add_argument("--split-summary", type=Path, default=default_split_summary_path()); parser.add_argument("--checkpoint", type=Path, default=Path(sys.prefix) / "hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"); parser.add_argument("--output", type=Path, default=root / "reports/generated" / REPORT); parser.add_argument("--progress", type=Path, default=root / "reports/generated" / PROGRESS); parser.add_argument("--preflight-only", action="store_true"); args=parser.parse_args()
    checkpoint=args.checkpoint.absolute(); output=args.output.resolve(); progress=args.progress.resolve()
    if output.exists() or not output.is_relative_to(root): raise ValueError("Output path is invalid or already exists")
    if checkpoint.name != TABICLV2_CHECKPOINT_NAME or not checkpoint.is_file() or _sha(checkpoint) != TABICLV2_CHECKPOINT_SHA256: raise ValueError("Checkpoint verification failed")
    if importlib.metadata.version("tabicl") != TABICLV2_PACKAGE_VERSION: raise ValueError("tabicl version mismatch")
    development=load_development_dataset(dataset_path=args.dataset, manifest_path=args.manifest, summary_path=args.split_summary); features, targets=modeling_arrays(development.records); folds=create_shared_folds(development.records); fingerprint=fold_plan_fingerprint(folds)
    if len(development.records)!=500 or features.shape!=(500,16) or targets.shape!=(500,3) or fingerprint!=FINGERPRINT: raise ValueError("Frozen development contract mismatch")
    _validate_serialization_structure()
    if args.preflight_only:
        print(json.dumps({"status":"preflight_passed","development_participants":500,"feature_count":16,"target":TARGET,"positive_count":int(targets[:, TARGET_COLUMNS.index(TARGET)].sum()),"outer_evaluations":25,"shared_fold_plan_sha256":fingerprint,"test_cohort_loaded":False},indent=2)); return
    started=time.perf_counter(); y=targets[:,TARGET_COLUMNS.index(TARGET)]; models={"logistic_l2_balanced":{"predict":_lr,"configuration":{"penalty":"l2","C":1.0,"class_weight":"balanced","solver":"liblinear","max_iter":2000}},"tabiclv2_default_eight_view":{"predict":_tabic,"configuration":{"n_estimators":8,"device":"cpu","automatic_downloads":False,"fine_tuning":False,"n_jobs":4}}}; probs={m:{p:[] for p in ("youdens_j","sensitivity_constrained")} for m in models}; decisions={m:{p:[] for p in ("youdens_j","sensitivity_constrained")} for m in models}; rows=[]
    for repeat in range(5):
        rp={m:{p:np.full(len(y),np.nan) for p in probs[m]} for m in models}; rd={m:{p:np.zeros(len(y),dtype=np.int8) for p in probs[m]} for m in models}
        for fold in (f for f in folds if f.repeat_index==repeat):
            seed=MODEL_SEED+fold.repeat_index*10000+fold.fold_index*100; tx,ty,vx=features[fold.train_indices],y[fold.train_indices],features[fold.validation_indices]; vy=y[fold.validation_indices]; row={"repeat":repeat+1,"fold":fold.fold_index+1,"training_participants":int(len(ty)),"validation_participants":int(len(vy)),"thresholds":{},"validation_metrics":{}}
            for name,item in models.items():
                inner_raw=_inner_probability(item["predict"],tx,ty,checkpoint,seed+1); cal=_fit_calibrator(inner_raw,ty,seed=seed+50); inner=cal.predict_proba(_logit(inner_raw))[:,1]; selected={"youdens_j":select_balanced_accuracy_threshold(ty,inner),"sensitivity_constrained":select_sensitivity_constrained_threshold(ty,inner)}; outer=cal.predict_proba(_logit(item["predict"](tx,ty,vx,checkpoint,seed+75)))[:,1]; row["thresholds"][name]={}; row["validation_metrics"][name]={}
                for policy,value in selected.items():
                    d=(outer>=value["threshold"]).astype(np.int8); rp[name][policy][fold.validation_indices]=outer; rd[name][policy][fold.validation_indices]=d; row["thresholds"][name][policy]=float(value["threshold"]); row["validation_metrics"][name][policy]=_metrics(vy,outer,d)
            rows.append(row); done=len(rows); elapsed=time.perf_counter()-started; projected_total=elapsed / done * 25; eta=projected_total-elapsed
            _write_progress(progress, {"scope":"development_only","test_cohort_loaded":False,"completed_outer_folds":done,"total_outer_folds":25,"last_completed":{"repeat":repeat+1,"fold":fold.fold_index+1},"elapsed_seconds":elapsed,"estimated_remaining_seconds":eta,"estimated_total_seconds":projected_total,"fold_results":rows,"privacy":{"participant_identifiers_written":False,"row_level_predictions_written":False,"aggregate_metrics_only":True}})
            print(f"[{done}/25] repeat={repeat+1} fold={fold.fold_index+1} elapsed={elapsed:.1f}s eta={eta:.1f}s projected_total={projected_total:.1f}s",flush=True)
            if done >= 3 and projected_total > RUNTIME_LIMIT_SECONDS: raise RuntimeError(f"Measured runtime projection exceeds {RUNTIME_LIMIT_SECONDS} seconds; stopped after {done}/25 folds.")
            if elapsed >= RUNTIME_LIMIT_SECONDS: raise RuntimeError(f"Runtime limit reached; stopped after {done}/25 folds.")
        for m in models:
            for p in probs[m]: probs[m][p].append(rp[m][p]); decisions[m][p].append(rd[m][p])
    methods={}; comparisons={}
    for m in models:
        methods[m]={}
        for p in probs[m]:
            avgp=np.mean(probs[m][p],axis=0); avgd=(np.mean(decisions[m][p],axis=0)>=.5).astype(np.int8); methods[m][p]={"repeat_metric_summary":_metric_summary([_metrics(y,probs[m][p][r],decisions[m][p][r]) for r in range(5)]),"participant_averaged_oof_metrics":_metrics(y,avgp,avgd),"threshold_independent_metric_note":"PR-AUC, AUROC, and Brier score are unchanged within a model across policies."}; methods[m][p]["_avg_probability"]=avgp; methods[m][p]["_avg_decision"]=avgd
    for p in ("youdens_j","sensitivity_constrained"):
        comparisons[p]=_paired(y,methods["logistic_l2_balanced"][p]["_avg_probability"],methods["logistic_l2_balanced"][p]["_avg_decision"],methods["tabiclv2_default_eight_view"][p]["_avg_probability"],methods["tabiclv2_default_eight_view"][p]["_avg_decision"])
    for m in methods:
        for p in methods[m]:
            methods[m][p].pop("_avg_probability"); methods[m][p].pop("_avg_decision")
    report={"artifact_schema_version":"cognitive-head-to-head-development-v1","status":"completed_development_only_post_hoc_exploratory","scope":"development_only","test_cohort_loaded":False,"development_participants":500,"feature_contract":list(FEATURE_COLUMNS),"target":TARGET,"positive_count":int(y.sum()),"negative_count":int(len(y)-y.sum()),"cross_validation":{"outer_splits":5,"outer_repeats":5,"outer_evaluations_completed":25,"inner_splits":3,"shared_fold_plan_sha256":fingerprint},"calibration":"inner_oof_regularized_logistic_sigmoid_on_raw_probability_logit","threshold_policies":{"youdens_j":"maximize_balanced_accuracy_equivalent_to_youdens_j","sensitivity_constrained":"maximize_specificity_subject_to_inner_sensitivity_at_least_0.85","sensitivity_target":.85,"target_interpretation":"illustrative_exploratory_assumption_not_clinical_standard"},"models":{name:item["configuration"] for name,item in models.items()},"methods":methods,"fold_results":rows,"threshold_stability":{m:{p:_threshold_stability(rows,m,p) for p in ("youdens_j","sensitivity_constrained")} for m in models},"paired_uncertainty":comparisons,"runtime_seconds":time.perf_counter()-started,"checkpoint_sha256":_sha(checkpoint),"privacy":{"participant_identifiers_written":False,"row_level_predictions_written":False,"aggregate_results_only":True},"limitations":["Post-hoc development-only comparison.","No configuration tuning was performed.","The 85% target is illustrative and not clinically endorsed.","No test or external validation."],"runtime_versions":{"python":platform.python_version(),"tabicl":importlib.metadata.version("tabicl")}}
    _write_once(output,report); print(json.dumps({"status":report["status"],"runtime_seconds":report["runtime_seconds"],"outer_evaluations_completed":25,"test_cohort_loaded":False,"output":str(output.relative_to(root))},indent=2))

if __name__ == "__main__": main()
