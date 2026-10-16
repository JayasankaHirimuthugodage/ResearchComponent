"""One development-only Cognitive TabICLv2/L2-logistic weighted hybrid experiment."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, math, os, platform, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn, torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabicl import TabICLClassifier

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, TARGET_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, modeling_arrays, select_balanced_accuracy_threshold
from training.tabiclv2.candidate import TABICLV2_CHECKPOINT_NAME, TABICLV2_CHECKPOINT_SHA256, TABICLV2_PACKAGE_VERSION, tabiclv2_frame

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CV_SEED, MODEL_SEED, INNER_SPLITS, BOOTSTRAP_REPLICATES = 20260821, 20260822, 3, 2000
TARGET = TARGET_COLUMNS[1]
EPS = 1e-6
REPORT = Path("reports/generated/development-cognitive-tabiclv2-weighted-hybrid-report.json")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest().upper()

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p)).reshape(-1,1)

def calibrator(raw,y,seed):
    m=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,random_state=seed); m.fit(logit(raw),y); return m

def logistic_pipeline(seed):
    return Pipeline((("preprocess",ColumnTransformer((("checkpoint",OneHotEncoder(categories=[list(CHECKPOINT_IDS)],drop="first",handle_unknown="ignore",sparse_output=False),[0]),("questionnaire",StandardScaler(),list(range(1,len(FEATURE_COLUMNS))))),remainder="drop",verbose_feature_names_out=False)),("estimator",LogisticRegression(penalty="l2",C=1.0,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=seed))))

def lr_prob(train_x,train_y,predict_x,seed):
    return logistic_pipeline(seed).fit(train_x,train_y).predict_proba(predict_x)[:,1].astype(float)

def tab_prob(train_x,train_y,predict_x,checkpoint,seed):
    m=TabICLClassifier(n_estimators=8,model_path=str(checkpoint),allow_auto_download=False,device="cpu",use_amp=False,use_fa3=False,offload_mode=False,random_state=seed,n_jobs=2,verbose=False)
    m.fit(tabiclv2_frame(train_x),train_y)
    p=np.asarray(m.predict_proba(tabiclv2_frame(predict_x))[:,1],float); del m
    if p.shape != (len(predict_x),) or np.any(~np.isfinite(p)): raise RuntimeError("Invalid TabICLv2 probabilities")
    return p

def metrics(y,p,t):
    d=(p>=t).astype(np.int8); tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
    return {"average_precision_pr_auc":float(average_precision_score(y,p)),"roc_auc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,d)),"balanced_accuracy":float(balanced_accuracy_score(y,d)),"sensitivity":float(recall_score(y,d,zero_division=0)),"specificity":float(tn/(tn+fp)),"precision":float(precision_score(y,d,zero_division=0)),"negative_predictive_value":float(tn/(tn+fn)) if tn+fn else 0.0,"f1":float(f1_score(y,d,zero_division=0)),"brier_score":float(brier_score_loss(y,p)),"confusion_matrix":{"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp)}}

def select_weight(y,tab,lr):
    scores={w:float(average_precision_score(y,w*tab+(1-w)*lr)) for w in WEIGHTS}
    best=max(scores.values()); tied=[w for w in WEIGHTS if math.isclose(scores[w],best,abs_tol=1e-12)]
    selected=sorted(tied,key=lambda w:(w!=0.5,abs(w-0.5),w))[0]
    return selected,scores

def bootstrap(y,weighted,tab,lr,equal):
    rng=np.random.default_rng(CV_SEED+90001); classes=[np.flatnonzero(y==v) for v in (0,1)]
    refs={"weighted_minus_tabiclv2":tab,"weighted_minus_logistic":lr,"weighted_minus_equal_weight":equal}; values=defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        ix=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes])
        wp=average_precision_score(y[ix],weighted[ix])
        for name,p in refs.items(): values[name].append(float(wp-average_precision_score(y[ix],p[ix])))
    return {name:{"point_difference":float(average_precision_score(y,weighted)-average_precision_score(y,p)),"bootstrap_mean_difference":float(np.mean(v)),"confidence_interval_95":[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]} for name,(p,v) in zip(refs.keys(),[(refs[k],values[k]) for k in refs])}

def summary(rows):
    names=["average_precision_pr_auc","roc_auc","accuracy","balanced_accuracy","sensitivity","specificity","precision","negative_predictive_value","f1","brier_score"]
    return {n:{"mean":float(np.mean([r[n] for r in rows])),"standard_deviation":float(np.std([r[n] for r in rows],ddof=1)),"minimum":float(np.min([r[n] for r in rows])),"maximum":float(np.max([r[n] for r in rows]))} for n in names}

def preflight(root,checkpoint):
    release=json.loads((root/"models/experimental-tabiclv2-secondary-v1/release-manifest.json").read_text())
    final=json.loads((root/"reports/generated/development-tabiclv2-finalization-report.json").read_text())
    if (root/REPORT).exists(): raise FileExistsError("Refusing to repeat completed weighted-hybrid experiment")
    if release["test_cohort_evaluated_for_these_candidates"] is not False or final["final_test_evaluated"] is not False: raise ValueError("Frozen artifacts record unexpected test access")
    if release["tabiclv2"]["package_version"]!=TABICLV2_PACKAGE_VERSION or importlib.metadata.version("tabicl")!=TABICLV2_PACKAGE_VERSION: raise ValueError("TabICLv2 version mismatch")
    if checkpoint.name!=TABICLV2_CHECKPOINT_NAME or sha(checkpoint.resolve())!=TABICLV2_CHECKPOINT_SHA256: raise ValueError("Checkpoint verification failed")
    dev=load_development_dataset(); folds=create_shared_folds(dev.records,seed=CV_SEED)
    if len(dev.records)!=500 or dev.dataset_sha256!=release["dataset_sha256"] or dev.split_fingerprint_sha256!=release["split_fingerprint_sha256"]: raise ValueError("Development provenance mismatch")
    if fold_plan_fingerprint(folds)!=release["shared_fold_plan_sha256"]: raise ValueError("Fold fingerprint mismatch")
    if tuple(release["feature_contract"])!=FEATURE_COLUMNS or TARGET not in release["targets"]: raise ValueError("Feature/target contract mismatch")
    return dev,folds,release,final

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--checkpoint",type=Path,default=Path(os.environ.get("TABICLV2_CHECKPOINT",Path(os.sys.prefix)/"hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"))); a=p.parse_args()
    root=repository_root(); checkpoint=a.checkpoint.absolute(); dev,folds,release,final=preflight(root,checkpoint)
    if a.preflight_only:
        print(json.dumps({"status":"preflight_passed","development_participants":len(dev.records),"outer_folds":len(folds),"weights":WEIGHTS,"test_outcomes_accessed":False},indent=2)); return 0
    start=time.perf_counter(); x,yall=modeling_arrays(dev.records); y=yall[:,1]; methods={"logistic":{},"tabiclv2":{},"equal_weight":{},"selected_weight":{}}; repeat_metrics={k:[] for k in methods}; repeat_probs={k:[] for k in methods}; repeat_decisions={k:[] for k in methods}; fold_rows=[]
    for repeat in range(5):
        probs={k:np.full(len(y),np.nan) for k in methods}; decisions={k:np.zeros(len(y),np.int8) for k in methods}
        for fold in [f for f in folds if f.repeat_index==repeat]:
            seed=MODEL_SEED+1_000_000+fold.repeat_index*10_000+fold.fold_index*100; tx,ty=x[fold.train_indices],y[fold.train_indices]; vx=x[fold.validation_indices]
            split=StratifiedKFold(n_splits=INNER_SPLITS,shuffle=True,random_state=seed+1); it=np.full(len(ty),np.nan); il=np.full(len(ty),np.nan)
            for inner,(fi,vi) in enumerate(split.split(tx,ty)):
                it[vi]=tab_prob(tx[fi],ty[fi],tx[vi],checkpoint,seed+10+inner); il[vi]=lr_prob(tx[fi],ty[fi],tx[vi],seed+10+inner)
            ct,cl=calibrator(it,ty,seed+50),calibrator(il,ty,seed+60); cit=ct.predict_proba(logit(it))[:,1]; cil=cl.predict_proba(logit(il))[:,1]
            selected,inner_scores=select_weight(ty,cit,cil); inner={"logistic":cil,"tabiclv2":cit,"equal_weight":.5*cit+.5*cil,"selected_weight":selected*cit+(1-selected)*cil}; thresholds={k:float(select_balanced_accuracy_threshold(ty,v)["threshold"]) for k,v in inner.items()}
            ot=ct.predict_proba(logit(tab_prob(tx,ty,vx,checkpoint,seed+75)))[:,1]; ol=cl.predict_proba(logit(lr_prob(tx,ty,vx,seed+75)))[:,1]; outer={"logistic":ol,"tabiclv2":ot,"equal_weight":.5*ot+.5*ol,"selected_weight":selected*ot+(1-selected)*ol}
            for name in methods: probs[name][fold.validation_indices]=outer[name]; decisions[name][fold.validation_indices]=(outer[name]>=thresholds[name]).astype(np.int8)
            fold_rows.append({"repeat":repeat+1,"fold":fold.fold_index+1,"training_participants":len(tx),"validation_participants":len(vx),"inner_selected_weight":selected,"inner_weight_pr_auc":inner_scores,"thresholds":thresholds,"validation_metrics":{k:metrics(y[fold.validation_indices],outer[k],thresholds[k]) for k in methods}})
            done=len(fold_rows); elapsed=time.perf_counter()-start; eta=elapsed/done*(25-done); print(f"[{done}/25] repeat={repeat+1} fold={fold.fold_index+1} weight={selected:.2f} elapsed={elapsed:.1f}s ETA={eta:.1f}s",flush=True)
        for name in methods: repeat_metrics[name].append(metrics(y,probs[name],float('nan')) if False else metrics(y,probs[name],0.5))
        # Replace temporary 0.5 decisions with the fold-local decisions for decision metrics while preserving probability metrics.
        for name in methods:
            m=metrics(y,probs[name],0.5); d=decisions[name]; tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel(); m.update({"accuracy":float(accuracy_score(y,d)),"balanced_accuracy":float(balanced_accuracy_score(y,d)),"sensitivity":float(recall_score(y,d,zero_division=0)),"specificity":float(tn/(tn+fp)),"precision":float(precision_score(y,d,zero_division=0)),"negative_predictive_value":float(tn/(tn+fn)),"f1":float(f1_score(y,d,zero_division=0)),"confusion_matrix":{"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp)}}); repeat_metrics[name][-1]=m; repeat_probs[name].append(probs[name]); repeat_decisions[name].append(decisions[name])
    avgp={k:np.mean(v,axis=0) for k,v in repeat_probs.items()}; avgd={k:(np.mean(repeat_decisions[k],axis=0)>=.5).astype(np.int8) for k in methods}
    participant={}
    for name in methods:
        m=metrics(y,avgp[name],.5); d=avgd[name]; tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel(); m.update({"accuracy":float(accuracy_score(y,d)),"balanced_accuracy":float(balanced_accuracy_score(y,d)),"sensitivity":float(recall_score(y,d,zero_division=0)),"specificity":float(tn/(tn+fp)),"precision":float(precision_score(y,d,zero_division=0)),"negative_predictive_value":float(tn/(tn+fn)),"f1":float(f1_score(y,d,zero_division=0)),"confusion_matrix":{"true_negative":int(tn),"false_positive":int(fp),"false_negative":int(fn),"true_positive":int(tp)}}); participant[name]=m
    counts=Counter(r["inner_selected_weight"] for r in fold_rows); report={"artifact_schema_version":"cognitive-tabiclv2-weighted-hybrid-development-v1","status":"completed_development_only_exploratory_weighted_hybrid","scope":"development_only_post_hoc_exploratory","development_participants":len(y),"feature_contract":list(FEATURE_COLUMNS),"target":TARGET,"weights":list(WEIGHTS),"weight_selection":"maximize_inner_development_pr_auc; ties=0.50_then_nearest_to_0.50_then_lower","cross_validation":{"outer_splits":5,"outer_repeats":5,"outer_folds_completed":len(fold_rows),"inner_splits":3,"shared_fold_plan_sha256":fold_plan_fingerprint(folds)},"methods":{k:{"repeat_metric_summary":summary(repeat_metrics[k]),"participant_averaged_oof_metrics":participant[k]} for k in methods},"paired_bootstrap_pr_auc":bootstrap(y,avgp["selected_weight"],avgp["tabiclv2"],avgp["logistic"],avgp["equal_weight"]),"weight_selection_frequency":{str(w):int(counts[w]) for w in WEIGHTS},"weight_selection_proportion":{str(w):float(counts[w]/len(fold_rows)) for w in WEIGHTS},"threshold_variability":{k:{"mean":float(np.mean([r['thresholds'][k] for r in fold_rows])),"standard_deviation":float(np.std([r['thresholds'][k] for r in fold_rows],ddof=1)),"minimum":float(np.min([r['thresholds'][k] for r in fold_rows])),"maximum":float(np.max([r['thresholds'][k] for r in fold_rows]))} for k in methods},"calibration":"separate_inner_oof_sigmoid_on_logit_for_each_component; weighted_probabilities_are_combined_after_calibration","frozen_artifact_hashes":{"release_manifest":sha(root/'models/experimental-tabiclv2-secondary-v1/release-manifest.json'),"finalization_report":sha(root/'reports/generated/development-tabiclv2-finalization-report.json'),"checkpoint":sha(checkpoint.resolve())},"models_refit_on_test":False,"test_cohort_loaded":False,"privacy":{"participant_identifiers_written":False,"row_level_predictions_written":False,"labelled_context_persisted":False},"limitations":["Post hoc exploratory development-only analysis.","Weights add model-selection risk.","Repeated CV estimates share participants.","No independent validation or test-cohort evaluation.","Targets are questionnaire-defined follow-up concerns, not diagnoses."],"runtime_seconds":time.perf_counter()-start,"runtime_versions":{"python":platform.python_version(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"torch":torch.__version__,"tabicl":importlib.metadata.version('tabicl')}}
    out=root/REPORT; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError("Refusing overwrite")
    temp=out.with_name('.'+out.name+'.tmp')
    try:
        with temp.open('xb') as f: f.write((json.dumps(report,indent=2)+"\n").encode()); f.flush(); os.fsync(f.fileno())
        os.replace(temp,out)
    finally:
        if temp.exists(): temp.unlink()
    print(json.dumps({"status":report["status"],"outer_folds":len(fold_rows),"runtime_seconds":report["runtime_seconds"],"report":str(REPORT)},indent=2))

if __name__=='__main__': main()
