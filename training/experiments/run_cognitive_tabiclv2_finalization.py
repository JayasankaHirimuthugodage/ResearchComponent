"""Cognitive-only, development-only TabICLv2 finalization infrastructure.

This module is intentionally not invoked by Phase 1.  A later approved run derives
one final calibrator/Youden threshold from repeat-averaged Cognitive OOF predictions
and prepares the 500-person context only in memory.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from training.contracts import FEATURE_COLUMNS, TARGET_COLUMNS, default_dataset_path, repository_root
from training.data.development_data import default_manifest_path, default_split_summary_path, load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.tabiclv2_evaluation import _fit_calibrator, _logit, _metrics, fold_plan_fingerprint, modeling_arrays, select_balanced_accuracy_threshold
from training.tabiclv2.candidate import TABICLV2_CHECKPOINT_NAME, TABICLV2_CHECKPOINT_SHA256, TABICLV2_PACKAGE_VERSION, tabiclv2_frame, tabiclv2_probabilities

TARGET="target_cognitive_concern"; INNER_SPLITS=3; MODEL_SEED=20260822; FINGERPRINT="7AB8604BB0DD59DA33420D2743730A468D22C06FFD7EE1E78395EB5050B92C39"; LIMIT=3600
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def validate_aggregate(value:object)->None:
    encoded=json.dumps(value)
    if 'participant_key' in encoded or 'feature_values' in encoded: raise ValueError('Aggregate artifact must not contain participant-level data.')
def inner_oof(x,y,checkpoint,seed):
    out=np.full(len(y),np.nan)
    for i,(fit,val) in enumerate(StratifiedKFold(n_splits=INNER_SPLITS,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
        out[val]=tabiclv2_probabilities(x[fit],y[fit],x[val],checkpoint_path=checkpoint,seed=seed+i)
    if not np.isfinite(out).all():raise RuntimeError('Incomplete inner OOF probabilities.')
    return out
def _atomic_json(path:Path,value:object,*,overwrite:bool=False)->None:
    """Atomic aggregate-only JSON write; never leaves a valid-looking partial file."""
    if path.exists() and not overwrite:raise FileExistsError(f'Refusing to overwrite {path}')
    validate_aggregate(value); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp')
    try:
        with tmp.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
def write_once(path:Path,value:object)->None:
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    _atomic_json(path,value)
def _progress(path, value):
    _atomic_json(path,value,overwrite=True)
def artifact_sha256(path:Path)->str:return sha(path)
def validate_checkpoint_and_package(checkpoint:Path)->None:
    """Fail before data loading or model fitting unless the frozen runtime contract holds."""
    if checkpoint.name!=TABICLV2_CHECKPOINT_NAME or not checkpoint.is_file() or sha(checkpoint)!=TABICLV2_CHECKPOINT_SHA256:raise ValueError('Approved checkpoint verification failed.')
    if importlib.metadata.version('tabicl')!=TABICLV2_PACKAGE_VERSION:raise ValueError('tabicl version mismatch.')
def validate_repeat_probabilities(repeats:list[np.ndarray], participant_count:int=500)->np.ndarray:
    """Require one finite outer-held-out probability per participant in each repeat."""
    if len(repeats)!=5:raise ValueError('Expected exactly five repeat probability vectors.')
    vectors=[np.asarray(value,dtype=float) for value in repeats]
    if any(value.shape!=(participant_count,) or not np.isfinite(value).all() for value in vectors):raise ValueError('Incomplete or misaligned repeat probabilities.')
    return np.mean(np.stack(vectors,axis=0),axis=0)
def record_outer_probabilities(destination:np.ndarray, indices:np.ndarray, probabilities:np.ndarray)->None:
    """Assign one outer-held-out probability once; duplicate alignment is invalid."""
    indices=np.asarray(indices,dtype=int); probabilities=np.asarray(probabilities,dtype=float)
    if indices.ndim!=1 or probabilities.shape!=indices.shape or len(np.unique(indices))!=len(indices) or (indices<0).any() or (indices>=len(destination)).any() or not np.isfinite(probabilities).all() or np.isfinite(destination[indices]).any():raise ValueError('Invalid or duplicate outer-fold probability alignment.')
    destination[indices]=probabilities
def enforce_runtime_limit(start:float, completed_outer_folds:int, *, clock=time.perf_counter)->None:
    """Called only at completed outer-fold boundaries; it never restarts work."""
    if completed_outer_folds and clock()-start>=LIMIT:raise RuntimeError('60-minute limit reached at fold boundary.')
def verify_release_manifest(path:Path)->dict:
    value=json.loads(path.read_text(encoding='utf-8'))
    required={'artifact_schema_version','status','domain','checkpoint_sha256','tabicl_version','configuration','feature_contract','calibration','threshold','privacy','limitations'}
    if not required.issubset(value) or value.get('domain')!='cognitive' or value.get('status')!='development_only_experimental_not_deployed':raise ValueError('Invalid experimental release manifest.')
    if tuple(value['feature_contract'])!=tuple(FEATURE_COLUMNS) or not 0<=float(value['threshold']['value'])<=1:raise ValueError('Release feature/threshold contract invalid.')
    if value['privacy'].get('participant_identifiers_written') or value['privacy'].get('row_level_predictions_written'):raise ValueError('Release privacy contract invalid.')
    return value
def write_integrity_index(index:Path, artifacts:dict[str,Path], root:Path|None=None)->None:
    if index.exists(): raise FileExistsError(f'Refusing to overwrite {index}')
    root=(root or index.parent).resolve()
    _atomic_json(index,{'schema':'cognitive-tabiclv2-integrity-index-v1','artifacts':{n:{'path':str(p.resolve().relative_to(root)),'sha256':sha(p)} for n,p in artifacts.items()}})
def verify_integrity_index(index:Path, root:Path)->dict:
    value=json.loads(index.read_text(encoding='utf-8'))
    if value.get('schema')!='cognitive-tabiclv2-integrity-index-v1' or not value.get('artifacts'): raise ValueError('Invalid integrity index.')
    for name,item in value['artifacts'].items():
        p=(root/item.get('path','')).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=item.get('sha256'): raise ValueError(f'Integrity failure: {name}')
    return value
def verify_release_bundle(manifest:Path, index:Path, root:Path)->None:
    """A release is complete only when its manifest and detached index both verify."""
    verify_release_manifest(manifest); verify_integrity_index(index,root)
def execute_finalization(d,x,y,folds,checkpoint,report_path,release_path,progress_path):
    """Approved algorithm; call only through --execute after separate authorization."""
    start=time.perf_counter(); raw=[]; decisions=[]; rows=[]
    for repeat in range(5):
        p=np.full(len(y),np.nan); q=np.zeros(len(y),dtype=np.int8)
        for fold in (f for f in folds if f.repeat_index==repeat):
            seed=MODEL_SEED+fold.repeat_index*10000+fold.fold_index*100; tx,ty=x[fold.train_indices],y[fold.train_indices]
            inner=inner_oof(tx,ty,checkpoint,seed+1); cal=_fit_calibrator(inner,ty,seed=seed+50); inner_c=cal.predict_proba(_logit(inner))[:,1]; selected=select_balanced_accuracy_threshold(ty,inner_c)
            outer=tabiclv2_probabilities(tx,ty,x[fold.validation_indices],checkpoint_path=checkpoint,seed=seed+75); outer_c=cal.predict_proba(_logit(outer))[:,1]; dec=(outer_c>=selected['threshold']).astype(np.int8); record_outer_probabilities(p,fold.validation_indices,outer); q[fold.validation_indices]=dec
            rows.append({'repeat':repeat+1,'fold':fold.fold_index+1,'threshold':float(selected['threshold']),'validation_metrics':_metrics(y[fold.validation_indices],outer_c,dec)})
            done=len(rows); elapsed=time.perf_counter()-start; projected=elapsed/done*25
            _progress(progress_path,{'scope':'development_only','test_cohort_loaded':False,'completed_outer_folds':done,'total_outer_folds':25,'elapsed_seconds':elapsed,'estimated_total_seconds':projected,'fold_results':rows,'privacy':{'participant_identifiers_written':False,'row_level_predictions_written':False}})
            if done>=3 and projected>LIMIT: raise RuntimeError('Measured projection exceeds 60-minute limit.')
            enforce_runtime_limit(start,done)
        raw.append(p); decisions.append(q)
    average_raw=validate_repeat_probabilities(raw,len(y)); final_cal=_fit_calibrator(average_raw,y,seed=MODEL_SEED+8000000); final_p=final_cal.predict_proba(_logit(average_raw))[:,1]; threshold=select_balanced_accuracy_threshold(y,final_p)
    from tabicl import TabICLClassifier
    context=TabICLClassifier(n_estimators=8,model_path=str(checkpoint),allow_auto_download=False,device='cpu',use_amp=False,use_fa3=False,offload_mode=False,random_state=MODEL_SEED,n_jobs=4,verbose=False); context.fit(tabiclv2_frame(x),y); del context
    release={'artifact_schema_version':'experimental-cognitive-tabiclv2-v1','status':'development_only_experimental_not_deployed','domain':'cognitive','checkpoint_sha256':sha(checkpoint),'tabicl_version':TABICLV2_PACKAGE_VERSION,'configuration':{'n_estimators':8,'device':'cpu','automatic_downloads':False,'fine_tuning':False,'seed':MODEL_SEED},'feature_contract':list(FEATURE_COLUMNS),'development_participants':500,'dataset_sha256':d.dataset_sha256,'split_fingerprint_sha256':d.split_fingerprint_sha256,'shared_fold_plan_sha256':fold_plan_fingerprint(folds),'calibration':{'method':'sigmoid_on_logit_of_five_repeat_averaged_raw_oof','coefficient':float(final_cal.coef_[0,0]),'intercept':float(final_cal.intercept_[0])},'threshold':{'value':float(threshold['threshold']),'policy':threshold['objective'],'tie_breaking':threshold['tie_breaking']},'context_reconstruction':{'participants':500,'persistence':'reconstruct_in_memory_from_authorized_checksum_verified_development_partition','warning':'application inference integration requires separately approved secure local data-access design'},'limitations':['post_hoc_development_only','OOF calibration/threshold selection reuse development outcomes and are selection-optimistic','not clinically approved or independently validated'],'privacy':{'participant_identifiers_written':False,'row_level_predictions_written':False}}
    nested=_metrics(y, np.mean(raw,axis=0), (np.mean(decisions,axis=0)>=.5).astype(np.int8))
    report={'status':'completed_development_only_experimental_finalization','scope':'development_only','test_cohort_loaded':False,'nested_outer_metrics':[nested],'final_release':release,'fold_results':rows,'runtime_seconds':time.perf_counter()-start,'privacy':release['privacy']}; validate_aggregate(report); validate_aggregate(release); write_once(report_path,report); write_once(release_path,release); index_path=release_path.with_name('integrity-index.json'); write_integrity_index(index_path,{'aggregate_report':report_path,'release_manifest':release_path},repository_root()); verify_release_bundle(release_path,index_path,repository_root())
def main():
    root=repository_root(); p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=default_dataset_path());p.add_argument('--manifest',type=Path,default=default_manifest_path());p.add_argument('--split-summary',type=Path,default=default_split_summary_path());p.add_argument('--checkpoint',type=Path,default=Path(sys.prefix)/'hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt');p.add_argument('--report',type=Path,default=root/'reports/generated/development-cognitive-tabiclv2-finalization-report.json');p.add_argument('--release',type=Path,default=root/'models/experimental-cognitive-tabiclv2-v1/release-manifest.json');p.add_argument('--progress',type=Path,default=root/'reports/generated/development-cognitive-tabiclv2-finalization-progress.json');p.add_argument('--preflight-only',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args(); ck=a.checkpoint.absolute()
    if not a.preflight_only and not a.execute: raise RuntimeError('Finalization is disabled by default; use --execute only after separate explicit approval.')
    if a.report.exists() or a.release.exists():raise FileExistsError('Output exists.')
    validate_checkpoint_and_package(ck)
    d=load_development_dataset(a.dataset,a.manifest,a.split_summary);x,t=modeling_arrays(d.records);folds=create_shared_folds(d.records)
    if len(d.records)!=500 or x.shape!=(500,16) or fold_plan_fingerprint(folds)!=FINGERPRINT:raise ValueError('Frozen development contract mismatch.')
    if a.preflight_only: print(json.dumps({'status':'preflight_passed','target':TARGET,'development_participants':500,'feature_count':16,'outer_evaluations':25,'test_cohort_loaded':False}));return
    execute_finalization(d,x,t[:,TARGET_COLUMNS.index(TARGET)],folds,ck,a.report,a.release,a.progress)
if __name__=='__main__':main()
