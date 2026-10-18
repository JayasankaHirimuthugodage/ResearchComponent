"""Explicitly gated, offline generation of a protected Cognitive TabICLv2 KV cache."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from inference.tabiclv2_adapter import load_cognitive_release_contract, verify_tabiclv2_runtime
from inference.tabiclv2_context import ContextSourceAuthorization, load_authorized_cognitive_context
from training.contracts import FEATURE_COLUMNS, default_dataset_path
from training.tabiclv2.candidate import TABICLV2_PACKAGE_VERSION, tabiclv2_frame

PROTECTED=Path(r"F:\4thYear\2026July\Research_Protected\TabICLv2_Cognitive_Context")
DESTINATION=Path(r"F:\4thYear\2026July\Research_Protected\TabICLv2_Cognitive_KV_Cache")
CHECKPOINT=ROOT/".venv-tabiclv2/hf-cache/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt"
RELEASE=ROOT/"models/experimental-cognitive-tabiclv2-v1/release-manifest.json"
TOLERANCE=1e-8
ENGINEERING_TOLERANCE=1e-6

def comparison(left_raw, right_raw, threshold:float)->dict:
    """Aggregate-only comparison; deliberately contains no request rows or probabilities."""
    left=np.asarray(left_raw,dtype=float); right=np.asarray(right_raw,dtype=float)
    if left.shape!=right.shape or left.ndim!=1: raise ValueError('Comparable one-dimensional predictions are required.')
    raw=np.abs(left-right); left_cal=sigmoid(left); right_cal=sigmoid(right); calibrated=np.abs(left_cal-right_cal)
    return {'count':int(left.size),'raw_max_absolute_difference':float(raw.max()),'raw_mean_absolute_difference':float(raw.mean()),'calibrated_max_absolute_difference':float(calibrated.max()),'calibrated_mean_absolute_difference':float(calibrated.mean()),'changed_threshold_decisions':int(np.count_nonzero((left_cal>=threshold)!=(right_cal>=threshold)))}

def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest().upper()

def sigmoid(raw):
    raw=np.clip(np.asarray(raw,dtype=float),1e-12,1-1e-12); logit=np.log(raw)-np.log1p(-raw)
    contract=load_cognitive_release_contract(RELEASE)
    return 1/(1+np.exp(-(contract.coefficient*logit+contract.intercept)))

def synthetic_features(verification_suite:bool=False):
    if verification_suite:
        from training.experiments.investigate_cognitive_tabiclv2_kv_cache_synthetic import FROZEN_SUITE_SHA256, frozen_rows, suite_sha256
        rows=frozen_rows()
        if suite_sha256(rows)!=FROZEN_SUITE_SHA256 or len(rows)!=128 or any(len(row)!=len(FEATURE_COLUMNS) for row in rows):
            raise ValueError('Frozen synthetic verification suite contract failed.')
        return tabiclv2_frame(np.asarray(rows,dtype=object))
    return tabiclv2_frame(np.asarray([["CP24M",*[str((row+column+1)%3) for column in range(15)]] for row in range(3)],dtype=object))

def authorization()->ContextSourceAuthorization:
    provenance=json.loads((PROTECTED/'provenance.json').read_text(encoding='utf-8'))
    return ContextSourceAuthorization(PROTECTED/'cognitive-development-context.csv',PROTECTED/'provenance.json',provenance['source_sha256'],provenance['membership_sha256'],provenance['dataset_sha256'],provenance['split_fingerprint_sha256'],default_dataset_path())

def preflight(destination:Path)->dict:
    contract=load_cognitive_release_contract(RELEASE); verify_tabiclv2_runtime(CHECKPOINT)
    if importlib.metadata.version('tabicl')!=TABICLV2_PACKAGE_VERSION: raise ValueError('TabICL version mismatch')
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Cache destination must be absent or empty.')
    if authorization().source_path.resolve()==authorization().forbidden_combined_source.resolve(): raise ValueError('Combined source is forbidden.')
    return {'checkpoint_sha256':contract.checkpoint_sha256,'tabicl_version':TABICLV2_PACKAGE_VERSION,'feature_count':len(FEATURE_COLUMNS),'destination':str(destination)}

def model(kv_cache:bool):
    from tabicl import TabICLClassifier
    return TabICLClassifier(n_estimators=8,kv_cache=kv_cache,model_path=str(CHECKPOINT),allow_auto_download=False,device='cpu',use_amp=False,use_fa3=False,offload_mode=False,random_state=20260822,n_jobs=4,verbose=False)

def write_json(path:Path,value:dict): path.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')

def persist_equivalence_diagnostic(destination:Path, diagnostic:dict)->Path:
    """Write aggregate failure evidence before an acceptance decision is raised."""
    path=destination/'equivalence-diagnostic.json'; write_json(path,diagnostic); return path

def reload_only(cache:Path, output:Path, expected_sha256:str, verification_suite:bool=False)->None:
    from tabicl import TabICLClassifier
    if sha(cache)!=expected_sha256: raise ValueError('New cache artifact checksum mismatch before reload.')
    started=time.perf_counter(); loaded=TabICLClassifier.load(cache,device='cpu'); loaded_at=time.perf_counter(); raw=loaded.predict_proba(synthetic_features(verification_suite))[:,1]; predicted=time.perf_counter()
    generator=loaded.ensemble_generator_; absent=getattr(generator,'X_',None) is None and getattr(generator,'y_',None) is None and all(getattr(item,'X_transformed_',None) is None for item in generator.preprocessors_.values())
    write_json(output,{'raw':raw.tolist(),'calibrated':sigmoid(raw).tolist(),'decisions':(sigmoid(raw)>=load_cognitive_release_contract(RELEASE).threshold).tolist(),'load_seconds':loaded_at-started,'prediction_seconds':predicted-loaded_at,'rss_bytes':psutil.Process(os.getpid()).memory_info().rss,'training_arrays_absent':absent,'cache_present':loaded.model_kv_cache_ is not None})

def threshold_proximity(raw)->dict:
    distance=np.abs(sigmoid(raw)-load_cognitive_release_contract(RELEASE).threshold)
    return {'minimum_calibrated_distance_to_threshold':float(distance.min()),'within_0.01':int(np.count_nonzero(distance<=.01)),'within_0.001':int(np.count_nonzero(distance<=.001)),'within_0.00001':int(np.count_nonzero(distance<=.00001))}

def execute(destination:Path, engineering_verification:bool=False)->dict:
    preflight(destination)
    if not destination.exists():
        destination.mkdir(parents=True,exist_ok=False)
    context=load_authorized_cognitive_context(authorization())
    X=tabiclv2_frame(context.feature_rows); y=context.cognitive_labels; test=synthetic_features(engineering_verification); started=time.perf_counter()
    direct=model(False); direct.fit(X,y); direct_raw=direct.predict_proba(test)[:,1]
    cached=model(True); cached.fit(X,y); cached_raw=cached.predict_proba(test)[:,1]; fit_at=time.perf_counter()
    cache=destination/'cognitive-tabiclv2-kv-cache.pkl'; cached.save(cache,save_model_weights=False,save_training_data=False,save_kv_cache=True); cache_sha256=sha(cache)
    cache_at=time.perf_counter(); reload_result=destination/'reload-summary.json'; reload_command=[sys.executable,str(Path(__file__)),'--reload-cache',str(cache),'--reload-output',str(reload_result),'--reload-sha256',cache_sha256]
    if engineering_verification: reload_command.append('--verification-suite')
    subprocess.run(reload_command,check=True,timeout=900)
    reloaded=json.loads(reload_result.read_text()); reload_raw=np.asarray(reloaded['raw']); contract=load_cognitive_release_contract(RELEASE)
    direct_cal=sigmoid(direct_raw); cached_cal=sigmoid(cached_raw); reload_cal=np.asarray(reloaded['calibrated'])
    comparisons={'direct_context_vs_cache_before_serialization':comparison(direct_raw,cached_raw,contract.threshold),'cache_before_serialization_vs_reloaded_cache':comparison(cached_raw,reload_raw,contract.threshold),'direct_context_vs_reloaded_cache':comparison(direct_raw,reload_raw,contract.threshold)}
    for item in comparisons.values():
        item['strict_1e-8_numerical_condition_passed']=item['raw_max_absolute_difference']<=TOLERANCE
        item['engineering_1e-6_numerical_condition_passed']=item['raw_max_absolute_difference']<=ENGINEERING_TOLERANCE and item['calibrated_max_absolute_difference']<=ENGINEERING_TOLERANCE
        item['unchanged_decision_condition_passed']=item['changed_threshold_decisions']==0
    threshold_coverage={'direct_context':threshold_proximity(direct_raw),'cache_before_serialization':threshold_proximity(cached_raw),'reloaded_cache':threshold_proximity(reload_raw)}
    suite={'kind':'phase_3i_frozen_128_synthetic_requests'} if engineering_verification else {'kind':'phase_3g_predefined_3_synthetic_requests'}
    if engineering_verification:
        from training.experiments.investigate_cognitive_tabiclv2_kv_cache_synthetic import FROZEN_SUITE_SHA256
        suite['sha256']=FROZEN_SUITE_SHA256
    engineering_pass=all(item['engineering_1e-6_numerical_condition_passed'] and item['unchanged_decision_condition_passed'] for item in comparisons.values())
    diagnostic={'schema':'experimental-cognitive-tabiclv2-kv-cache-engineering-equivalence-v1' if engineering_verification else 'experimental-cognitive-tabiclv2-kv-cache-diagnostic-v1','status':'engineering_equivalence_accepted_on_finite_suite' if engineering_verification and engineering_pass else 'equivalence_not_accepted','synthetic_assessments':int(len(direct_raw)),'suite':suite,'tolerance':TOLERANCE,'engineering_tolerance':ENGINEERING_TOLERANCE,'threshold':contract.threshold,'comparisons':comparisons,'threshold_proximity':threshold_coverage,'engineering_equivalence_passed':engineering_pass,'performance':{'generation_seconds':fit_at-started,'save_seconds':cache_at-fit_at,'reload_seconds':reloaded['load_seconds'],'warm_prediction_seconds':reloaded['prediction_seconds'],'generation_rss_bytes':psutil.Process(os.getpid()).memory_info().rss,'reload_rss_bytes':reloaded['rss_bytes'],'cache_bytes':cache.stat().st_size},'cache_artifact':{'name':cache.name,'sha256':cache_sha256},'configuration':{'n_estimators':8,'device':'cpu','random_state':20260822,'n_jobs':4,'use_amp':False,'use_fa3':False,'offload_mode':False,'allow_auto_download':False,'kv_cache':True},'runtime':{'python':sys.version.split()[0],'tabicl':importlib.metadata.version('tabicl'),'numpy':np.__version__,'scikit_learn':importlib.metadata.version('scikit-learn'),'torch':importlib.metadata.version('torch')},'privacy':{'ordinary_training_arrays_absent_after_reload':reloaded['training_arrays_absent'],'csv_not_used_by_reload_process':True,'row_level_predictions_written':False},'limitations':'A finite synthetic suite establishes bounded engineering evidence only. Threshold-adjacent coverage is reported separately.','interpretation':'Differences may arise in cache construction, cache serialization/reload, or numerical execution. This aggregate report does not identify a cause.'}
    persist_equivalence_diagnostic(destination,diagnostic)
    failed=not engineering_pass if engineering_verification else any(not item['strict_1e-8_numerical_condition_passed'] or not item['unchanged_decision_condition_passed'] for item in comparisons.values())
    if failed: raise RuntimeError('KV cache equivalence verification failed; aggregate diagnostic was written.')
    differences={'direct_vs_cache_raw':comparisons['direct_context_vs_cache_before_serialization']['raw_max_absolute_difference'],'cache_vs_reload_raw':comparisons['cache_before_serialization_vs_reloaded_cache']['raw_max_absolute_difference'],'direct_vs_cache_calibrated':comparisons['direct_context_vs_cache_before_serialization']['calibrated_max_absolute_difference'],'cache_vs_reload_calibrated':comparisons['cache_before_serialization_vs_reloaded_cache']['calibrated_max_absolute_difference']}
    manifest={'schema':'experimental-cognitive-tabiclv2-kv-cache-v1','status':'development_only_experimental_not_deployed','cache_artifact':cache.name,'cache_sha256':cache_sha256,'checkpoint':{'path':str(CHECKPOINT),'sha256':sha(CHECKPOINT)},'runtime':{'python':sys.version.split()[0],'tabicl':importlib.metadata.version('tabicl'),'numpy':np.__version__,'scikit_learn':importlib.metadata.version('scikit-learn'),'torch':importlib.metadata.version('torch')},'configuration':{'n_estimators':8,'device':'cpu','random_state':20260822,'kv_cache':True,'save_training_data':False,'save_kv_cache':True,'save_model_weights':False},'context_provenance':{'source_sha256':context.source_sha256,'participants':500,'cognitive_positive_labels':53},'calibration':{'coefficient':contract.coefficient,'intercept':contract.intercept},'threshold':{'value':contract.threshold,'comparison':'>='},'equivalence':{'tolerance':TOLERANCE,'max_absolute_differences':differences,'changed_decisions':0,'training_arrays_absent_after_reload':reloaded['training_arrays_absent'],'cache_present_after_reload':reloaded['cache_present']},'performance':{'generation_seconds':fit_at-started,'save_seconds':cache_at-fit_at,'reload_seconds':reloaded['load_seconds'],'warm_prediction_seconds':reloaded['prediction_seconds'],'generation_rss_bytes':psutil.Process(os.getpid()).memory_info().rss,'reload_rss_bytes':reloaded['rss_bytes'],'cache_bytes':cache.stat().st_size},'privacy':{'ordinary_training_arrays_serialized':False,'participant_derived_cache_requires_protection':True,'row_level_predictions_written':False}}
    if engineering_verification:
        write_json(destination/'engineering-equivalence-report.json',diagnostic)
        return diagnostic
    write_json(destination/'release-manifest.json',manifest); return manifest

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--engineering-verification',action='store_true'); p.add_argument('--preflight',action='store_true'); p.add_argument('--destination',type=Path,default=DESTINATION); p.add_argument('--reload-cache',type=Path); p.add_argument('--reload-output',type=Path); p.add_argument('--reload-sha256'); p.add_argument('--verification-suite',action='store_true'); a=p.parse_args(argv)
    if a.reload_cache:
        if not a.reload_output or not a.reload_sha256:
            p.error('--reload-output and --reload-sha256 are required')
        reload_only(a.reload_cache,a.reload_output,a.reload_sha256,a.verification_suite)
        return 0
    if a.preflight: print(json.dumps(preflight(a.destination),sort_keys=True)); return 0
    if not a.execute: print(json.dumps({'status':'blocked','reason':'--execute is required; no protected context was read.'})); return 2
    if a.engineering_verification and not a.execute: p.error('--engineering-verification requires --execute')
    print(json.dumps({'status':'completed','manifest':execute(a.destination,a.engineering_verification)},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
