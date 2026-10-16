"""One development-only HistGradientBoostingClassifier comparison."""
from __future__ import annotations
import argparse,json,os,platform,time
from pathlib import Path
import numpy as np, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS, repository_root
from training.data.development_data import load_development_dataset
from training.experiments.cross_validation import create_shared_folds
from training.experiments.development_decision_report import sha256_file
from training.experiments.tabiclv2_evaluation import fold_plan_fingerprint, run_tabiclv2_evaluation

REPORT=Path('reports/generated/development-histgradientboosting-comparison-report.json')
CONFIG={'max_iter':150,'learning_rate':0.05,'max_leaf_nodes':15,'min_samples_leaf':10,'l2_regularization':1.0,'class_weight':'balanced','early_stopping':False}
NAME='histgradientboosting_balanced_onehot_checkpoint_v1'
def predict(train_x,train_y,validation_x,seed):
 pre=ColumnTransformer((('checkpoint',OneHotEncoder(categories=[list(CHECKPOINT_IDS)],drop='first',handle_unknown='ignore',sparse_output=False),[0]),('questionnaire','passthrough',list(range(1,len(FEATURE_COLUMNS))))),remainder='drop')
 model=HistGradientBoostingClassifier(**CONFIG,random_state=seed)
 return Pipeline((('preprocess',pre),('estimator',model))).fit(train_x,train_y).predict_proba(validation_x)[:,1].astype(float)
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args();root=repository_root();out=root/REPORT
 if out.exists():raise FileExistsError('Refusing to overwrite completed report')
 dev=load_development_dataset(); hist_path=root/'reports/generated/development-additional-model-comparison-report.json'; hist=json.loads(hist_path.read_text())
 folds=create_shared_folds(dev.records,seed=20260821)
 if len(dev.records)!=500 or fold_plan_fingerprint(folds)!=hist['cross_validation']['shared_fold_plan_sha256']:raise ValueError('Development provenance or fold fingerprint mismatch')
 if a.preflight_only:
  print(json.dumps({'status':'preflight_passed','development_participants':len(dev.records),'outer_folds':75,'configuration':CONFIG,'test_outcomes_accessed':False},indent=2));return
 start=time.perf_counter()
 def progress(e):
  if e['event']=='fold_completed': print(f"[{e['completed_outer_folds']}/75] {e['domain']} repeat={e['repeat']} fold={e['fold']} elapsed={e['elapsed_seconds']:.1f}s ETA={e['estimated_remaining_seconds']:.1f}s",flush=True)
 result=run_tabiclv2_evaluation(dev.records,fit_predict=predict,historical_report=hist,progress_callback=progress)
 result['artifact_schema_version']='histgradientboosting-development-comparison-v1';result['status']='completed_development_only_histgradientboosting_evaluation';result['scope']='development_only_exploratory_histgradientboosting_evaluation';result['candidate_names']=[NAME];result['histgradientboosting_configuration']=CONFIG;result['early_stopping_leakage_note']='Disabled; no internal validation partition was used.';result['test_cohort_loaded']=False;result['runtime_seconds']=time.perf_counter()-start;result['historical_additional_report_sha256']=sha256_file(hist_path);result['runtime_versions']={'python':platform.python_version(),'scikit_learn':sklearn.__version__}
 for d in result['domains'].values():
  d['candidate']=NAME
  for x in d['eight_model_pr_auc_ranking']:
   if x['candidate']=='tabiclv2_default_eight_view':x['candidate']=NAME;x['source']='current_histgradientboosting_evaluation'
  c=d['numerical_comparison_to_historical_selected_model']
  for v in c.values():v['histgradientboosting_mean']=v.pop('tabiclv2_mean');v['histgradientboosting_minus_reference']=v.pop('tabiclv2_minus_reference')
 temp=out.with_name('.'+out.name+'.tmp');out.parent.mkdir(parents=True,exist_ok=True)
 try:
  with temp.open('xb') as f:f.write((json.dumps(result,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
  os.replace(temp,out)
 finally:
  if temp.exists():temp.unlink()
 print(json.dumps({'status':result['status'],'outer_folds':75,'runtime_seconds':result['runtime_seconds'],'report':str(REPORT)},indent=2))
if __name__=='__main__':main()
