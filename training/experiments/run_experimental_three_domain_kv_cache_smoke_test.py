"""Explicit local-only smoke test for the cache-backed experimental composite."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from inference.contracts import AssessmentStatus, QuestionnaireSubmission
from inference.experimental_release import EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE
from inference.questionnaire import load_questionnaire_catalog
from inference.tabiclv2_kv_cache_runtime import CACHE_RUNTIME_AUTHORIZATION, build_kv_cache_experimental_runtime_service

COMPOSITE_MANIFEST = ROOT / "models/experimental-three-domain-tabiclv2-kv-cache-v1/release-manifest.json"

def submission(catalog, incomplete=False):
    checkpoint="CP24M"; answers={q.question_id:"YES_USUALLY" for q in catalog.questions_for_checkpoint(checkpoint)}
    if incomplete: answers[next(iter(answers))]="NOT_SURE"
    return QuestionnaireSubmission(catalog.questionnaire_schema_version,checkpoint,answers)

def execute(checkpoint:Path)->dict:
    started=time.perf_counter(); process=psutil.Process(os.getpid()); catalog=load_questionnaire_catalog()
    service=build_kv_cache_experimental_runtime_service(catalog,repository_root=ROOT,composite_manifest=COMPOSITE_MANIFEST,checkpoint=checkpoint,execution_authorization=CACHE_RUNTIME_AUTHORIZATION)
    initialized=time.perf_counter(); complete=service.predict(submission(catalog)); warmed=time.perf_counter(); incomplete=service.predict(submission(catalog,True))
    if complete.status is not AssessmentStatus.COMPLETE or len(complete.domain_predictions)!=3 or complete.overall_concern is None: raise RuntimeError("Complete synthetic smoke contract failed.")
    if incomplete.status is not AssessmentStatus.INCOMPLETE or incomplete.domain_predictions or incomplete.overall_concern is not None: raise RuntimeError("Incomplete synthetic smoke contract failed.")
    return {"status":"completed","release_id":EXPERIMENTAL_KV_CACHE_COMPOSITE_RELEASE,"complete_contract_ok":True,"incomplete_contract_ok":True,"domain_order":[item.domain for item in complete.domain_predictions],"overall_or_verified":complete.overall_concern==any(item.decision for item in complete.domain_predictions),"initialization_seconds":initialized-started,"warm_prediction_seconds":warmed-initialized,"rss_bytes":process.memory_info().rss,"participant_data_read":False,"row_level_predictions_written":False}

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--execute-cache-inference",action="store_true"); parser.add_argument("--checkpoint",type=Path); args=parser.parse_args(argv)
    if not args.execute_cache_inference: print(json.dumps({"status":"blocked","reason":"--execute-cache-inference is required; no model I/O occurred."})); return 2
    if args.checkpoint is None: parser.error("--checkpoint is required")
    print(json.dumps(execute(args.checkpoint),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
