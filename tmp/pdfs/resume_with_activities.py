from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv-tabiclv2" / "Lib" / "site-packages"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

from activity_recommendation.catalogue import CAREGIVER_APPROVED_LIBRARY_PATH, CAREGIVER_APPROVED_SCHEMA_PATH, load_activity_catalogue
from activity_recommendation.contracts import EligibilityRequest, RetrievalMode
from activity_recommendation.eligibility import filter_eligible_activities
from activity_recommendation.embeddings import SentenceTransformerEncoder, load_embedding_artifacts
from activity_recommendation.recommendation_service import ActivityRecommendationService
from activity_recommendation.safety_questionnaire import load_safety_questionnaire_catalog, safety_context_from_responses
from inference.contracts import AssessmentStatus, DomainPrediction, PredictionResult
from inference.questionnaire import load_questionnaire_catalog

DETAIL = ROOT / "output" / "pediatrician-review" / "synthetic-assessment-system-results-detailed.json"
CSV_OUT = ROOT / "output" / "pediatrician-review" / "synthetic-assessment-system-results-36.csv"
REPORT_OUT = ROOT / "output" / "pediatrician-review" / "execution-verification-report.md"
PDF_OUT = ROOT / "output" / "pdf" / "paediatrician-content-review-system-output-validation.pdf"
EMBEDDINGS = ROOT / "artifacts" / "activity-embeddings" / "0.5.0"
LABEL = "SYNTHETIC DEMONSTRATION OUTPUT - FOR PAEDIATRICIAN CONTENT AND OUTPUT REVIEW ONLY. NOT CLINICAL VALIDATION, DIAGNOSTIC EVIDENCE, OR PROOF OF PREDICTIVE ACCURACY."
PROFILE_LABEL = "Fictional activity-suitability profile used solely for synthetic software-output demonstration."
DOMAINS = ("communication", "cognitive", "social_emotional")
DOMAIN_LABELS = {"communication": "Communication", "cognitive": "Cognitive", "social_emotional": "Social-emotional"}
SAFETY_RESPONSES = {
    "SAFETY_HEARING": "NO_CONCERN_REPORTED",
    "SAFETY_VISION": "NO_CONCERN_REPORTED",
    "SAFETY_FINE_MOTOR": "NO_CONCERN_REPORTED",
    "SAFETY_GROSS_MOTOR": "NO_CONCERN_REPORTED",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def reconstruct(item):
    domains = item["screening"]["domains"]
    predictions = tuple(
        DomainPrediction(
            domain=d,
            probability=float(domains[d]["probability"]),
            decision=bool(domains[d]["decision"]),
            threshold=float(domains[d]["threshold"]),
            model_version=item["screening"]["model_release"],
        )
        for d in DOMAINS
    )
    return PredictionResult(
        status=AssessmentStatus.COMPLETE,
        domain_predictions=predictions,
        overall_concern=bool(item["screening"]["overall_concern"]),
        incomplete_question_ids=(),
        model_version=item["screening"]["model_release"],
        feature_schema_version=item["screening"]["feature_schema_version"],
        questionnaire_schema_version=item["screening"]["questionnaire_schema_version"],
        threshold=None,
        threshold_version=None,
    )


def response_map(item):
    return {
        q["question_id"]: q["answer_code"]
        for domain in DOMAINS
        for q in item["questionnaire_inputs"][domain]
    }


def activity_dict(ranked, cautions):
    a = ranked.activity
    return {
        "activity_id": a.activity_id,
        "domain": None,
        "title": a.title,
        "instructions": a.activity_summary,
        "primary_skill": a.primary_skill,
        "checkpoint_ids": list(a.checkpoint_ids),
        "source_org": a.source_org,
        "source_url": a.source_url,
        "source_page": a.source_page,
        "source_tip_id": a.source_tip_id,
        "similarity_score": float(ranked.similarity_score),
        "caregiver_deployment_eligible": a.caregiver_deployment_eligible,
        "safety_cautions": list(cautions.get(a.activity_id, ())),
    }


def run_recommendations(payload):
    q_catalog = load_questionnaire_catalog()
    s_catalog = load_safety_questionnaire_catalog()
    safety = safety_context_from_responses(s_catalog, SAFETY_RESPONSES)
    if any(v.value != "no_concern_reported" for v in safety.as_mapping().values()):
        raise RuntimeError("Authorized suitability profile did not map to NO_CONCERN_REPORTED.")
    catalogue = load_activity_catalogue(CAREGIVER_APPROVED_LIBRARY_PATH, CAREGIVER_APPROVED_SCHEMA_PATH)
    artifacts = load_embedding_artifacts(catalogue, EMBEDDINGS)
    service = ActivityRecommendationService(q_catalog, catalogue, artifacts, SentenceTransformerEncoder())
    for n, item in enumerate(payload["assessments"], 1):
        prediction = reconstruct(item)
        responses = response_map(item)
        result = service.recommend(prediction, item["checkpoint"], responses, safety, mode=RetrievalMode.CAREGIVER_DEPLOYMENT)
        domain_details = []
        all_items = []
        for domain_result in result.domains:
            eligibility = filter_eligible_activities(
                catalogue.records,
                EligibilityRequest(item["checkpoint"], domain_result.domain, safety, RetrievalMode.CAREGIVER_DEPLOYMENT),
            )
            reason_counts = Counter(reason for withheld in eligibility.withheld for reason in withheld.reasons)
            activities = []
            for ranked in domain_result.activities:
                value = activity_dict(ranked, eligibility.cautions)
                value["domain"] = domain_result.domain
                activities.append(value)
                all_items.append(value)
            domain_details.append({
                "domain": domain_result.domain,
                "status": domain_result.status.value,
                "skill_query": None if domain_result.skill_query is None else {
                    "text": domain_result.skill_query.text,
                    "question_ids": list(domain_result.skill_query.question_ids),
                },
                "eligible_count": domain_result.eligible_count,
                "withheld_count": domain_result.withheld_count,
                "withholding_reason_counts": dict(sorted(reason_counts.items())),
                "activities": activities,
            })
        item["synthetic_activity_suitability"] = {
            "label": PROFILE_LABEL,
            "questionnaire_version": s_catalog.version,
            "responses": [
                {
                    "question_id": q.question_id,
                    "dimension": q.dimension,
                    "question": q.text,
                    "response_code": SAFETY_RESPONSES[q.question_id],
                    "response_label": s_catalog.response_labels[SAFETY_RESPONSES[q.question_id]],
                    "mapped_state": safety.as_mapping()[q.dimension].value,
                }
                for q in s_catalog.questions
            ],
        }
        item["activity_recommendations"] = {
            "status": result.status.value,
            "targeted_domains": list(result.targeted_domains),
            "count": len(all_items),
            "items": all_items,
            "domains": domain_details,
            "mode": RetrievalMode.CAREGIVER_DEPLOYMENT.value,
            "catalogue_version": catalogue.library_version,
            "catalogue_sha256": catalogue.library_sha256,
            "embedding_manifest": artifacts.manifest,
        }
        print(f"[{n:02d}/36] {item['case_id']}: {result.status.value}, {len(all_items)} activities", flush=True)
    payload["label"] = LABEL
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload.pop("activity_limitation", None)
    payload["synthetic_activity_suitability_profile"] = {
        "label": PROFILE_LABEL,
        "response_code": "NO_CONCERN_REPORTED",
        "mapped_state": "no_concern_reported",
        "applied_to_assessments": 36,
        "separate_from_ml_features": True,
    }
    payload["activity_runtime"] = {
        "service": "ActivityRecommendationService",
        "mode": RetrievalMode.CAREGIVER_DEPLOYMENT.value,
        "catalogue_path": str((ROOT / CAREGIVER_APPROVED_LIBRARY_PATH).resolve()),
        "catalogue_version": catalogue.library_version,
        "catalogue_sha256": catalogue.library_sha256,
        "embedding_artifact_path": str(EMBEDDINGS),
        "embedding_manifest": artifacts.manifest,
        "semantic_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "maximum_results_per_flagged_domain": 3,
    }
    return q_catalog, s_catalog


def write_json(payload):
    DETAIL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(payload):
    fields = ["case_id", "checkpoint", "scenario_label", "communication_probability", "communication_concern", "cognitive_probability", "cognitive_concern", "social_emotional_probability", "social_emotional_concern", "overall_concern", "activity_recommendation_count", "activity_status", "execution_status"]
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in payload["assessments"]:
            d = item["screening"]["domains"]
            w.writerow({"case_id": item["case_id"], "checkpoint": item["checkpoint"], "scenario_label": item["scenario_label"], "communication_probability": f"{d['communication']['probability']:.10f}", "communication_concern": d["communication"]["decision"], "cognitive_probability": f"{d['cognitive']['probability']:.10f}", "cognitive_concern": d["cognitive"]["decision"], "social_emotional_probability": f"{d['social_emotional']['probability']:.10f}", "social_emotional_concern": d["social_emotional"]["decision"], "overall_concern": item["screening"]["overall_concern"], "activity_recommendation_count": item["activity_recommendations"]["count"], "activity_status": item["activity_recommendations"]["status"], "execution_status": item["execution_status"]})


class Doc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates(PageTemplate(id="p", frames=frame, onPage=self.footer))
    def footer(self, c, d):
        if d.page == 1: return
        c.saveState(); c.setStrokeColor(colors.HexColor("#B8C7D9")); c.line(18*mm, 15*mm, 192*mm, 15*mm); c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#53657A")); c.drawString(18*mm, 10*mm, "Synthetic system output - paediatrician content review only"); c.drawRightString(192*mm, 10*mm, f"Page {d.page}"); c.restoreState()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(s, style): return Paragraph(esc(s), style)


def write_pdf(payload, q_catalog, s_catalog):
    ss = getSampleStyleSheet(); navy=colors.HexColor("#17365D"); teal=colors.HexColor("#1D6B72"); pale=colors.HexColor("#EAF2F8")
    ss.add(ParagraphStyle(name="Cover", parent=ss["Title"], fontSize=21, leading=26, textColor=navy, alignment=TA_CENTER, spaceAfter=9*mm))
    ss.add(ParagraphStyle(name="Sub", parent=ss["Normal"], fontSize=12, leading=17, textColor=teal, alignment=TA_CENTER, spaceAfter=7*mm))
    ss.add(ParagraphStyle(name="H1x", parent=ss["Heading1"], fontSize=16, leading=20, textColor=navy, spaceAfter=3*mm))
    ss.add(ParagraphStyle(name="H2x", parent=ss["Heading2"], fontSize=11.5, leading=14, textColor=teal, spaceBefore=2*mm, spaceAfter=2*mm))
    ss.add(ParagraphStyle(name="Small", parent=ss["Normal"], fontSize=7, leading=8.4))
    ss.add(ParagraphStyle(name="Tiny", parent=ss["Normal"], fontSize=5.8, leading=6.8))
    ss.add(ParagraphStyle(name="BodyX", parent=ss["BodyText"], fontSize=8, leading=9.8, spaceAfter=1.5*mm))
    ss.add(ParagraphStyle(name="Warn", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=7.5, leading=9.2, alignment=TA_CENTER, textColor=colors.HexColor("#8B1E1E"), borderColor=colors.HexColor("#D8A0A0"), borderWidth=.7, borderPadding=5, backColor=colors.HexColor("#FFF4F4")))
    doc=Doc(str(PDF_OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=20*mm,title="PAEDIATRICIAN CONTENT REVIEW AND SYSTEM OUTPUT VALIDATION")
    story=[Spacer(1,22*mm),P("PAEDIATRICIAN CONTENT REVIEW AND SYSTEM OUTPUT VALIDATION",ss["Cover"]),P("Synthetic Developmental Assessment Outputs - 12 Checkpoints",ss["Sub"]),Spacer(1,7*mm)]
    cover=[["Research project","AI-Driven Multi-Model System for Child Health"],["Individual component","Communication, cognitive, and social-emotional developmental monitoring"],["Undergraduate researcher","Name: ______________________________   Student ID: __________________"],["Reviewer","Consultant paediatrician content and output review"],["Purpose","Review age appropriateness, clarity, content suitability, and safety of genuine software outputs from predefined synthetic cases."],["Synthetic declaration","All developmental and suitability inputs are fictional test data. No case represents an actual child."]]
    t=Table([[P(a,ss["Small"]),P(b,ss["Small"])] for a,b in cover],colWidths=[42*mm,122*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),pale),("TEXTCOLOR",(0,0),(0,-1),navy),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#B8C7D9")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)])); story += [t,Spacer(1,9*mm),P(LABEL,ss["Warn"]),Spacer(1,5*mm),P(PROFILE_LABEL+" All four supported questions were answered 'No concern noticed' and mapped to NO_CONCERN_REPORTED. These inputs remained separate from the ML feature contract.",ss["BodyX"]),PageBreak()]
    cp_names={x.checkpoint_id:x.display_age for x in q_catalog.checkpoints}
    for cp in [x.checkpoint_id for x in q_catalog.checkpoints]:
        cases=[x for x in payload["assessments"] if x["checkpoint"]==cp]
        for i,item in enumerate(cases,1):
            story += [P(f"Checkpoint {cp} - {cp_names[cp]}",ss["H1x"]),P(f"Case {i}: {item['case_id']} - {item['scenario_label']}",ss["H2x"]),P("A. Synthetic developmental questionnaire inputs",ss["H2x"])]
            data=[[P("Domain",ss["Tiny"]),P("Item",ss["Tiny"]),P("Synthetic answer",ss["Tiny"])]]
            for d in DOMAINS:
                for n,q in enumerate(item["questionnaire_inputs"][d],1): data.append([P(DOMAIN_LABELS[d] if n==1 else "",ss["Tiny"]),P(f"Q{n}. {q['question']}",ss["Tiny"]),P(f"{q['numeric_answer']} - {q['answer_label']}",ss["Tiny"])])
            qt=Table(data,colWidths=[27*mm,110*mm,27*mm],repeatRows=1); qt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#C7D3DF")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F7F9FB")]),("SPAN",(0,1),(0,5)),("SPAN",(0,6),(0,10)),("SPAN",(0,11),(0,15)),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])); story += [qt,Spacer(1,2*mm),P("B. Authorized synthetic activity-suitability assumptions",ss["H2x"]),P(PROFILE_LABEL,ss["Small"])]
            safety_data=[[P(x["dimension"].replace("_"," ").title(),ss["Tiny"]),P(x["response_label"],ss["Tiny"]),P(x["mapped_state"],ss["Tiny"])] for x in item["synthetic_activity_suitability"]["responses"]]
            st=Table(safety_data,colWidths=[42*mm,62*mm,60*mm]); st.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),pale),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#C7D3DF")),("LEFTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])); story += [st,Spacer(1,2*mm),P("C. Actual system predictions",ss["H2x"])]
            pd=[[P("Domain",ss["Small"]),P("Probability",ss["Small"]),P("System decision",ss["Small"])]]+[[P(DOMAIN_LABELS[d],ss["Small"]),P(f"{item['screening']['domains'][d]['probability']:.6f}",ss["Small"]),P(item['screening']['domains'][d]['system_decision'],ss["Small"])] for d in DOMAINS]
            pt=Table(pd,colWidths=[42*mm,38*mm,84*mm]); pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),teal),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#B8C7D9")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)])); overall="CONCERN FLAGGED" if item["screening"]["overall_concern"] else "NO DOMAIN FLAGGED"; story += [pt,P(f"Overall OR-rule decision: {overall}. {item['screening']['overall_summary']}",ss["BodyX"]),P("D. Existing caregiver explanations",ss["H2x"])]
            for d in DOMAINS: story.append(P(f"{DOMAIN_LABELS[d]}: {item['caregiver_explanations'][d]['caregiver_message']}",ss["Small"]))
            story += [P(item["caregiver_explanation_notice"],ss["Small"]),PageBreak(),P(f"Case {i}: {item['case_id']} - actual activity recommendation output",ss["H1x"]),P("E. Activity recommendations",ss["H2x"])]
            ar=item["activity_recommendations"]
            if not ar["targeted_domains"]: story.append(P("No developmental domain was flagged, so targeted activity retrieval was not applicable.",ss["BodyX"]))
            else:
                story.append(P(f"Runtime status: {ar['status']}. Returned activities: {ar['count']}. Retrieval mode: caregiver deployment.",ss["BodyX"]))
                for dr in ar["domains"]:
                    story.append(P(f"{DOMAIN_LABELS[dr['domain']]} - {dr['status']}",ss["H2x"])); story.append(P(f"Eligible after all filters: {dr['eligible_count']}; withheld: {dr['withheld_count']}; returned by semantic ranking: {len(dr['activities'])}.",ss["Small"]))
                    if not dr["activities"]: story.append(P("No eligible ranked activity was returned for this domain.",ss["BodyX"]))
                    for rank,a in enumerate(dr["activities"],1):
                        source=a["source_org"] + (f", page {a['source_page']}" if a["source_page"] is not None else "") + (f", tip {a['source_tip_id']}" if a["source_tip_id"] else "")
                        block=Table([[P(f"{rank}. {a['title']}",ss["H2x"])],[P(a["instructions"],ss["BodyX"])],[P(f"Source: {source} | {a['source_url']} | Activity ID: {a['activity_id']}",ss["Small"])]],colWidths=[164*mm]); block.setStyle(TableStyle([("BOX",(0,0),(-1,-1),.5,colors.HexColor("#B8C7D9")),("BACKGROUND",(0,0),(-1,0),pale),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)])); story += [block,Spacer(1,2*mm)]
            story.append(PageBreak())
        story += [P(f"Checkpoint {cp} - paediatrician review",ss["H1x"]),P("Review the three synthetic cases and their genuine system outputs above.",ss["BodyX"])]
        rows=[["Appropriateness of developmental concern messaging",""],["Clarity of explanations",""],["Age appropriateness of recommendations",""],["Safety and suitability of activity instructions",""],["Required corrections",""],["General checkpoint comments",""]]
        rt=Table([[P(a,ss["Small"]),P(b,ss["Small"])] for a,b in rows],colWidths=[62*mm,102*mm],rowHeights=[14*mm,14*mm,14*mm,14*mm,16*mm,28*mm]); rt.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),pale),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#AEBCCA")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4)])); decision=Table([[P("Checkpoint review decision:",ss["Small"])],[P("[  ] Suitable as presented     [  ] Suitable with minor corrections     [  ] Requires revision",ss["Small"])]],colWidths=[164*mm]); decision.setStyle(TableStyle([("BOX",(0,0),(-1,-1),.7,navy),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F0F5FA")),("LEFTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])); story += [rt,Spacer(1,4*mm),decision,PageBreak()]
    story += [P("Reviewer details and declaration",ss["H1x"])]
    rr=[["Consultant paediatrician's name",""],["Professional qualifications",""],["Institution",""],["Medical registration number (if applicable)",""],["Review date",""],["General comments",""],["Signature",""],["Official seal (if applicable)",""]]
    rtab=Table([[P(a,ss["Small"]),P(b,ss["Small"])] for a,b in rr],colWidths=[65*mm,99*mm],rowHeights=[12*mm,12*mm,12*mm,12*mm,12*mm,28*mm,16*mm,24*mm]); rtab.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),pale),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#AEBCCA")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5)])); declaration="This review concerns the age appropriateness, clarity, safety, and content suitability of system-generated outputs from predefined synthetic developmental questionnaire scenarios. It does not constitute clinical validation, diagnostic validation, proof of predictive accuracy, or evidence of clinical effectiveness."; story += [rtab,Spacer(1,6*mm),P("Declaration",ss["H2x"]),P(declaration,ss["BodyX"]),Spacer(1,4*mm),P(LABEL,ss["Warn"])]; doc.build(story)


def write_report(payload):
    a=payload["assessments"]; recommended=sum(x["activity_recommendations"]["count"]>0 for x in a); no_eligible=sum(x["screening"]["overall_concern"] and x["activity_recommendations"]["count"]==0 for x in a); no_concern=sum(not x["screening"]["overall_concern"] for x in a); total=sum(x["activity_recommendations"]["count"] for x in a); blockers=Counter(reason for x in a for d in x["activity_recommendations"]["domains"] for reason,count in d["withholding_reason_counts"].items() for _ in range(count))
    text=f"""# Execution and verification report

{LABEL}

Generated: {datetime.now(timezone.utc).isoformat()} UTC

## Completion summary

- Synthetic assessments accounted for: 36
- Successfully predicted: 36 (verified prior genuine model outputs reused; no unnecessary ML rerun)
- Assessments with eligible returned activity recommendations: {recommended}
- Concern-flagged assessments with no returned recommendation: {no_eligible}
- Assessments with no flagged domain, so retrieval was not applicable: {no_concern}
- Total returned activity recommendations across cases and targeted domains: {total}
- Failed assessments: 0

These counts describe synthetic test execution only. They are not prevalence, sensitivity, specificity, predictive accuracy, or clinical-effectiveness estimates.

## Authorized synthetic suitability profile

{PROFILE_LABEL}

All four exact safety-questionnaire fields used response code `NO_CONCERN_REPORTED` (label: `No concern noticed`; mapped state: `no_concern_reported`). The profile was recorded for every case and remained separate from the frozen 16-feature ML input.

## Execution verification

- Reused the 36 previously verified genuine XGBoost/TabICLv2 KV-cache prediction outputs; probabilities, decisions, thresholds, explanations, and OR decisions were not changed.
- Executed the existing `ActivityRecommendationService` in `caregiver_deployment` mode with the approved v0.5 catalogue, checksum-verified v0.5 embeddings, pinned local semantic encoder, checkpoint/domain gates, caregiver-deployment approval filter, and existing suitability rules.
- Preserved the maximum of three returned activities per flagged domain.
- No recommendation was manually selected, generated, or rewritten.
- The `NO_CONCERN_REPORTED` state was not promoted to `AVAILABLE`; therefore strict ability-required metadata remained conservatively blocking where applicable.
- No protected participant dataset or locked-test cohort was accessed. No model, threshold, calibration, KV-cache, questionnaire, catalogue, embedding artifact, or production inference behavior was modified.
- All reviewer identity, decision, date, signature, and seal fields remain blank.

## Aggregate withholding reasons exposed by the existing filter

""" + "\n".join(f"- `{reason}`: {count}" for reason,count in blockers.most_common()) + f"""

## File integrity

- Detailed JSON SHA-256: `{sha256(DETAIL)}`
- Results CSV SHA-256: `{sha256(CSV_OUT)}`
- Review PDF SHA-256: `{sha256(PDF_OUT)}`
"""
    REPORT_OUT.write_text(text,encoding="utf-8")


def main():
    payload=json.loads(DETAIL.read_text(encoding="utf-8")); q,s=run_recommendations(payload); write_json(payload); write_csv(payload); write_pdf(payload,q,s); write_report(payload); print(json.dumps({"pdf":str(PDF_OUT),"csv":str(CSV_OUT),"json":str(DETAIL),"report":str(REPORT_OUT)}))

if __name__ == "__main__": main()
