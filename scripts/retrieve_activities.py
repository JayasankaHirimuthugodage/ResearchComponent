"""Run a non-identifying, internal-research activity retrieval query."""

from __future__ import annotations

import argparse
from pathlib import Path

from activity_recommendation.catalogue import DEFAULT_LIBRARY_PATH, load_activity_catalogue
from activity_recommendation.contracts import (
    AbilityState,
    EligibilityRequest,
    RetrievalMode,
    SafetyContext,
)
from activity_recommendation.eligibility import filter_eligible_activities
from activity_recommendation.embeddings import SentenceTransformerEncoder, load_embedding_artifacts
from activity_recommendation.retrieval import retrieve_top_k


DEFAULT_ARTIFACTS = Path("artifacts/activity-embeddings/0.4.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--skill-query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    for dimension in ("hearing", "vision", "fine-motor", "gross-motor"):
        parser.add_argument(
            f"--{dimension}",
            required=True,
            choices=[state.value for state in AbilityState],
            help="Explicit safety-context state; unknown is handled conservatively.",
        )
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    catalogue = load_activity_catalogue(args.library)
    safety_context = SafetyContext(
        hearing=AbilityState(args.hearing),
        vision=AbilityState(args.vision),
        fine_motor=AbilityState(args.fine_motor),
        gross_motor=AbilityState(args.gross_motor),
    )
    eligibility = filter_eligible_activities(
        catalogue.records,
        EligibilityRequest(
            checkpoint_id=args.checkpoint,
            domain=args.domain,
            safety_context=safety_context,
            mode=RetrievalMode.INTERNAL_RESEARCH_DEMO,
        ),
    )
    artifacts = load_embedding_artifacts(catalogue, args.artifacts)
    results = retrieve_top_k(
        args.skill_query,
        eligibility,
        artifacts,
        SentenceTransformerEncoder(),
        top_k=args.top_k,
    )
    for rank, result in enumerate(results, start=1):
        activity = result.activity
        print(
            f"{rank}. {activity.activity_id} | {activity.source_org} | "
            f"{','.join(activity.checkpoint_ids)} | {','.join(activity.domains)} | "
            f"similarity={result.similarity_score:.6f}"
        )


if __name__ == "__main__":
    main()
