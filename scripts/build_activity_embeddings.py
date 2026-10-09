"""Build the local activity embedding matrix after validating the source library."""

from __future__ import annotations

import argparse
from pathlib import Path

from activity_recommendation.catalogue import DEFAULT_LIBRARY_PATH, load_activity_catalogue
from activity_recommendation.embeddings import (
    MODEL_NAME,
    MODEL_REVISION,
    SentenceTransformerEncoder,
    build_embedding_artifacts,
)


DEFAULT_OUTPUT = Path("artifacts/activity-embeddings/0.4.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    catalogue = load_activity_catalogue(args.library)
    encoder = SentenceTransformerEncoder(
        MODEL_NAME,
        MODEL_REVISION,
        local_files_only=False,
    )
    manifest = build_embedding_artifacts(catalogue, args.output, encoder)
    print(
        f"Built {manifest['record_count']} embeddings for library "
        f"{manifest['library_version']} using {manifest['embedding_model']} "
        f"at revision {manifest['embedding_model_revision']}."
    )


if __name__ == "__main__":
    main()
