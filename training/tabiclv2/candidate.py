"""Pinned TabICLv2 adapter preserving the 16-feature research contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier

from training.contracts import CHECKPOINT_IDS, FEATURE_COLUMNS


TABICLV2_PACKAGE_VERSION = "2.2.0"
TABICLV2_CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"
TABICLV2_CHECKPOINT_SHA256 = (
    "BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0"
)
TABICLV2_REPOSITORY = "jingang/TabICL"
TABICLV2_REPOSITORY_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICLV2_LICENSE = "BSD-3-Clause"
TABICLV2_CONFIGURATION_VERSION = "tabiclv2-default-eight-view-v1"


def tabiclv2_frame(features: np.ndarray) -> pd.DataFrame:
    """Represent checkpoint as categorical and questionnaire responses as ordinal integers."""

    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("TabICLv2 input must match the exact 16-feature contract.")
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame[FEATURE_COLUMNS[0]] = pd.Categorical(
        frame[FEATURE_COLUMNS[0]], categories=CHECKPOINT_IDS
    )
    for column in FEATURE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
        if not frame[column].isin((0, 1, 2)).all():
            raise ValueError("Questionnaire responses must remain ordinal values in {0,1,2}.")
    return frame


def tabiclv2_probabilities(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    *,
    checkpoint_path: Path,
    seed: int,
) -> np.ndarray:
    """Run one fresh, local-only in-context prediction."""

    model = TabICLClassifier(
        n_estimators=8,
        model_path=str(checkpoint_path),
        allow_auto_download=False,
        device="cpu",
        use_amp=False,
        use_fa3=False,
        offload_mode=False,
        random_state=seed,
        n_jobs=4,
        verbose=False,
    )
    model.fit(tabiclv2_frame(train_features), train_targets)
    probabilities = np.asarray(
        model.predict_proba(tabiclv2_frame(validation_features))[:, 1], dtype=float
    )
    if probabilities.shape != (len(validation_features),):
        raise RuntimeError("TabICLv2 returned an unexpected probability shape.")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise RuntimeError("TabICLv2 returned invalid probabilities.")
    return probabilities

