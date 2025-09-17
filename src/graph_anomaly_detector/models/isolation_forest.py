from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def fit_and_score(
    features: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Fit IsolationForest and return (model, anomaly_scores, anomaly_flags).

    anomaly_scores: higher means more anomalous.
    anomaly_flags: boolean array where True indicates anomaly.
    """
    scaler = RobustScaler()
    X = scaler.fit_transform(features.values)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # score_samples: higher is more normal → negate to get anomaly-ness
    scores = -model.score_samples(X)
    flags = model.predict(X) == -1

    return model, scores, flags
