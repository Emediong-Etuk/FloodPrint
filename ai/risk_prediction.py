"""Flood-risk model wrapper and inference.

Prediction target (defined before training):
    Probability that a large flood event recorded by the Dartmouth Flood
    Observatory (DFO) *begins* within the next 7 days in an area covering the
    location, given recent and forecast daily rainfall, season, elevation and
    the area's recent flood record.

The model is trained on West / Central African weather stations (NOAA GSOD)
labelled with DFO events. It estimates *regional, rainfall-driven* flood risk.
It has not been validated against street-level flooding in Port Harcourt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ai.feature_engineering import FEATURES, PREDICTION_WINDOW_LABEL

log = logging.getLogger(__name__)

RISK_BANDS: list[tuple[float, str]] = [
    (0.20, "Low"),
    (0.45, "Moderate"),
    (0.70, "High"),
    (1.00, "Very High"),
]
RISK_COLORS = {"Low": "#22C55E", "Moderate": "#EAB308", "High": "#F59E0B", "Very High": "#EF4444"}


def risk_category(probability: float) -> str:
    """Map a probability to the interface bands (0-20 / 21-45 / 46-70 / 71-100 %)."""
    if probability is None or not np.isfinite(probability):
        raise ValueError("probability must be a finite number")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability out of range: {probability}")
    for upper, name in RISK_BANDS:
        if probability <= upper + 1e-12:
            return name
    return "Very High"


class CalibratedRiskModel:
    """A fitted classifier plus an isotonic calibrator fitted on held-out data."""

    def __init__(self, estimator, calibrator, features: list[str], medians: dict[str, float]):
        self.estimator = estimator
        self.calibrator = calibrator
        self.features = list(features)
        self.medians = dict(medians)

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.features].to_numpy(dtype=float))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_proba(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


@dataclass
class RiskPrediction:
    probability: float
    category: str
    window: str
    model_version: str
    features: dict[str, float]
    imputed_features: list[str]
    contributions: list[dict] = field(default_factory=list)
    base_rate: float | None = None
    explanation: str = ""

    @property
    def lift(self) -> float | None:
        if not self.base_rate:
            return None
        return self.probability / self.base_rate


class ModelUnavailableError(RuntimeError):
    pass


class RiskPredictor:
    """Loads the persisted model bundle and produces predictions."""

    def __init__(self, bundle: dict):
        required = {"model", "version", "features", "medians"}
        missing = required - set(bundle)
        if missing:
            raise ModelUnavailableError(f"Model bundle is missing keys: {sorted(missing)}")
        self.bundle = bundle
        self.model: CalibratedRiskModel = bundle["model"]
        self.version: str = bundle["version"]
        self.features: list[str] = bundle["features"]
        self.medians: dict[str, float] = bundle["medians"]
        self.metadata: dict = bundle.get("metadata", {})
        self.base_rate: float | None = bundle.get("metadata", {}).get("train_base_rate")
        self.reference: dict[str, float] = bundle.get("reference", self.medians)

    @classmethod
    def load(cls, path: Path | str) -> "RiskPredictor":
        import joblib

        path = Path(path)
        if not path.exists():
            raise ModelUnavailableError(
                f"Risk model not found at {path}. Run `python scripts/train_model.py` first."
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # corrupt / incompatible pickle
            raise ModelUnavailableError(f"Could not load risk model: {exc}") from exc
        return cls(bundle)

    def _prepare(self, features: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
        unknown = set(features) - set(self.features)
        if unknown:
            log.debug("Ignoring unknown features: %s", sorted(unknown))
        row, imputed = {}, []
        for name in self.features:
            val = features.get(name)
            if val is None or not np.isfinite(val):
                row[name] = float(self.medians[name])
                imputed.append(name)
            else:
                row[name] = float(val)
        return pd.DataFrame([row], columns=self.features), imputed

    def predict(self, features: dict[str, float], explain: bool = True) -> RiskPrediction:
        if len(set(self.features) - {k for k, v in features.items() if v is not None}) == len(self.features):
            raise ValueError("No usable features supplied")
        X, imputed = self._prepare(features)
        prob = float(self.model.predict_proba(X)[0])
        contributions = []
        if explain:
            from ai.explainability import local_contributions

            contributions = local_contributions(self.model, X, self.reference)
        pred = RiskPrediction(
            probability=prob,
            category=risk_category(prob),
            window=PREDICTION_WINDOW_LABEL,
            model_version=self.version,
            features={k: float(X.iloc[0][k]) for k in self.features},
            imputed_features=imputed,
            contributions=contributions,
            base_rate=self.base_rate,
        )
        if explain:
            from ai.explainability import explain_prediction

            pred.explanation = explain_prediction(pred)
        return pred


def default_model_path() -> Path:
    from config import MODELS_DIR

    return MODELS_DIR / "risk_model.joblib"


__all__ = [
    "FEATURES",
    "RISK_BANDS",
    "RISK_COLORS",
    "CalibratedRiskModel",
    "ModelUnavailableError",
    "RiskPrediction",
    "RiskPredictor",
    "default_model_path",
    "risk_category",
]
