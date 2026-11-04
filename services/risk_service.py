"""Location-level flood risk: live weather -> ML model -> Flood Memory adjustment.

Two numbers are kept separate so users can see where each comes from:

1. ``model_probability`` - output of the trained Random Forest (regional,
   rainfall-driven; see ``ai/risk_prediction.py``).
2. ``risk_probability`` - the model probability after a *rule-based* local
   adjustment from FLOODPRINT's own incident history (Flood Memory). The
   adjustment multiplies the odds by
   ``exp(A * ln(1 + corroborated) + B * ln(1 + pending))`` for reports within
   ``RADIUS_M`` over the last ``LOOKBACK_DAYS``. A and B are fixed, documented
   constants, not learned values, because there is not yet enough labelled
   local data to learn them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai.feature_engineering import LiveFeatureInput, count_prior_events, haversine_km, live_feature_vector
from ai.risk_prediction import RiskPrediction, RiskPredictor, risk_category
from services.weather_service import WeatherService, WeatherSnapshot

log = logging.getLogger(__name__)

ADJ_CORROBORATED = 0.35
ADJ_PENDING = 0.10
RADIUS_M = 750
LOOKBACK_DAYS = 3 * 365
MAX_ADJUSTED = 0.95


@dataclass
class LocationRisk:
    location_name: str
    latitude: float
    longitude: float
    model_probability: float
    risk_probability: float
    risk_category: str
    odds_multiplier: float
    corroborated_nearby: int
    pending_nearby: int
    prediction: RiskPrediction
    weather: WeatherSnapshot
    notes: list[str] = field(default_factory=list)

    @property
    def data_timestamp(self) -> str:
        return self.weather.observed_at


def local_adjustment(incidents: pd.DataFrame, lat: float, lon: float, now: datetime) -> tuple[float, int, int]:
    if incidents is None or incidents.empty:
        return 1.0, 0, 0
    recent = incidents[incidents["reported_at"] >= pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))]
    if recent.empty:
        return 1.0, 0, 0
    d = haversine_km(lat, lon, recent["latitude"].values, recent["longitude"].values) * 1000
    near = recent[d <= RADIUS_M]
    n_c = int((near["verification_status"] == "CORROBORATED").sum())
    n_p = int(((near["verification_status"] == "PENDING") & near["duplicate_of"].isna()).sum())
    mult = math.exp(ADJ_CORROBORATED * math.log1p(n_c) + ADJ_PENDING * math.log1p(n_p))
    return mult, n_c, n_p


def apply_odds_multiplier(p: float, mult: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    odds = p / (1 - p) * mult
    return float(min(MAX_ADJUSTED, odds / (1 + odds)))


class RiskService:
    def __init__(self, predictor: RiskPredictor, weather: WeatherService, events_path: Path | None = None):
        self.predictor = predictor
        self.weather = weather
        self.events = None
        self.events_until = None
        if events_path and Path(events_path).exists():
            ev = pd.read_csv(events_path, parse_dates=["began", "ended"])
            self.events = ev
            self.events_until = ev["began"].max()

    def prior_events(self, lat: float, lon: float, today: pd.Timestamp) -> tuple[float | None, str]:
        if self.events is None:
            return None, "DFO event list unavailable"
        as_of = min(today, self.events_until + pd.Timedelta(days=1))
        note = ""
        if as_of < today:
            note = (f"Recent flood record uses the latest 5 years in the DFO archive "
                    f"(to {self.events_until.date()}).")
        return float(count_prior_events(self.events, lat, lon, as_of)), note

    def assess(self, points: list[tuple[str, float, float]], incidents: pd.DataFrame | None = None) -> list[LocationRisk | None]:
        snaps = self.weather.get_many([(la, lo) for _, la, lo in points])
        now = datetime.now(timezone.utc)
        out: list[LocationRisk | None] = []
        for (name, la, lo), snap in zip(points, snaps):
            if snap is None:
                out.append(None)
                continue
            today = pd.Timestamp(snap.today)
            prior, prior_note = self.prior_events(la, lo, today)
            feats, missing = live_feature_vector(
                LiveFeatureInput(
                    past_daily_mm=snap.past_daily,
                    forecast_daily_mm=snap.forecast_daily,
                    day_of_year=int(today.dayofyear),
                    elevation_m=snap.elevation_m,
                    prior_events_5y=prior,
                )
            )
            pred = self.predictor.predict(feats)
            mult, n_c, n_p = local_adjustment(incidents, la, lo, now)
            adj = apply_odds_multiplier(pred.probability, mult)
            notes = [n for n in [prior_note] if n]
            if snap.stale:
                notes.append(f"Weather data is stale (last update {snap.fetched_at}).")
            out.append(
                LocationRisk(
                    location_name=name, latitude=la, longitude=lo,
                    model_probability=pred.probability, risk_probability=adj,
                    risk_category=risk_category(adj), odds_multiplier=mult,
                    corroborated_nearby=n_c, pending_nearby=n_p,
                    prediction=pred, weather=snap, notes=notes,
                )
            )
        return out

    @staticmethod
    def to_rows(risks: list[LocationRisk]) -> list[dict]:
        rows = []
        for r in risks:
            if r is None:
                continue
            rows.append(
                {
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "location_name": r.location_name,
                    "risk_probability": r.risk_probability,
                    "model_probability": r.model_probability,
                    "risk_category": r.risk_category,
                    "prediction_window": r.prediction.window,
                    "contributing_factors": r.prediction.contributions[:5],
                    "model_version": r.prediction.model_version,
                    "weather_observed_at": r.weather.observed_at,
                }
            )
        return rows
