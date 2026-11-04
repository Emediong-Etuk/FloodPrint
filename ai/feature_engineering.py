"""Feature engineering shared by risk-model training and live inference.

The model works on *daily* rainfall. For an issue day ``t``:

- past features use rainfall observed on days ``<= t``;
- forecast features use rainfall on days ``t+1 .. t+7``. In training these come
  from the *observed* station record (a "perfect forecast" hindcast). Live
  predictions use the Open-Meteo forecast, which has its own error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_30d",
    "max_daily_7d",
    "wet_days_7d",
    "fcst_3d",
    "fcst_7d",
    "fcst_max_daily_7d",
    "doy_sin",
    "doy_cos",
    "elevation_m",
    "prior_events_5y",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "rain_1d": "Rainfall today (mm)",
    "rain_3d": "Rainfall, last 3 days (mm)",
    "rain_7d": "Rainfall, last 7 days (mm)",
    "rain_14d": "Rainfall, last 14 days (mm)",
    "rain_30d": "Rainfall, last 30 days (mm)",
    "max_daily_7d": "Heaviest single day, last 7 days (mm)",
    "wet_days_7d": "Wet days (>= 1 mm), last 7 days",
    "fcst_3d": "Forecast rainfall, next 3 days (mm)",
    "fcst_7d": "Forecast rainfall, next 7 days (mm)",
    "fcst_max_daily_7d": "Heaviest forecast day, next 7 days (mm)",
    "doy_sin": "Season (day-of-year, sine)",
    "doy_cos": "Season (day-of-year, cosine)",
    "elevation_m": "Ground elevation (m)",
    "prior_events_5y": "Large recorded floods nearby, previous 5 years",
}

WET_DAY_MM = 1.0
PREDICTION_WINDOW_DAYS = 7
PREDICTION_WINDOW_LABEL = "next 7 days"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def event_radius_km(area_sq_km: float | None) -> float:
    """Radius of influence for a DFO event: equal-area circle, clipped to 50-300 km."""
    if area_sq_km is None or not np.isfinite(area_sq_km) or area_sq_km <= 0:
        return 50.0
    return float(np.clip(math.sqrt(area_sq_km / math.pi), 50.0, 300.0))


def season_features(day_of_year: int | np.ndarray) -> tuple:
    angle = 2 * np.pi * (np.asarray(day_of_year) - 1) / 365.25
    return np.sin(angle), np.cos(angle)


def rolling_past(rain: pd.Series, window: int, min_frac: float = 0.8) -> pd.Series:
    return rain.rolling(window, min_periods=max(1, int(math.ceil(window * min_frac)))).sum()


def build_daily_features(rain: pd.Series) -> pd.DataFrame:
    """Rainfall features for every day of a gap-aware daily series.

    ``rain`` must be indexed by a complete daily ``DatetimeIndex`` (missing days
    as NaN). Windows with more than 20% missing days become NaN.
    """
    rain = rain.astype(float).asfreq("D")
    out = pd.DataFrame(index=rain.index)
    out["rain_1d"] = rain
    for w in (3, 7, 14, 30):
        out[f"rain_{w}d"] = rolling_past(rain, w)
    out["max_daily_7d"] = rain.rolling(7, min_periods=6).max()
    out["wet_days_7d"] = (rain >= WET_DAY_MM).astype(float).where(rain.notna()).rolling(7, min_periods=6).sum()
    # Future windows: reverse, roll, reverse back, then shift so day t sees t+1..t+k.
    rev = rain[::-1]
    f3 = rev.rolling(3, min_periods=3).sum()[::-1].shift(-1)
    f7 = rev.rolling(7, min_periods=6).sum()[::-1].shift(-1)
    fmax = rev.rolling(7, min_periods=6).max()[::-1].shift(-1)
    out["fcst_3d"] = f3
    out["fcst_7d"] = f7
    out["fcst_max_daily_7d"] = fmax
    s, c = season_features(out.index.dayofyear.values)
    out["doy_sin"] = s
    out["doy_cos"] = c
    return out


@dataclass
class LiveFeatureInput:
    """Inputs gathered at prediction time for one location."""

    past_daily_mm: list[float]  # oldest -> newest, last element is "today"
    forecast_daily_mm: list[float]  # tomorrow onward
    day_of_year: int
    elevation_m: float | None
    prior_events_5y: float | None


def live_feature_vector(inp: LiveFeatureInput) -> tuple[dict[str, float], list[str]]:
    """Build one feature row from live weather. Returns (features, missing_names)."""
    past = np.array([np.nan if v is None else float(v) for v in inp.past_daily_mm], dtype=float)
    fut = np.array([np.nan if v is None else float(v) for v in inp.forecast_daily_mm], dtype=float)
    feats: dict[str, float] = {}

    def tail_sum(n: int) -> float:
        if len(past) < n:
            return np.nan
        seg = past[-n:]
        if np.isnan(seg).mean() > 0.2:
            return np.nan
        return float(np.nansum(seg))

    feats["rain_1d"] = float(past[-1]) if len(past) else np.nan
    for w in (3, 7, 14, 30):
        feats[f"rain_{w}d"] = tail_sum(w)
    last7 = past[-7:] if len(past) >= 7 else np.array([])
    feats["max_daily_7d"] = float(np.nanmax(last7)) if len(last7) and not np.isnan(last7).all() else np.nan
    feats["wet_days_7d"] = float(np.sum(last7 >= WET_DAY_MM)) if len(last7) else np.nan
    f3, f7 = fut[:3], fut[:7]
    feats["fcst_3d"] = float(np.nansum(f3)) if len(f3) == 3 and not np.isnan(f3).any() else np.nan
    feats["fcst_7d"] = float(np.nansum(f7)) if len(f7) >= 6 and np.isnan(f7).sum() <= 1 else np.nan
    feats["fcst_max_daily_7d"] = float(np.nanmax(f7)) if len(f7) and not np.isnan(f7).all() else np.nan
    s, c = season_features(inp.day_of_year)
    feats["doy_sin"], feats["doy_cos"] = float(s), float(c)
    feats["elevation_m"] = np.nan if inp.elevation_m is None else float(inp.elevation_m)
    feats["prior_events_5y"] = np.nan if inp.prior_events_5y is None else float(inp.prior_events_5y)
    missing = [k for k in FEATURES if not np.isfinite(feats.get(k, np.nan))]
    return feats, missing


def count_prior_events(
    events: pd.DataFrame, lat: float, lon: float, as_of: pd.Timestamp, years: int = 5
) -> int:
    """DFO events whose influence radius covers (lat, lon) and began in the prior ``years``."""
    if events is None or events.empty:
        return 0
    start = as_of - pd.DateOffset(years=years)
    sub = events[(events["began"] >= start) & (events["began"] < as_of)]
    if sub.empty:
        return 0
    d = haversine_km(lat, lon, sub["lat"].values, sub["lon"].values)
    return int(np.sum(d <= sub["radius_km"].values))


def load_dfo_events(path) -> pd.DataFrame:
    """Normalise the DFO FloodArchive export to began/ended/lat/lon/radius columns."""
    raw = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    ev = pd.DataFrame(
        {
            "dfo_id": raw["ID"],
            "country": raw["Country"],
            "began": pd.to_datetime(raw["Began"], errors="coerce"),
            "ended": pd.to_datetime(raw["Ended"], errors="coerce"),
            "lat": pd.to_numeric(raw["lat"], errors="coerce"),
            "lon": pd.to_numeric(raw["long"], errors="coerce"),
            "area_sq_km": pd.to_numeric(raw["Area"], errors="coerce"),
            "severity": pd.to_numeric(raw["Severity"], errors="coerce"),
            "main_cause": raw["MainCause"],
        }
    ).dropna(subset=["began", "lat", "lon"])
    ev["ended"] = ev["ended"].fillna(ev["began"])
    ev["radius_km"] = ev["area_sq_km"].apply(event_radius_km)
    return ev.reset_index(drop=True)
