"""Open-Meteo weather client with validation, caching and stale fallback.

API: https://open-meteo.com/en/docs (no key required). All precipitation is
requested and stored in millimetres, temperature in degrees Celsius, and times
in the location's local timezone (ISO 8601).

If the API fails the service returns the last successful observation stored in
the database, flagged ``stale=True`` with its original timestamp. It never
invents weather values.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PAST_DAYS = 31
FORECAST_DAYS = 8  # today + 7

CURRENT_VARS = "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m"
HOURLY_VARS = "precipitation,precipitation_probability"
DAILY_VARS = "precipitation_sum,precipitation_probability_max"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherDataError(RuntimeError):
    """The API response was missing or malformed."""


class WeatherUnavailableError(RuntimeError):
    """No fresh data and no stored observation to fall back on."""


def validate_coordinates(lat: float, lon: float) -> tuple[float, float]:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be numbers")
    if not (np.isfinite(lat) and np.isfinite(lon)) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: ({lat}, {lon})")
    return lat, lon


@dataclass
class WeatherSnapshot:
    latitude: float
    longitude: float
    elevation_m: float | None
    timezone: str
    observed_at: str  # local time of the "current" block
    fetched_at: str  # UTC time the data was retrieved
    temperature_c: float | None
    humidity_pct: float | None
    wind_kmh: float | None
    precipitation_now_mm: float | None
    weather_code: int | None
    hourly: pd.DataFrame  # time, precipitation_mm, precipitation_probability
    daily: pd.DataFrame  # date, precipitation_mm, precipitation_probability_max
    today: str
    source: str = "Open-Meteo forecast API"
    stale: bool = False
    stale_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def description(self) -> str:
        return WMO_CODES.get(self.weather_code, "Unknown") if self.weather_code is not None else "Unknown"

    def _now_local(self) -> pd.Timestamp:
        return pd.Timestamp(self.observed_at)

    @property
    def past_daily(self) -> list[float | None]:
        """Daily totals from 30 days ago up to and including today (oldest first)."""
        d = self.daily[self.daily["date"] <= self.today]
        return [None if pd.isna(v) else float(v) for v in d["precipitation_mm"].tolist()]

    @property
    def forecast_daily(self) -> list[float | None]:
        d = self.daily[self.daily["date"] > self.today]
        return [None if pd.isna(v) else float(v) for v in d["precipitation_mm"].tolist()]

    def hourly_window(self, hours_back: int = 0, hours_ahead: int = 0) -> pd.DataFrame:
        now = self._now_local()
        h = self.hourly
        return h[(h["time"] > now - pd.Timedelta(hours=hours_back)) & (h["time"] <= now + pd.Timedelta(hours=hours_ahead))]

    @property
    def rain_last_24h_mm(self) -> float | None:
        h = self.hourly
        now = self._now_local()
        seg = h[(h["time"] > now - pd.Timedelta(hours=24)) & (h["time"] <= now)]["precipitation_mm"]
        return None if seg.empty or seg.isna().all() else float(seg.sum())

    @property
    def forecast_next_24h_mm(self) -> float | None:
        h = self.hourly
        now = self._now_local()
        seg = h[(h["time"] > now) & (h["time"] <= now + pd.Timedelta(hours=24))]["precipitation_mm"]
        return None if seg.empty or seg.isna().all() else float(seg.sum())

    @property
    def max_intensity_next_24h(self) -> float | None:
        h = self.hourly
        now = self._now_local()
        seg = h[(h["time"] > now) & (h["time"] <= now + pd.Timedelta(hours=24))]["precipitation_mm"]
        return None if seg.empty or seg.isna().all() else float(seg.max())

    @property
    def forecast_7d_mm(self) -> float | None:
        f = [v for v in self.forecast_daily[:7] if v is not None]
        return float(sum(f)) if f else None

    def context_summary(self) -> dict:
        """Compact rainfall context saved alongside incident reports."""
        past = self.past_daily
        return {
            "source": self.source,
            "observed_at": self.observed_at,
            "fetched_at": self.fetched_at,
            "stale": self.stale,
            "rain_last_24h_mm": self.rain_last_24h_mm,
            "rain_last_3d_mm": float(np.nansum([v for v in past[-3:] if v is not None])) if past else None,
            "rain_last_7d_mm": float(np.nansum([v for v in past[-7:] if v is not None])) if past else None,
            "forecast_next_24h_mm": self.forecast_next_24h_mm,
            "weather": self.description,
        }


def parse_response(payload: dict, fetched_at: str | None = None) -> WeatherSnapshot:
    """Validate and normalise one Open-Meteo location response."""
    if not isinstance(payload, dict):
        raise WeatherDataError("Response is not a JSON object")
    if payload.get("error"):
        raise WeatherDataError(f"API error: {payload.get('reason', 'unknown')}")
    for key in ("latitude", "longitude", "current", "hourly", "daily"):
        if key not in payload:
            raise WeatherDataError(f"Response missing '{key}'")
    units = payload.get("hourly_units", {})
    if units.get("precipitation") not in (None, "mm"):
        raise WeatherDataError(f"Unexpected precipitation unit: {units.get('precipitation')}")
    hourly, daily, cur = payload["hourly"], payload["daily"], payload["current"]
    if "time" not in hourly or "precipitation" not in hourly:
        raise WeatherDataError("Hourly precipitation missing")
    if len(hourly["time"]) != len(hourly["precipitation"]):
        raise WeatherDataError("Hourly arrays have mismatched lengths")
    if "time" not in daily or "precipitation_sum" not in daily:
        raise WeatherDataError("Daily precipitation missing")
    if "time" not in cur:
        raise WeatherDataError("Current block missing time")

    def arr(block, key, n):
        v = block.get(key)
        return [None] * n if v is None else v

    nh = len(hourly["time"])
    h = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "precipitation_mm": pd.to_numeric(pd.Series(hourly["precipitation"]), errors="coerce"),
            "precipitation_probability": pd.to_numeric(pd.Series(arr(hourly, "precipitation_probability", nh)), errors="coerce"),
        }
    )
    nd = len(daily["time"])
    d = pd.DataFrame(
        {
            "date": [str(x) for x in daily["time"]],
            "precipitation_mm": pd.to_numeric(pd.Series(daily["precipitation_sum"]), errors="coerce"),
            "precipitation_probability_max": pd.to_numeric(
                pd.Series(arr(daily, "precipitation_probability_max", nd)), errors="coerce"),
        }
    )
    if (h["precipitation_mm"].dropna() < 0).any() or (d["precipitation_mm"].dropna() < 0).any():
        raise WeatherDataError("Negative precipitation values")

    def num(x):
        try:
            return None if x is None else float(x)
        except (TypeError, ValueError):
            return None

    code = cur.get("weather_code")
    return WeatherSnapshot(
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        elevation_m=num(payload.get("elevation")),
        timezone=payload.get("timezone", "UTC"),
        observed_at=str(cur["time"]),
        fetched_at=fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        temperature_c=num(cur.get("temperature_2m")),
        humidity_pct=num(cur.get("relative_humidity_2m")),
        wind_kmh=num(cur.get("wind_speed_10m")),
        precipitation_now_mm=num(cur.get("precipitation")),
        weather_code=int(code) if code is not None else None,
        hourly=h,
        daily=d,
        today=str(cur["time"])[:10],
        raw=payload,
    )


class WeatherService:
    def __init__(
        self,
        conn_factory: Callable | None = None,
        ttl_s: int = 900,
        timeout_s: float = 10.0,
        session: requests.Session | None = None,
        store_interval_s: int = 3600,
    ):
        self.conn_factory = conn_factory
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.store_interval_s = store_interval_s
        self._cache: dict[tuple[float, float], tuple[float, WeatherSnapshot]] = {}
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @staticmethod
    def _key(lat: float, lon: float) -> tuple[float, float]:
        return (round(lat, 3), round(lon, 3))

    def _params(self, lats: list[float], lons: list[float]) -> dict:
        return {
            "latitude": ",".join(f"{v:.4f}" for v in lats),
            "longitude": ",".join(f"{v:.4f}" for v in lons),
            "current": CURRENT_VARS,
            "hourly": HOURLY_VARS,
            "daily": DAILY_VARS,
            "past_days": PAST_DAYS,
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
            "precipitation_unit": "mm",
            "temperature_unit": "celsius",
        }

    def _fetch(self, lats: list[float], lons: list[float]) -> list[dict]:
        r = self.session.get(FORECAST_URL, params=self._params(lats, lons), timeout=self.timeout_s)
        if r.status_code != 200:
            reason = ""
            try:
                reason = r.json().get("reason", "")
            except Exception:
                pass
            raise WeatherDataError(f"HTTP {r.status_code} {reason}".strip())
        data = r.json()
        return data if isinstance(data, list) else [data]

    def get_many(self, points: list[tuple[float, float]]) -> list[WeatherSnapshot | None]:
        """Weather for several points in one request. Falls back per point."""
        pts = [validate_coordinates(*p) for p in points]
        now = time.time()
        results: list[WeatherSnapshot | None] = [None] * len(pts)
        todo = []
        with self._lock:
            for i, (la, lo) in enumerate(pts):
                hit = self._cache.get(self._key(la, lo))
                if hit and now - hit[0] < self.ttl_s:
                    results[i] = hit[1]
                else:
                    todo.append(i)
        if todo:
            try:
                payloads = self._fetch([pts[i][0] for i in todo], [pts[i][1] for i in todo])
                if len(payloads) != len(todo):
                    raise WeatherDataError("Response count does not match requested locations")
                fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                for i, pl in zip(todo, payloads):
                    snap = parse_response(pl, fetched_at)
                    snap.latitude, snap.longitude = pts[i]  # keep requested point as the key
                    results[i] = snap
                    with self._lock:
                        self._cache[self._key(*pts[i])] = (now, snap)
                    self._store(snap)
                self.last_error = None
            except (requests.RequestException, WeatherDataError, ValueError) as exc:
                self.last_error = str(exc)
                log.warning("Open-Meteo request failed: %s", exc)
                for i in todo:
                    results[i] = self._fallback(*pts[i], reason=str(exc))
        return results

    def get(self, lat: float, lon: float) -> WeatherSnapshot:
        snap = self.get_many([(lat, lon)])[0]
        if snap is None:
            raise WeatherUnavailableError(self.last_error or "Weather data unavailable")
        return snap

    # ------------------------------------------------------- persistence
    def _store(self, snap: WeatherSnapshot) -> None:
        if not self.conn_factory:
            return
        from database import repository as repo

        try:
            conn = self.conn_factory()
            last = repo.latest_weather_observation(conn, snap.latitude, snap.longitude, tolerance=0.0005)
            if last:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(last["payload"]["_fetched_at"])).total_seconds() \
                    if last.get("payload") and last["payload"].get("_fetched_at") else 1e9
                if age < self.store_interval_s:
                    return
            payload = dict(snap.raw)
            payload["_fetched_at"] = snap.fetched_at
            repo.insert_weather_observation(
                conn,
                {
                    "latitude": snap.latitude,
                    "longitude": snap.longitude,
                    "temperature": snap.temperature_c,
                    "precipitation": snap.rain_last_24h_mm,
                    "rainfall_intensity": snap.max_intensity_next_24h,
                    "forecast_precipitation": snap.forecast_7d_mm,
                    "weather_description": snap.description,
                    "payload": payload,
                    "source": "open-meteo",
                    "observed_at": snap.observed_at,
                },
            )
        except Exception as exc:  # storage must never break the live view
            log.warning("Could not store weather observation: %s", exc)

    def _fallback(self, lat: float, lon: float, reason: str) -> WeatherSnapshot | None:
        if not self.conn_factory:
            return None
        from database import repository as repo

        try:
            last = repo.latest_weather_observation(self.conn_factory(), lat, lon)
        except Exception:
            return None
        if not last or not last.get("payload"):
            return None
        try:
            snap = parse_response(last["payload"], last["payload"].get("_fetched_at"))
        except WeatherDataError:
            return None
        snap.stale = True
        snap.stale_reason = reason
        return snap
