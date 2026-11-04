"""Clearly labelled demonstration data.

- Every demo incident has ``is_demo = 1`` and a description starting "[DEMO]".
- Demo incidents have no photo and **no AI classification** (AI output is never
  fabricated).
- Historical demo incidents are placed on days when the Port Harcourt
  International NOAA GSOD station recorded heavy rain, and carry that real
  station rainfall as context. The incidents themselves are synthetic.
- The station's daily rainfall is also loaded into ``weather_observations``
  with ``source = 'noaa-gsod'`` (real data, not demo).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import SAMPLE_DIR, Region
from database import repository as repo
from database.models import Incident

SEED = 2026
GSOD_FILE = SAMPLE_DIR / "port_harcourt_gsod_daily.csv"

# Neighbourhoods given extra demo reports so that clusters exist to demonstrate DBSCAN.
CLUSTER_WEIGHTS = {
    "Mile 1 Diobu": 6.0,
    "D-Line": 4.0,
    "Rumuola": 4.0,
    "Trans-Amadi": 3.0,
    "Borokiri": 3.0,
    "Rumuokoro": 2.0,
}

DESCRIPTIONS = {
    "minor": ["Standing water at the roadside after rain", "Drainage blocked, water pooling on the road"],
    "moderate": ["Water covering most of the road, cars slowing down", "Shops at the junction have water at the entrance"],
    "severe": ["Road impassable, water above car tyres", "Houses on the street flooded, residents moving belongings"],
}
TYPES = {
    "minor": "Blocked drainage / waterlogging",
    "moderate": "Road flooding",
    "severe": "Residential flooding",
}


def _jitter(rng, lat, lon, metres):
    dlat = rng.normal(0, metres) / 111_320
    dlon = rng.normal(0, metres) / (111_320 * np.cos(np.radians(lat)))
    return float(lat + dlat), float(lon + dlon)


def load_station_rain() -> pd.DataFrame:
    if not GSOD_FILE.exists():
        return pd.DataFrame(columns=["date", "precipitation_mm"])
    d = pd.read_csv(GSOD_FILE, parse_dates=["date"])
    return d.dropna(subset=["precipitation_mm"])


def seed_demo_data(conn: sqlite3.Connection, region: Region, n_historical: int = 80, n_recent: int = 14) -> dict:
    rng = np.random.default_rng(SEED)
    rain = load_station_rain()
    now = datetime.now(timezone.utc).replace(microsecond=0)

    names = [l.name for l in region.locations]
    weights = np.array([CLUSTER_WEIGHTS.get(n, 1.0) for n in names])
    weights = weights / weights.sum()
    loc_by_name = {l.name: l for l in region.locations}

    heavy = rain[rain["precipitation_mm"] >= 30].reset_index(drop=True)
    created = 0
    for i in range(n_historical):
        if heavy.empty:
            break
        day = heavy.iloc[int(rng.integers(0, len(heavy)))]
        mm = float(day["precipitation_mm"])
        loc = loc_by_name[names[int(rng.choice(len(names), p=weights))]]
        lat, lon = _jitter(rng, loc.lat, loc.lon, 180 if loc.name in CLUSTER_WEIGHTS else 350)
        sev = "severe" if mm >= 70 and rng.random() < 0.6 else "moderate" if mm >= 40 or rng.random() < 0.4 else "minor"
        status = rng.choice(["CORROBORATED", "PENDING", "REJECTED"], p=[0.62, 0.28, 0.10])
        prev = rain[(rain["date"] < day["date"]) & (rain["date"] >= day["date"] - pd.Timedelta(days=3))]
        ts = pd.Timestamp(day["date"]).tz_localize("Africa/Lagos") + pd.Timedelta(hours=int(rng.integers(7, 20)))
        inc = Incident(
            latitude=lat, longitude=lon, location_name=loc.name,
            description=f"[DEMO] {rng.choice(DESCRIPTIONS[sev])}",
            incident_type=TYPES[sev], user_reported_severity=sev,
            reported_at=ts.tz_convert("UTC").isoformat(),
            verification_status=str(status),
            verification_note="Demonstration record" if status != "PENDING" else None,
            verified_at=now.isoformat() if status != "PENDING" else None,
            quality_flags=["demo_record"],
            rainfall_context={
                "source": "NOAA GSOD station 65250 (Port Harcourt Intl) - real station data",
                "observed_at": str(day["date"].date()),
                "station_rain_day_mm": round(mm, 1),
                "station_rain_prev_3d_mm": round(float(prev["precipitation_mm"].sum()), 1),
            },
            is_demo=True,
        )
        repo.insert_incident(conn, inc)
        created += 1

    for i in range(n_recent):
        loc = loc_by_name[names[int(rng.choice(len(names), p=weights))]]
        lat, lon = _jitter(rng, loc.lat, loc.lon, 160)
        sev = str(rng.choice(["minor", "moderate", "severe"], p=[0.35, 0.45, 0.20]))
        ts = now - timedelta(hours=float(rng.uniform(1, 96)))
        status = str(rng.choice(["PENDING", "CORROBORATED"], p=[0.6, 0.4]))
        inc = Incident(
            latitude=lat, longitude=lon, location_name=loc.name,
            description=f"[DEMO] {rng.choice(DESCRIPTIONS[sev])}",
            incident_type=TYPES[sev], user_reported_severity=sev,
            reported_at=ts.isoformat(), verification_status=status,
            verified_at=now.isoformat() if status != "PENDING" else None,
            quality_flags=["demo_record"],
            rainfall_context={"source": "Not recorded for demonstration records"},
            is_demo=True,
        )
        repo.insert_incident(conn, inc)
        created += 1

    # Real station rainfall (not demo) for historical rainfall charts.
    existing = conn.execute("SELECT COUNT(*) FROM weather_observations WHERE source = 'noaa-gsod'").fetchone()[0]
    station = region.reference_station or {}
    n_obs = 0
    if not existing and not rain.empty and station:
        conn.executemany(
            "INSERT INTO weather_observations (latitude, longitude, precipitation, weather_description, source, observed_at) "
            "VALUES (?,?,?,?,?,?)",
            [(station["lat"], station["lon"], float(r.precipitation_mm), "Daily total (station)", "noaa-gsod",
              r.date.date().isoformat()) for r in rain.itertuples()],
        )
        conn.commit()
        n_obs = len(rain)
    return {"incidents": created, "station_days": n_obs}


def has_demo_data(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM incidents WHERE is_demo = 1 LIMIT 1").fetchone() is not None
