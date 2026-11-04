"""Emerging flood hotspot detection with DBSCAN on haversine distance.

Two scopes are produced so that confirmed clusters are never confused with
clusters that rely on unverified reports:

- ``corroborated``: only CORROBORATED incidents;
- ``all_reports``: CORROBORATED + PENDING (REJECTED are always excluded).

A single report can never form a hotspot: ``min_samples`` (default 3) sets the
minimum cluster size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai.feature_engineering import haversine_km
from database.models import SEVERITY_SCORE

EARTH_RADIUS_M = 6_371_008.8


@dataclass
class Hotspot:
    hotspot_id: str
    scope: str
    center_latitude: float
    center_longitude: float
    radius_m: float
    incident_count: int
    unique_locations: int
    average_severity: float
    verification_count: int
    pending_count: int
    incident_ids: list[int]
    last_incident_at: str | None
    first_incident_at: str | None
    severity_counts: dict[str, int]
    cluster_generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or impossible coordinates."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = lat.between(-90, 90) & lon.between(-180, 180)
    return df.loc[ok].assign(latitude=lat[ok], longitude=lon[ok])


def detect_hotspots(
    incidents: pd.DataFrame,
    eps_m: float = 450.0,
    min_samples: int = 3,
    scope: str = "all_reports",
) -> list[Hotspot]:
    """Cluster incidents. ``incidents`` needs id, latitude, longitude,
    verification_status, user_reported_severity, reported_at, location_name."""
    from sklearn.cluster import DBSCAN

    if scope not in {"corroborated", "all_reports"}:
        raise ValueError("scope must be 'corroborated' or 'all_reports'")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2 so single reports never form a hotspot")
    if incidents is None or len(incidents) == 0:
        return []
    df = validate_coordinates(incidents.copy())
    allowed = {"CORROBORATED"} if scope == "corroborated" else {"CORROBORATED", "PENDING"}
    df = df[df["verification_status"].isin(allowed)]
    if len(df) < min_samples:
        return []

    coords = np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))
    labels = DBSCAN(
        eps=eps_m / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine", algorithm="ball_tree"
    ).fit_predict(coords)
    df = df.assign(cluster=labels)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out: list[Hotspot] = []
    for k, g in df[df["cluster"] >= 0].groupby("cluster"):
        # Spherical mean of the member points
        la, lo = np.radians(g["latitude"].values), np.radians(g["longitude"].values)
        x, y, z = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
        cx, cy, cz = x.mean(), y.mean(), z.mean()
        c_lat = float(np.degrees(np.arctan2(cz, np.hypot(cx, cy))))
        c_lon = float(np.degrees(np.arctan2(cy, cx)))
        dists = haversine_km(c_lat, c_lon, g["latitude"].values, g["longitude"].values) * 1000
        sev = g["user_reported_severity"].map(SEVERITY_SCORE).astype(float)
        times = pd.to_datetime(g["reported_at"], utc=True, format="ISO8601")
        uniq = g.assign(lr=g["latitude"].round(3), lor=g["longitude"].round(3))[["lr", "lor"]].drop_duplicates()
        out.append(
            Hotspot(
                hotspot_id=f"{'C' if scope == 'corroborated' else 'A'}{int(k) + 1:02d}",
                scope=scope,
                center_latitude=c_lat,
                center_longitude=c_lon,
                radius_m=float(max(dists.max(), 75.0)),
                incident_count=int(len(g)),
                unique_locations=int(len(uniq)),
                average_severity=float(sev.mean()),
                verification_count=int((g["verification_status"] == "CORROBORATED").sum()),
                pending_count=int((g["verification_status"] == "PENDING").sum()),
                incident_ids=[int(i) for i in g["id"]],
                last_incident_at=times.max().isoformat() if len(times) else None,
                first_incident_at=times.min().isoformat() if len(times) else None,
                severity_counts={s: int((g["user_reported_severity"] == s).sum()) for s in SEVERITY_SCORE},
                cluster_generated_at=now,
            )
        )
    out.sort(key=lambda h: h.incident_count, reverse=True)
    return out
