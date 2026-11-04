"""Incident submission, quality checks, duplicate detection and evidence scoring."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ai.feature_engineering import haversine_km
from config import Region
from database import repository as repo
from database.models import INCIDENT_TYPES, SEVERITY_SCORE, Incident, VerificationStatus
from services.image_service import hash_distance

MAX_DESCRIPTION = 1000
MAX_LOCATION_NAME = 120
FUTURE_TOLERANCE = timedelta(minutes=10)
MAX_REPORT_AGE = timedelta(days=7)
PHASH_DUPLICATE_DISTANCE = 8
EXIF_MISMATCH_KM = 2.0


class IncidentValidationError(ValueError):
    pass


@dataclass
class SubmissionResult:
    incident_id: int
    flags: list[str] = field(default_factory=list)
    duplicate_of: int | None = None
    nearby_reports: int = 0


def parse_time(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_submission(
    region: Region,
    latitude: float,
    longitude: float,
    location_name: str,
    description: str,
    severity: str,
    incident_type: str,
    reported_at: datetime,
    now: datetime | None = None,
) -> list[str]:
    """Raise on invalid input; return soft quality flags for review."""
    now = now or datetime.now(timezone.utc)
    flags: list[str] = []
    try:
        latitude, longitude = float(latitude), float(longitude)
    except (TypeError, ValueError):
        raise IncidentValidationError("Latitude and longitude must be numbers.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180) or not np.isfinite([latitude, longitude]).all():
        raise IncidentValidationError("Coordinates are outside the valid range.")
    if not region.contains(latitude, longitude, margin=0.05):
        raise IncidentValidationError(
            f"Location is outside the monitored area ({region.display_name})."
        )
    if not location_name or not location_name.strip():
        raise IncidentValidationError("Please give a street or neighbourhood name.")
    if len(location_name) > MAX_LOCATION_NAME:
        raise IncidentValidationError("Location name is too long.")
    if len(description or "") > MAX_DESCRIPTION:
        raise IncidentValidationError(f"Description must be under {MAX_DESCRIPTION} characters.")
    if severity not in SEVERITY_SCORE:
        raise IncidentValidationError("Unknown severity.")
    if incident_type not in INCIDENT_TYPES:
        raise IncidentValidationError("Unknown incident type.")
    rt = parse_time(reported_at)
    if rt > now + FUTURE_TOLERANCE:
        raise IncidentValidationError("Observation time cannot be in the future.")
    if rt < now - MAX_REPORT_AGE:
        raise IncidentValidationError("Observations older than 7 days cannot be submitted as current reports.")
    if not region.contains(latitude, longitude):
        flags.append("near_region_edge")
    if len((description or "").strip()) < 10:
        flags.append("short_description")
    return flags


def find_duplicates(
    conn: sqlite3.Connection,
    latitude: float,
    longitude: float,
    reported_at: datetime,
    image_hash: str | None,
    radius_m: float,
    window_h: float,
) -> tuple[int | None, list[Incident]]:
    """Return (likely duplicate id, nearby reports in the time window)."""
    rt = parse_time(reported_at)
    since = (rt - timedelta(hours=window_h)).isoformat()
    until = (rt + timedelta(hours=window_h)).isoformat()
    candidates = repo.list_incidents(conn, since=since, until=until)
    nearby, dup = [], None
    for c in candidates:
        if c.verification_status == VerificationStatus.REJECTED.value:
            continue
        d_m = float(haversine_km(latitude, longitude, c.latitude, c.longitude)) * 1000
        if d_m <= radius_m:
            nearby.append(c)
            dist = hash_distance(image_hash, c.image_hash)
            if dist is not None and dist <= PHASH_DUPLICATE_DISTANCE and dup is None:
                dup = c.id
    # Same photo anywhere in the last 30 days is also suspicious.
    if dup is None and image_hash:
        recent = repo.list_incidents(conn, since=(rt - timedelta(days=30)).isoformat())
        for c in recent:
            dist = hash_distance(image_hash, c.image_hash)
            if dist is not None and dist <= PHASH_DUPLICATE_DISTANCE:
                dup = c.id
                break
    return dup, nearby


def submit_incident(
    conn: sqlite3.Connection,
    region: Region,
    *,
    latitude: float,
    longitude: float,
    location_name: str,
    description: str,
    severity: str,
    incident_type: str,
    reported_at: datetime,
    image_path: str | None = None,
    image_hash: str | None = None,
    image_gps: tuple[float, float] | None = None,
    vision: dict | None = None,
    rainfall_context: dict | None = None,
    duplicate_radius_m: float = 150,
    duplicate_window_h: float = 3,
) -> SubmissionResult:
    flags = validate_submission(region, latitude, longitude, location_name, description,
                                severity, incident_type, reported_at)
    if image_gps:
        d = float(haversine_km(latitude, longitude, image_gps[0], image_gps[1]))
        if d > EXIF_MISMATCH_KM:
            flags.append(f"photo_gps_{d:.1f}km_from_reported_location")
    if not image_path:
        flags.append("no_photo")
    dup, nearby = find_duplicates(conn, latitude, longitude, reported_at, image_hash,
                                  duplicate_radius_m, duplicate_window_h)
    if dup is not None:
        flags.append("possible_duplicate")
    if vision and vision.get("analysis_status") == "completed":
        ai_sev = vision.get("severity")
        if ai_sev is not None and abs(SEVERITY_SCORE[ai_sev] - SEVERITY_SCORE[severity]) >= 2:
            flags.append("ai_user_severity_disagree")
    if rainfall_context and rainfall_context.get("rain_last_24h_mm") is not None \
            and rainfall_context["rain_last_24h_mm"] < 1 and (rainfall_context.get("rain_last_3d_mm") or 0) < 5:
        flags.append("little_recent_rain")

    inc = Incident(
        latitude=float(latitude),
        longitude=float(longitude),
        location_name=location_name.strip(),
        description=(description or "").strip(),
        incident_type=incident_type,
        user_reported_severity=severity,
        reported_at=parse_time(reported_at).replace(microsecond=0).isoformat(),
        image_path=image_path,
        image_hash=image_hash,
        ai_classification=(vision or {}).get("classification"),
        ai_confidence=(vision or {}).get("confidence"),
        ai_details=vision,
        duplicate_of=dup,
        quality_flags=flags,
        rainfall_context=rainfall_context,
    )
    incident_id = repo.insert_incident(conn, inc)
    return SubmissionResult(incident_id, flags, dup, len(nearby))


def evidence_score(inc: Incident, nearby_independent: int) -> dict:
    """Transparent, rule-based confidence indicator (0-100) for reviewers.

    This is a checklist score, not a learned probability. It never changes the
    verification status by itself.
    """
    parts = []

    def add(name, pts):
        parts.append({"factor": name, "points": pts})

    if inc.verification_status == VerificationStatus.CORROBORATED.value:
        add("Corroborated by reviewer", 40)
    elif inc.verification_status == VerificationStatus.REJECTED.value:
        add("Rejected by reviewer", -100)
    if inc.image_path:
        add("Photo attached", 10)
    ai = inc.ai_details or {}
    if ai.get("analysis_status") == "completed":
        if ai.get("water_detected") and inc.user_reported_severity != "none":
            add("Photo analysis sees water", 15)
        elif not ai.get("water_detected") and inc.user_reported_severity in {"moderate", "severe"}:
            add("Photo analysis sees no water", -10)
    if nearby_independent:
        add(f"{nearby_independent} independent nearby report(s)", min(25, 10 * nearby_independent))
    rc = inc.rainfall_context or {}
    if (rc.get("rain_last_24h_mm") or 0) >= 10 or (rc.get("rain_last_3d_mm") or 0) >= 25:
        add("Significant recent rainfall", 10)
    flags = set(inc.quality_flags or [])
    if "possible_duplicate" in flags:
        add("Possible duplicate", -15)
    if any(f.startswith("photo_gps_") for f in flags):
        add("Photo location mismatch", -15)
    if "little_recent_rain" in flags:
        add("Little recent rain", -5)
    score = int(np.clip(20 + sum(p["points"] for p in parts), 0, 100))
    level = "High" if score >= 70 else "Medium" if score >= 45 else "Low"
    return {"score": score, "level": level, "factors": parts}


def independent_nearby(df: pd.DataFrame, inc: Incident, radius_m: float = 300, window_h: float = 6) -> int:
    """Count other non-rejected, non-duplicate reports close in space and time."""
    if df is None or df.empty:
        return 0
    t = pd.Timestamp(parse_time(inc.reported_at))
    sub = df[(df["id"] != inc.id) & (df["verification_status"] != "REJECTED") & (df["duplicate_of"].isna())
             & ((df["reported_at"] - t).abs() <= pd.Timedelta(hours=window_h))]
    if sub.empty:
        return 0
    d = haversine_km(inc.latitude, inc.longitude, sub["latitude"].values, sub["longitude"].values) * 1000
    return int((d <= radius_m).sum())


def assign_area(df: pd.DataFrame, region: Region, max_km: float = 2.5) -> pd.Series:
    """Nearest monitored neighbourhood for each incident (or 'Other')."""
    if df.empty:
        return pd.Series(dtype=str)
    locs = region.locations
    la = np.array([l.lat for l in locs])
    lo = np.array([l.lon for l in locs])
    names = []
    for lat, lon in zip(df["latitude"].values, df["longitude"].values):
        d = haversine_km(lat, lon, la, lo)
        i = int(np.argmin(d))
        names.append(locs[i].name if d[i] <= max_km else "Other")
    return pd.Series(names, index=df.index)
