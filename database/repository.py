"""Data-access functions. All SQL lives here."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from database.models import Incident, VerificationStatus


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def _loads(value: str | None) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- incidents
def insert_incident(conn: sqlite3.Connection, inc: Incident) -> int:
    created = inc.created_at or utcnow_iso()
    cur = conn.execute(
        """
        INSERT INTO incidents (
            latitude, longitude, location_name, description, incident_type, image_path,
            image_hash, user_reported_severity, ai_classification, ai_confidence, ai_details,
            verification_status, verification_note, verified_at, duplicate_of, quality_flags,
            rainfall_context, is_demo, reported_at, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            inc.latitude, inc.longitude, inc.location_name, inc.description, inc.incident_type,
            inc.image_path, inc.image_hash, inc.user_reported_severity, inc.ai_classification,
            inc.ai_confidence, _dumps(inc.ai_details), inc.verification_status,
            inc.verification_note, inc.verified_at, inc.duplicate_of,
            _dumps(inc.quality_flags or []), _dumps(inc.rainfall_context), int(inc.is_demo),
            inc.reported_at, created,
        ),
    )
    conn.commit()
    incident_id = int(cur.lastrowid)
    if inc.verification_status == VerificationStatus.CORROBORATED.value:
        _sync_history(conn, incident_id)
    return incident_id


def _row_to_incident(row: sqlite3.Row) -> Incident:
    d = dict(row)
    return Incident(
        id=d["id"],
        latitude=d["latitude"],
        longitude=d["longitude"],
        location_name=d["location_name"],
        description=d["description"],
        incident_type=d["incident_type"],
        image_path=d["image_path"],
        image_hash=d["image_hash"],
        user_reported_severity=d["user_reported_severity"],
        ai_classification=d["ai_classification"],
        ai_confidence=d["ai_confidence"],
        ai_details=_loads(d["ai_details"]),
        verification_status=d["verification_status"],
        verification_note=d["verification_note"],
        verified_at=d["verified_at"],
        duplicate_of=d["duplicate_of"],
        quality_flags=_loads(d["quality_flags"]) or [],
        rainfall_context=_loads(d["rainfall_context"]),
        is_demo=bool(d["is_demo"]),
        reported_at=d["reported_at"],
        created_at=d["created_at"],
    )


def get_incident(conn: sqlite3.Connection, incident_id: int) -> Incident | None:
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    return _row_to_incident(row) if row else None


def list_incidents(
    conn: sqlite3.Connection,
    statuses: Iterable[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int | None = None,
) -> list[Incident]:
    sql = "SELECT * FROM incidents WHERE 1=1"
    params: list[Any] = []
    if statuses:
        statuses = list(statuses)
        sql += f" AND verification_status IN ({','.join('?' * len(statuses))})"
        params += statuses
    if since:
        sql += " AND reported_at >= ?"
        params.append(since)
    if until:
        sql += " AND reported_at <= ?"
        params.append(until)
    if bbox:
        sql += " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
        params += [bbox[0], bbox[1], bbox[2], bbox[3]]
    sql += " ORDER BY reported_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [_row_to_incident(r) for r in conn.execute(sql, params).fetchall()]


def incidents_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY reported_at DESC", conn)
    if df.empty:
        return df
    df["reported_at"] = pd.to_datetime(df["reported_at"], utc=True, format="ISO8601")
    df["is_demo"] = df["is_demo"].astype(bool)
    return df


def update_verification(
    conn: sqlite3.Connection, incident_id: int, status: str, note: str | None = None
) -> None:
    if status not in {s.value for s in VerificationStatus}:
        raise ValueError(f"Invalid verification status: {status}")
    cur = conn.execute(
        "UPDATE incidents SET verification_status = ?, verification_note = ?, verified_at = ? WHERE id = ?",
        (status, note, utcnow_iso(), incident_id),
    )
    if cur.rowcount == 0:
        raise KeyError(f"Incident {incident_id} not found")
    conn.commit()
    _sync_history(conn, incident_id)


def _sync_history(conn: sqlite3.Connection, incident_id: int) -> None:
    """Keep location_history in step with corroborated incidents."""
    inc = get_incident(conn, incident_id)
    if inc is None:
        return
    if inc.verification_status == VerificationStatus.CORROBORATED.value:
        conn.execute(
            """
            INSERT INTO location_history (location_name, latitude, longitude, incident_id,
                                          rainfall_context, severity, recorded_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(incident_id) DO UPDATE SET severity = excluded.severity,
                rainfall_context = excluded.rainfall_context
            """,
            (inc.location_name, inc.latitude, inc.longitude, inc.id,
             _dumps(inc.rainfall_context), inc.user_reported_severity, inc.reported_at),
        )
    else:
        conn.execute("DELETE FROM location_history WHERE incident_id = ?", (incident_id,))
    conn.commit()


def delete_demo_data(conn: sqlite3.Connection) -> int:
    cur = conn.execute("DELETE FROM incidents WHERE is_demo = 1")
    conn.execute("DELETE FROM weather_observations WHERE source = 'demo'")
    conn.execute("DELETE FROM hotspots")
    conn.commit()
    return cur.rowcount


def count_incidents(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT verification_status, COUNT(*) AS n FROM incidents GROUP BY verification_status"
    ).fetchall()
    out = {s.value: 0 for s in VerificationStatus}
    for r in rows:
        out[r["verification_status"]] = r["n"]
    out["TOTAL"] = sum(out.values())
    out["DEMO"] = conn.execute("SELECT COUNT(*) FROM incidents WHERE is_demo = 1").fetchone()[0]
    return out


def history_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM location_history ORDER BY recorded_at DESC", conn)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, format="ISO8601")
    return df


# ------------------------------------------------------------------ weather
def insert_weather_observation(conn: sqlite3.Connection, obs: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO weather_observations (latitude, longitude, temperature, precipitation,
            rainfall_intensity, forecast_precipitation, weather_description, payload, source, observed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            obs["latitude"], obs["longitude"], obs.get("temperature"), obs.get("precipitation"),
            obs.get("rainfall_intensity"), obs.get("forecast_precipitation"),
            obs.get("weather_description"), _dumps(obs.get("payload")),
            obs.get("source", "open-meteo"), obs["observed_at"],
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def latest_weather_observation(
    conn: sqlite3.Connection, lat: float, lon: float, tolerance: float = 0.02
) -> dict | None:
    row = conn.execute(
        """
        SELECT * FROM weather_observations
        WHERE source = 'open-meteo' AND ABS(latitude - ?) <= ? AND ABS(longitude - ?) <= ?
        ORDER BY observed_at DESC LIMIT 1
        """,
        (lat, tolerance, lon, tolerance),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["payload"] = _loads(d["payload"])
    return d


def weather_dataframe(conn: sqlite3.Connection, limit: int = 500) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id, latitude, longitude, temperature, precipitation, rainfall_intensity, "
        "forecast_precipitation, weather_description, source, observed_at "
        "FROM weather_observations ORDER BY observed_at DESC LIMIT ?",
        conn,
        params=(limit,),
    )


# ---------------------------------------------------------- risk predictions
def insert_risk_predictions(conn: sqlite3.Connection, rows: list[dict]) -> None:
    now = utcnow_iso()
    conn.executemany(
        """
        INSERT INTO risk_predictions (latitude, longitude, location_name, risk_probability,
            model_probability, risk_category, prediction_window, contributing_factors,
            model_version, weather_observed_at, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                r["latitude"], r["longitude"], r["location_name"], r["risk_probability"],
                r.get("model_probability"), r["risk_category"], r["prediction_window"],
                _dumps(r.get("contributing_factors")), r["model_version"],
                r.get("weather_observed_at"), now,
            )
            for r in rows
        ],
    )
    conn.commit()


def risk_history_dataframe(conn: sqlite3.Connection, limit: int = 2000) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM risk_predictions ORDER BY created_at DESC LIMIT ?", conn, params=(limit,)
    )
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    return df


# ----------------------------------------------------------------- hotspots
def replace_hotspots(conn: sqlite3.Connection, run_id: str, hotspots: list[dict]) -> None:
    conn.execute("DELETE FROM hotspots")
    conn.executemany(
        """
        INSERT INTO hotspots (run_id, scope, center_latitude, center_longitude, radius_m,
            incident_count, unique_locations, average_severity, verification_count,
            pending_count, incident_ids, last_incident_at, cluster_generated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                run_id, h["scope"], h["center_latitude"], h["center_longitude"], h["radius_m"],
                h["incident_count"], h["unique_locations"], h["average_severity"],
                h["verification_count"], h["pending_count"], _dumps(h["incident_ids"]),
                h["last_incident_at"], h["cluster_generated_at"],
            )
            for h in hotspots
        ],
    )
    conn.commit()


def list_hotspots(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM hotspots ORDER BY incident_count DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["incident_ids"] = _loads(d["incident_ids"]) or []
        out.append(d)
    return out
