"""SQLite schema and record definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 1


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    CORROBORATED = "CORROBORATED"
    REJECTED = "REJECTED"


class Severity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


SEVERITY_SCORE = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
SEVERITY_LABEL = {0: "None", 1: "Minor", 2: "Moderate", 3: "Severe"}

INCIDENT_TYPES = (
    "Road flooding",
    "Residential flooding",
    "Blocked drainage / waterlogging",
    "Overflowing canal or creek",
    "Commercial area flooding",
    "Other",
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    location_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    incident_type TEXT NOT NULL DEFAULT 'Road flooding',
    image_path TEXT,
    image_hash TEXT,
    user_reported_severity TEXT NOT NULL CHECK (user_reported_severity IN ('none','minor','moderate','severe')),
    ai_classification TEXT,
    ai_confidence REAL CHECK (ai_confidence IS NULL OR ai_confidence BETWEEN 0 AND 1),
    ai_details TEXT,
    verification_status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (verification_status IN ('PENDING','CORROBORATED','REJECTED')),
    verification_note TEXT,
    verified_at TEXT,
    duplicate_of INTEGER REFERENCES incidents(id) ON DELETE SET NULL,
    quality_flags TEXT,
    rainfall_context TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    reported_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_latlon ON incidents (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_incidents_reported_at ON incidents (reported_at);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (verification_status);
CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents (location_name);

CREATE TABLE IF NOT EXISTS weather_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    temperature REAL,
    precipitation REAL,
    rainfall_intensity REAL,
    forecast_precipitation REAL,
    weather_description TEXT,
    payload TEXT,
    source TEXT NOT NULL DEFAULT 'open-meteo',
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_weather_latlon_time ON weather_observations (latitude, longitude, observed_at);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    location_name TEXT NOT NULL,
    risk_probability REAL NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    model_probability REAL,
    risk_category TEXT NOT NULL,
    prediction_window TEXT NOT NULL,
    contributing_factors TEXT,
    model_version TEXT NOT NULL,
    weather_observed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_location_time ON risk_predictions (location_name, created_at);

CREATE TABLE IF NOT EXISTS hotspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('corroborated','all_reports')),
    center_latitude REAL NOT NULL,
    center_longitude REAL NOT NULL,
    radius_m REAL,
    incident_count INTEGER NOT NULL,
    unique_locations INTEGER NOT NULL,
    average_severity REAL,
    verification_count INTEGER NOT NULL,
    pending_count INTEGER NOT NULL,
    incident_ids TEXT NOT NULL,
    last_incident_at TEXT,
    cluster_generated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hotspots_run ON hotspots (run_id);

CREATE TABLE IF NOT EXISTS location_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    incident_id INTEGER NOT NULL UNIQUE REFERENCES incidents(id) ON DELETE CASCADE,
    rainfall_context TEXT,
    severity TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_location ON location_history (location_name, recorded_at);
"""


@dataclass
class Incident:
    latitude: float
    longitude: float
    location_name: str
    user_reported_severity: str
    reported_at: str
    description: str = ""
    incident_type: str = "Road flooding"
    image_path: str | None = None
    image_hash: str | None = None
    ai_classification: str | None = None
    ai_confidence: float | None = None
    ai_details: dict | None = None
    verification_status: str = VerificationStatus.PENDING.value
    verification_note: str | None = None
    verified_at: str | None = None
    duplicate_of: int | None = None
    quality_flags: list[str] = field(default_factory=list)
    rainfall_context: dict | None = None
    is_demo: bool = False
    created_at: str | None = None
    id: int | None = None
