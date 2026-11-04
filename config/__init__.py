"""Central configuration for FLOODPRINT.

Values come from environment variables (optionally loaded from a .env file)
and, when running inside Streamlit, from ``st.secrets``. Region details live
in ``config/regions/<region>.json`` so the monitored area can be changed
without touching code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REGIONS_DIR = Path(__file__).resolve().parent / "regions"


def _secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment, then Streamlit secrets."""
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    admin_area: str
    timezone: str
    center_lat: float
    center_lon: float
    default_zoom: int
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    locations: tuple[Location, ...]
    reference_station: dict = field(default_factory=dict)
    coordinate_note: str = ""

    def contains(self, lat: float, lon: float, margin: float = 0.0) -> bool:
        return (
            self.min_lat - margin <= lat <= self.max_lat + margin
            and self.min_lon - margin <= lon <= self.max_lon + margin
        )

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.admin_area}"


@lru_cache(maxsize=8)
def load_region(region_id: str | None = None) -> Region:
    region_id = region_id or _secret("FLOODPRINT_REGION", "port_harcourt")
    path = REGIONS_DIR / f"{region_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Region config not found: {path}")
    raw = json.loads(path.read_text())
    return Region(
        id=raw["id"],
        name=raw["name"],
        admin_area=raw.get("admin_area", ""),
        timezone=raw.get("timezone", "UTC"),
        center_lat=raw["center"]["lat"],
        center_lon=raw["center"]["lon"],
        default_zoom=raw.get("default_zoom", 12),
        min_lat=raw["bbox"]["min_lat"],
        max_lat=raw["bbox"]["max_lat"],
        min_lon=raw["bbox"]["min_lon"],
        max_lon=raw["bbox"]["max_lon"],
        locations=tuple(Location(**loc) for loc in raw["locations"]),
        reference_station=raw.get("reference_station", {}),
        coordinate_note=raw.get("coordinate_note", ""),
    )


@dataclass(frozen=True)
class Settings:
    db_path: Path
    uploads_dir: Path
    models_dir: Path
    admin_password: str | None
    seed_demo_on_empty: bool
    weather_cache_ttl_s: int
    weather_timeout_s: float
    max_upload_mb: float
    vision_confidence_threshold: float
    duplicate_radius_m: float
    duplicate_window_h: float
    hotspot_eps_m: float
    hotspot_min_samples: int


def get_settings() -> Settings:
    return Settings(
        db_path=Path(_secret("FLOODPRINT_DB_PATH", str(DATA_DIR / "floodprint.db"))),
        uploads_dir=Path(_secret("FLOODPRINT_UPLOADS_DIR", str(UPLOADS_DIR))),
        models_dir=MODELS_DIR,
        admin_password=_secret("FLOODPRINT_ADMIN_PASSWORD"),
        seed_demo_on_empty=_bool(_secret("FLOODPRINT_SEED_DEMO"), True),
        weather_cache_ttl_s=int(_secret("FLOODPRINT_WEATHER_TTL_S", "900")),
        weather_timeout_s=float(_secret("FLOODPRINT_WEATHER_TIMEOUT_S", "10")),
        max_upload_mb=float(_secret("FLOODPRINT_MAX_UPLOAD_MB", "8")),
        vision_confidence_threshold=float(_secret("FLOODPRINT_VISION_THRESHOLD", "0.55")),
        duplicate_radius_m=float(_secret("FLOODPRINT_DUPLICATE_RADIUS_M", "150")),
        duplicate_window_h=float(_secret("FLOODPRINT_DUPLICATE_WINDOW_H", "3")),
        hotspot_eps_m=float(_secret("FLOODPRINT_HOTSPOT_EPS_M", "450")),
        hotspot_min_samples=int(_secret("FLOODPRINT_HOTSPOT_MIN_SAMPLES", "3")),
    )


DISCLAIMER = (
    "FLOODPRINT provides experimental environmental risk intelligence based on available "
    "weather information, historical observations, and machine learning models. Predictions "
    "are estimates, not guarantees. Always follow official emergency instructions and verify "
    "local conditions before travelling."
)
