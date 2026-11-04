from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def conn():
    from database.connection import connect, initialize

    c = connect(":memory:")
    initialize(c)
    yield c
    c.close()


@pytest.fixture
def region():
    from config import load_region

    return load_region("port_harcourt")


def make_payload(lat=4.8156, lon=7.0122, daily_rain=None, now="2026-11-04T14:00"):
    """Open-Meteo-shaped response (structure follows the public API docs)."""
    today = datetime.fromisoformat(now)
    days = [(today.date() - timedelta(days=31 - i)).isoformat() for i in range(31 + 8)]
    daily_rain = daily_rain or [5.0] * len(days)
    hours = []
    start = datetime.fromisoformat(days[0] + "T00:00")
    for i in range(len(days) * 24):
        hours.append((start + timedelta(hours=i)).isoformat(timespec="minutes"))
    hourly = [daily_rain[i // 24] / 24 for i in range(len(hours))]
    return {
        "latitude": lat, "longitude": lon, "elevation": 18.0, "timezone": "Africa/Lagos",
        "current": {"time": now, "temperature_2m": 27.5, "relative_humidity_2m": 85, "precipitation": 0.4,
                    "rain": 0.4, "weather_code": 61, "wind_speed_10m": 7.2},
        "hourly_units": {"precipitation": "mm"},
        "hourly": {"time": hours, "precipitation": hourly, "precipitation_probability": [50] * len(hours)},
        "daily": {"time": days, "precipitation_sum": daily_rain, "precipitation_probability_max": [60] * len(days)},
    }


@pytest.fixture
def payload():
    return make_payload()
