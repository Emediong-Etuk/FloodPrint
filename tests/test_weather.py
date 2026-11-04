from __future__ import annotations

import pytest
import requests

from services.weather_service import (WeatherDataError, WeatherService, WeatherUnavailableError, parse_response,
                                      validate_coordinates)
from tests.conftest import make_payload


class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, data=None, exc=None):
        self.data, self.exc, self.calls = data, exc, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return FakeResp(self.data)


def test_successful_response(payload):
    svc = WeatherService(session=FakeSession(payload))
    snap = svc.get(4.8156, 7.0122)
    assert snap.description == "Slight rain"
    assert snap.temperature_c == 27.5
    assert len(snap.past_daily) == 32 and len(snap.forecast_daily) == 7
    assert snap.forecast_7d_mm == pytest.approx(35.0)
    assert snap.rain_last_24h_mm == pytest.approx(5.0, rel=0.2)
    assert not snap.stale


def test_missing_data_rejected(payload):
    del payload["daily"]
    with pytest.raises(WeatherDataError):
        parse_response(payload)
    bad = make_payload()
    bad["hourly"]["precipitation"] = bad["hourly"]["precipitation"][:-3]
    with pytest.raises(WeatherDataError):
        parse_response(bad)


def test_timeout_without_fallback_raises():
    svc = WeatherService(session=FakeSession(exc=requests.Timeout("timed out")))
    with pytest.raises(WeatherUnavailableError):
        svc.get(4.8, 7.0)
    assert "timed out" in svc.last_error


def test_timeout_falls_back_to_stale_stored_observation(conn, payload):
    good = WeatherService(conn_factory=lambda: conn, session=FakeSession(payload))
    good.get(4.8156, 7.0122)
    bad = WeatherService(conn_factory=lambda: conn, session=FakeSession(exc=requests.Timeout("down")))
    snap = bad.get(4.8156, 7.0122)
    assert snap.stale and snap.stale_reason == "down"
    assert snap.temperature_c == 27.5  # the stored value, not an invented one


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), ("x", 1), (float("nan"), 1)])
def test_invalid_coordinates(lat, lon):
    with pytest.raises(ValueError):
        validate_coordinates(lat, lon)


def test_cached_response(payload):
    sess = FakeSession(payload)
    svc = WeatherService(session=sess, ttl_s=600)
    svc.get(4.8156, 7.0122)
    svc.get(4.8156, 7.0122)
    assert sess.calls == 1
