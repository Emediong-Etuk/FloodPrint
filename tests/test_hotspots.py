from __future__ import annotations

import pandas as pd
import pytest

from ai.hotspot_detection import detect_hotspots


def frame(points, status="CORROBORATED"):
    return pd.DataFrame([{"id": i + 1, "latitude": la, "longitude": lo, "verification_status": status,
                          "user_reported_severity": "moderate", "reported_at": "2026-10-01T10:00:00+00:00",
                          "location_name": "X"} for i, (la, lo) in enumerate(points)])


def test_multiple_nearby_incidents_form_cluster():
    pts = [(4.8000, 7.0000), (4.8010, 7.0010), (4.8005, 7.0002), (4.8012, 7.0008)]
    hs = detect_hotspots(frame(pts), eps_m=300, min_samples=3)
    assert len(hs) == 1 and hs[0].incident_count == 4
    assert abs(hs[0].center_latitude - 4.8007) < 0.001


def test_isolated_incidents_do_not_cluster():
    pts = [(4.75, 6.95), (4.80, 7.00), (4.85, 7.05), (4.90, 7.10)]
    assert detect_hotspots(frame(pts), eps_m=300, min_samples=3) == []


def test_single_report_never_hotspot():
    assert detect_hotspots(frame([(4.8, 7.0)]), eps_m=300, min_samples=3) == []
    with pytest.raises(ValueError):
        detect_hotspots(frame([(4.8, 7.0)]), min_samples=1)


def test_empty_dataset():
    assert detect_hotspots(pd.DataFrame(), eps_m=300) == []


def test_invalid_coordinates_dropped():
    pts = [(4.8, 7.0), (4.8001, 7.0001), (4.8002, 7.0), (999, 7.0), (None, 7.0)]
    hs = detect_hotspots(frame(pts), eps_m=300, min_samples=3)
    assert len(hs) == 1 and hs[0].incident_count == 3


def test_scopes_separate_unverified():
    pts = [(4.8, 7.0), (4.8001, 7.0001), (4.8002, 7.0)]
    df = frame(pts, status="PENDING")
    assert detect_hotspots(df, eps_m=300, min_samples=3, scope="corroborated") == []
    hs = detect_hotspots(df, eps_m=300, min_samples=3, scope="all_reports")
    assert len(hs) == 1 and hs[0].pending_count == 3 and hs[0].verification_count == 0
    rej = frame(pts, status="REJECTED")
    assert detect_hotspots(rej, eps_m=300, min_samples=3) == []
