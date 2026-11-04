from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from database import repository as repo
from services.image_service import perceptual_hash
from services.incident_service import IncidentValidationError, submit_incident


def base(**kw):
    d = dict(latitude=4.81, longitude=7.0, location_name="D-Line", description="Water across the road",
             severity="moderate", incident_type="Road flooding", reported_at=datetime.now(timezone.utc))
    d.update(kw)
    return d


def test_report_creation_and_retrieval(conn, region):
    res = submit_incident(conn, region, **base())
    inc = repo.get_incident(conn, res.incident_id)
    assert inc.location_name == "D-Line" and inc.verification_status == "PENDING"
    assert len(repo.list_incidents(conn)) == 1
    assert "no_photo" in inc.quality_flags


def test_verification_updates_history(conn, region):
    iid = submit_incident(conn, region, **base()).incident_id
    repo.update_verification(conn, iid, "CORROBORATED", "seen by team")
    assert len(repo.history_dataframe(conn)) == 1
    repo.update_verification(conn, iid, "REJECTED")
    assert len(repo.history_dataframe(conn)) == 0
    with pytest.raises(ValueError):
        repo.update_verification(conn, iid, "VERIFIED_BY_AI")


def test_duplicate_detection_flags_not_deletes(conn, region):
    img = Image.new("RGB", (200, 200), (40, 90, 160))
    h = perceptual_hash(img)
    a = submit_incident(conn, region, **base(), image_path="a.jpg", image_hash=h)
    b = submit_incident(conn, region, **base(latitude=4.8101), image_path="b.jpg", image_hash=h)
    assert b.duplicate_of == a.incident_id and "possible_duplicate" in b.flags
    assert len(repo.list_incidents(conn)) == 2


def test_nearby_reports_counted(conn, region):
    submit_incident(conn, region, **base())
    r = submit_incident(conn, region, **base(latitude=4.8105))
    assert r.nearby_reports == 1 and r.duplicate_of is None


@pytest.mark.parametrize("kw", [
    dict(latitude=6.5, longitude=3.3),  # Lagos: outside the monitored area
    dict(latitude=200),
    dict(location_name="  "),
    dict(severity="catastrophic"),
    dict(incident_type="Alien invasion"),
    dict(reported_at=datetime.now(timezone.utc) + timedelta(hours=2)),
    dict(reported_at=datetime.now(timezone.utc) - timedelta(days=30)),
    dict(description="x" * 1200),
])
def test_invalid_submissions(conn, region, kw):
    with pytest.raises(IncidentValidationError):
        submit_incident(conn, region, **base(**kw))


def test_exif_location_mismatch_flag(conn, region):
    r = submit_incident(conn, region, **base(), image_gps=(6.5, 3.3))
    assert any(f.startswith("photo_gps_") for f in r.flags)
