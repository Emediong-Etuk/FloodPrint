from __future__ import annotations

import numpy as np
import pytest

from ai.feature_engineering import FEATURES, LiveFeatureInput, build_daily_features, live_feature_vector
from ai.risk_prediction import ModelUnavailableError, RiskPredictor, default_model_path, risk_category
from services.risk_service import RiskService, apply_odds_multiplier
from services.weather_service import WeatherService
from tests.conftest import make_payload
from tests.test_weather import FakeSession


@pytest.fixture(scope="module")
def predictor():
    return RiskPredictor.load(default_model_path())


def test_model_loading(predictor):
    assert predictor.features == FEATURES
    assert predictor.version.startswith("rf-")


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelUnavailableError):
        RiskPredictor.load(tmp_path / "nope.joblib")


def test_valid_feature_input(predictor):
    feats, missing = live_feature_vector(LiveFeatureInput([4.0] * 31, [10.0] * 7, 250, 15.0, 3))
    assert missing == []
    pred = predictor.predict(feats)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.category in {"Low", "Moderate", "High", "Very High"}
    assert pred.window == "next 7 days" and pred.contributions and pred.explanation


def test_missing_features_are_imputed_and_reported(predictor):
    feats, missing = live_feature_vector(LiveFeatureInput([None] * 31, [10.0] * 7, 250, None, None))
    assert "rain_30d" in missing and "elevation_m" in missing
    pred = predictor.predict(feats)
    assert set(missing) <= set(pred.imputed_features)
    assert "unavailable" in pred.explanation


def test_more_rain_does_not_lower_risk_much(predictor):
    dry, _ = live_feature_vector(LiveFeatureInput([0.0] * 31, [0.0] * 7, 250, 15.0, 3))
    wet, _ = live_feature_vector(LiveFeatureInput([25.0] * 31, [60.0] * 7, 250, 15.0, 3))
    assert predictor.predict(wet, explain=False).probability >= predictor.predict(dry, explain=False).probability * 0.8


@pytest.mark.parametrize("p,cat", [(0.0, "Low"), (0.20, "Low"), (0.21, "Moderate"), (0.45, "Moderate"),
                                   (0.46, "High"), (0.70, "High"), (0.71, "Very High"), (1.0, "Very High")])
def test_risk_category_mapping(p, cat):
    assert risk_category(p) == cat


def test_risk_category_rejects_bad_values():
    for bad in (-0.1, 1.2, float("nan")):
        with pytest.raises(ValueError):
            risk_category(bad)


def test_daily_features_future_window():
    import pandas as pd

    s = pd.Series(np.arange(20, dtype=float), index=pd.date_range("2020-01-01", periods=20))
    f = build_daily_features(s)
    assert f.loc["2020-01-05", "fcst_3d"] == 5 + 6 + 7  # strictly after the issue day
    assert f.loc["2020-01-05", "rain_3d"] == 2 + 3 + 4


def test_end_to_end_risk_service(predictor):
    from config import MODELS_DIR

    payload = [make_payload(4.79, 7.0), make_payload(4.84, 7.0)]
    svc = RiskService(predictor, WeatherService(session=FakeSession(payload)), MODELS_DIR / "dfo_events_region.csv")
    out = svc.assess([("A", 4.79, 7.0), ("B", 4.84, 7.0)])
    assert len(out) == 2 and all(0 <= r.risk_probability <= 0.95 for r in out)


def test_odds_multiplier():
    assert apply_odds_multiplier(0.1, 1.0) == pytest.approx(0.1)
    assert apply_odds_multiplier(0.1, 2.0) > 0.1
    assert apply_odds_multiplier(0.99, 50) <= 0.95
