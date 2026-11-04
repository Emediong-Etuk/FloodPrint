"""Cached resources shared by all pages (DB, services, models)."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from config import MODELS_DIR, get_settings, load_region
from database import repository as repo
from database.connection import connect, initialize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@st.cache_resource(show_spinner=False)
def get_conn():
    s = get_settings()
    conn = connect(s.db_path)
    initialize(conn)
    if s.seed_demo_on_empty and repo.count_incidents(conn)["TOTAL"] == 0:
        from services.demo_data import seed_demo_data

        seed_demo_data(conn, load_region())
    return conn


def region():
    return load_region()


def settings():
    return get_settings()


@st.cache_resource(show_spinner=False)
def get_weather_service():
    from services.weather_service import WeatherService

    s = get_settings()
    return WeatherService(conn_factory=get_conn, ttl_s=s.weather_cache_ttl_s, timeout_s=s.weather_timeout_s)


@st.cache_resource(show_spinner=False)
def get_risk_predictor():
    from ai.risk_prediction import RiskPredictor, default_model_path

    return RiskPredictor.load(default_model_path())


@st.cache_resource(show_spinner=False)
def get_risk_service():
    from services.risk_service import RiskService

    return RiskService(get_risk_predictor(), get_weather_service(), MODELS_DIR / "dfo_events_region.csv")


@st.cache_resource(show_spinner="Loading Flood Vision model (first run downloads ~600 MB)...")
def get_vision():
    from ai.flood_vision import FloodVision

    s = get_settings()
    return FloodVision(MODELS_DIR, confidence_threshold=s.vision_confidence_threshold).load()


def incidents_df() -> pd.DataFrame:
    from services.incident_service import assign_area

    df = repo.incidents_dataframe(get_conn())
    if not df.empty:
        df["area"] = assign_area(df, region())
    return df


@st.cache_data(ttl=600, show_spinner=False)
def region_risk(_version: int = 0):
    """Risk for every monitored neighbourhood. Returns (rows, error)."""
    try:
        svc = get_risk_service()
    except Exception as exc:
        return None, f"Risk model unavailable: {exc}"
    r = region()
    pts = [(l.name, l.lat, l.lon) for l in r.locations]
    try:
        risks = svc.assess(pts, incidents_df())
    except Exception as exc:
        return None, f"Risk assessment failed: {exc}"
    risks = [x for x in risks if x is not None]
    if not risks:
        return None, f"Weather data unavailable: {get_weather_service().last_error or 'no response'}"
    try:
        repo.insert_risk_predictions(get_conn(), svc.to_rows(risks))
    except Exception:
        pass
    return risks, None
