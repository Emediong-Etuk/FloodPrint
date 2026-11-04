"""Live Flood Map: incidents, predicted risk, hotspots and history with filters."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from ai.feature_engineering import haversine_km
from ai.hotspot_detection import detect_hotspots
from components.maps import build_map, legend_html
from components.state import incidents_df, region, region_risk, settings
from components.theme import C, banner, demo_banner, disclaimer, esc, kv_rows, page_header

R, S = region(), settings()
page_header("Live Flood Map", "Observed reports (solid dots) are kept separate from predictions (dashed rings).")
df = incidents_df()
demo_banner(int(df["is_demo"].sum()) if not df.empty else 0)

with st.sidebar:
    st.markdown("#### Map filters")
    show_current = st.checkbox("Current incidents (last 7 days)", True)
    show_hist = st.checkbox("Historical incidents", True)
    show_risk = st.checkbox("Predicted risk", True)
    show_hot = st.checkbox("Hotspot clusters", True)
    sev = st.multiselect("Severity", ["minor", "moderate", "severe", "none"], ["minor", "moderate", "severe"])
    stat = st.multiselect("Verification status", ["CORROBORATED", "PENDING", "REJECTED"], ["CORROBORATED", "PENDING"])
    min_d = df["reported_at"].min().date() if not df.empty else date.today() - timedelta(days=365)
    period = st.date_input("Time period", (min_d, date.today()), min_value=min_d, max_value=date.today())
    focus = st.selectbox("Neighbourhood", ["Whole area"] + [l.name for l in R.locations])

f = df.copy()
if not f.empty:
    if isinstance(period, tuple) and len(period) == 2:
        f = f[(f["reported_at"].dt.date >= period[0]) & (f["reported_at"].dt.date <= period[1])]
    f = f[f["user_reported_severity"].isin(sev) & f["verification_status"].isin(stat)]
    recent = f["reported_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    f = f[(recent & show_current) | (~recent & show_hist)]

risks, risk_err = region_risk() if show_risk else (None, None)
if show_risk and risk_err:
    banner(risk_err)
hot = [h.to_dict() for h in detect_hotspots(df, S.hotspot_eps_m, S.hotspot_min_samples)] if show_hot and not df.empty else []

center, zoom = None, None
if focus != "Whole area":
    loc = next(l for l in R.locations if l.name == focus)
    center, zoom = (loc.lat, loc.lon), 15
m = build_map(R, f, risks, hot, height_center=center, zoom=zoom)
col_map, col_info = st.columns([3, 1.25], gap="medium")
with col_map:
    out = st_folium(m, height=620, width="stretch", returned_objects=["last_object_clicked"], key="live_map")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption(f"Showing {len(f)} of {len(df)} incident reports. {R.coordinate_note}")

with col_info:
    st.markdown("##### Location details")
    clicked = (out or {}).get("last_object_clicked")
    if clicked:
        lat, lon = clicked["lat"], clicked["lng"]
    elif focus != "Whole area":
        lat, lon = center
    else:
        lat = lon = None
    if lat is None:
        st.info("Click a marker or ring, or choose a neighbourhood, to see local details.")
    else:
        dist = haversine_km(lat, lon, [l.lat for l in R.locations], [l.lon for l in R.locations])
        near = R.locations[int(dist.argmin())]
        rows = [("Nearest area", esc(near.name)), ("Coordinates", f"{lat:.4f}, {lon:.4f}")]
        if not df.empty:
            d = haversine_km(lat, lon, df["latitude"].values, df["longitude"].values) * 1000
            local = df[d <= 500]
            rows += [("Reports within 500 m", str(len(local))),
                     ("Corroborated", str(int((local["verification_status"] == "CORROBORATED").sum()))),
                     ("Most recent", local["reported_at"].max().strftime("%d %b %Y") if len(local) else "none")]
            hit = df[d <= 30]
            if len(hit):
                r = hit.iloc[0]
                rows += [("Selected report", f"#{r.id}{' [DEMO]' if r.is_demo else ''}"), ("Type", esc(r.incident_type)),
                         ("Severity (reported)", esc(r.user_reported_severity)), ("Status", esc(r.verification_status)),
                         ("AI photo analysis", esc(r.ai_classification or "not analysed")),
                         ("Reported", r.reported_at.strftime("%d %b %Y %H:%M UTC"))]
        risk = next((x for x in (risks or []) if x.location_name == near.name), None)
        if risk:
            rows += [("Predicted risk", f"<b>{risk.risk_category}</b> ({risk.risk_probability:.1%})"),
                     ("Window", esc(risk.prediction.window)),
                     ("Rain last 24 h", f"{risk.weather.rain_last_24h_mm or 0:.1f} mm"),
                     ("Forecast 7 days", f"{risk.weather.forecast_7d_mm or 0:.0f} mm"),
                     ("Data time", esc(risk.weather.observed_at))]
        st.markdown(f"<div class='fp-card'>{kv_rows(rows)}</div>", unsafe_allow_html=True)
        if risk:
            st.caption(risk.prediction.explanation)
disclaimer()
