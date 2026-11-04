"""Dashboard: what is happening across the monitored area right now?"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from ai.hotspot_detection import detect_hotspots
from components.maps import build_map, legend_html
from components.state import get_conn, get_weather_service, incidents_df, region, region_risk, settings
from components.theme import C, STATUS_COLORS, badge, banner, demo_banner, disclaimer, esc, metric_card, page_header, plotly_layout
from database import repository as repo

R = region()
S = settings()
conn = get_conn()
counts = repo.count_incidents(conn)
df = incidents_df()

page_header("Flood Intelligence Dashboard", f"{R.display_name} - live overview")

names = ["Region centre"] + [l.name for l in R.locations]
sel = st.selectbox("Location", names, index=0, help="Weather and risk shown for this location")
if sel == "Region centre":
    lat, lon = R.center_lat, R.center_lon
else:
    loc = next(l for l in R.locations if l.name == sel)
    lat, lon = loc.lat, loc.lon

weather_err = None
snap = None
with st.spinner("Fetching weather from Open-Meteo..."):
    try:
        snap = get_weather_service().get(lat, lon)
    except Exception as exc:
        weather_err = str(exc)

risks, risk_err = region_risk()
alerts = [r for r in (risks or []) if r.risk_category in {"High", "Very High"}]
now_local = pd.Timestamp.now(tz=R.timezone)
strip = [f"<span>Location <b>{esc(sel)}</b></span>", f"<span>Date <b>{now_local:%a %d %b %Y}</b></span>"]
if snap:
    strip.append(f"<span>Weather <b>{esc(snap.description)}, {snap.temperature_c:.0f}°C</b></span>")
    strip.append(f"<span>Data <b>{esc(snap.observed_at.replace('T', ' '))}</b> local{' (STALE)' if snap.stale else ''}</span>")
strip.append(f"<span>Alerts <b style='color:{C['danger'] if alerts else C['success']}'>{len(alerts)} high-risk area(s)</b></span>")
st.markdown(f"<div class='fp-header-strip'>{''.join(strip)}</div>", unsafe_allow_html=True)

demo_banner(counts["DEMO"])
if weather_err:
    banner(f"Live weather unavailable and no stored observation to fall back on: {weather_err}")
elif snap and snap.stale:
    banner(f"Weather API unreachable - showing the last stored observation from {snap.fetched_at} (stale).")

hot = detect_hotspots(df, S.hotspot_eps_m, S.hotspot_min_samples, "all_reports") if not df.empty else []
hot_c = [h for h in hot if h.verification_count >= S.hotspot_min_samples]
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Reported incidents", counts["TOTAL"], f"{counts['PENDING']} pending review"), unsafe_allow_html=True)
c2.markdown(metric_card("Corroborated incidents", counts["CORROBORATED"], f"{counts['REJECTED']} rejected", C["success"]), unsafe_allow_html=True)
c3.markdown(metric_card("High-risk locations", len(alerts) if risks else "n/a",
                        "predicted, next 7 days" if risks else (risk_err or "")[:60],
                        C["danger"] if alerts else None), unsafe_allow_html=True)
c4.markdown(metric_card("Flood hotspots", len(hot), f"{len(hot_c)} with 3+ corroborated reports", C["accent"]), unsafe_allow_html=True)
st.write("")

left, right = st.columns([3, 2], gap="medium")
with left:
    st.markdown("##### Flood overview map")
    m = build_map(R, df, risks, [h.to_dict() for h in hot])
    st_folium(m, height=460, width="stretch", returned_objects=[], key="dash_map")
    st.markdown(legend_html(), unsafe_allow_html=True)
with right:
    st.markdown("##### Current rainfall")
    if snap:
        w1, w2, w3 = st.columns(3)
        w1.metric("Last 24 h", f"{snap.rain_last_24h_mm or 0:.1f} mm")
        w2.metric("Next 24 h", f"{snap.forecast_next_24h_mm or 0:.1f} mm")
        w3.metric("Next 7 days", f"{snap.forecast_7d_mm or 0:.0f} mm")
        h = snap.hourly_window(hours_back=48, hours_ahead=48)
        now = pd.Timestamp(snap.observed_at)
        fig = go.Figure()
        fig.add_bar(x=h["time"], y=h["precipitation_mm"],
                    marker_color=[C["accent"] if t <= now else "rgba(56,189,248,.4)" for t in h["time"]], name="mm/h")
        fig.add_vline(x=now, line_dash="dot", line_color=C["muted"])
        fig.update_yaxes(title="mm per hour")
        st.plotly_chart(plotly_layout(fig, 230, "Hourly rainfall: past 48 h (solid) and forecast (light)"), width="stretch")
        st.caption(f"Source: {snap.source}. Forecast values are model estimates.")
    else:
        st.info("No weather data to display.")

    st.markdown("##### Highest predicted risk")
    if risks:
        top = sorted(risks, key=lambda r: r.risk_probability, reverse=True)[:5]
        rows = "".join(
            f"<div class='fp-row'><span>{esc(r.location_name)}</span><span>{badge(r.risk_category, {'Low': C['success'], 'Moderate': C['yellow'], 'High': C['warning'], 'Very High': C['danger']}[r.risk_category])} "
            f"{r.risk_probability:.1%}</span></div>" for r in top)
        st.markdown(f"<div class='fp-card'>{rows}<div class='fp-metric-note'>Probability of a large-flood onset within 7 days (estimate). "
                    f"Model {esc(top[0].prediction.model_version)}.</div></div>", unsafe_allow_html=True)
    else:
        st.info(risk_err or "Risk estimates unavailable.")

st.write("")
a, b = st.columns(2, gap="medium")
with a:
    st.markdown("##### Recent incidents")
    if df.empty:
        st.info("No incidents reported yet. Use Report Flood to add the first one.")
    else:
        rec = df.head(8)
        rows = "".join(
            f"<div class='fp-row'><span>#{r.id} {esc(r.location_name)}{' [DEMO]' if r.is_demo else ''}<br>"
            f"<span style='color:{C['muted']};font-size:.78rem'>{r.reported_at:%d %b %Y %H:%M} UTC - {esc(r.user_reported_severity)}</span></span>"
            f"<span>{badge(r.verification_status, STATUS_COLORS[r.verification_status])}</span></div>"
            for r in rec.itertuples())
        st.markdown(f"<div class='fp-card'>{rows}</div>", unsafe_allow_html=True)
with b:
    st.markdown("##### Emerging hotspots")
    if not hot:
        st.info(f"No clusters of {S.hotspot_min_samples}+ reports within {S.hotspot_eps_m:.0f} m.")
    else:
        rows = "".join(
            f"<div class='fp-row'><span>Hotspot {h.hotspot_id} - {esc(df[df.id.isin(h.incident_ids)].area.mode().iat[0])}<br>"
            f"<span style='color:{C['muted']};font-size:.78rem'>{h.incident_count} reports, {h.verification_count} corroborated, last {h.last_incident_at[:10]}</span></span>"
            f"<span>{h.average_severity:.1f} / 3</span></div>" for h in hot[:6])
        st.markdown(f"<div class='fp-card'>{rows}<div class='fp-metric-note'>DBSCAN, haversine, eps {S.hotspot_eps_m:.0f} m, "
                    f"min {S.hotspot_min_samples} reports; rejected reports excluded.</div></div>", unsafe_allow_html=True)

if not df.empty:
    st.write("")
    t = df.set_index("reported_at").groupby([pd.Grouper(freq="QS"), "verification_status"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for s in ["CORROBORATED", "PENDING", "REJECTED"]:
        if s in t:
            fig.add_bar(x=t.index, y=t[s], name=s.title(), marker_color=STATUS_COLORS[s])
    fig.update_layout(barmode="stack")
    st.plotly_chart(plotly_layout(fig, 260, "Reported incidents per quarter"), width="stretch")
disclaimer()
