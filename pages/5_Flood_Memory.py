"""Flood Memory: what the city remembers about where water goes."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.state import get_conn, incidents_df, region
from components.theme import C, STATUS_COLORS, demo_banner, disclaimer, esc, kv_rows, page_header, plotly_layout
from config import SAMPLE_DIR
from database.models import SEVERITY_LABEL, SEVERITY_SCORE

R = region()
page_header("Flood Memory", "Every flood leaves evidence. Historical records per neighbourhood.")
df = incidents_df()
demo_banner(int(df["is_demo"].sum()) if not df.empty else 0)
if df.empty:
    st.info("No incidents recorded yet. The memory grows as reports are submitted and reviewed.")
    disclaimer()
    st.stop()

c1, c2 = st.columns([1, 1])
mind = df["reported_at"].min().date()
period = c1.date_input("Date range", (mind, date.today()), min_value=mind, max_value=date.today())
include_pending = c2.toggle("Include pending (unverified) reports", value=True)
f = df[df["verification_status"] != "REJECTED"]
if not include_pending:
    f = f[f["verification_status"] == "CORROBORATED"]
if isinstance(period, tuple) and len(period) == 2:
    f = f[(f["reported_at"].dt.date >= period[0]) & (f["reported_at"].dt.date <= period[1])]


def rain_of(rc):
    if isinstance(rc, str):
        import json
        try:
            rc = json.loads(rc)
        except Exception:
            return None
    if not isinstance(rc, dict):
        return None
    return rc.get("station_rain_day_mm", rc.get("rain_last_24h_mm"))


f = f.assign(sev=f["user_reported_severity"].map(SEVERITY_SCORE), rain=f["rainfall_context"].map(rain_of))
prof = f.groupby("area").agg(
    reports=("id", "size"), corroborated=("verification_status", lambda s: int((s == "CORROBORATED").sum())),
    first=("reported_at", "min"), last=("reported_at", "max"), avg_sev=("sev", "mean"), max_sev=("sev", "max"),
    years=("reported_at", lambda s: s.dt.year.nunique()), rain=("rain", "median")).reset_index()
span_years = max(1.0, (f["reported_at"].max() - f["reported_at"].min()).days / 365.25) if len(f) else 1.0
prof["per_year"] = prof["reports"] / span_years
q = prof["per_year"].quantile([0.33, 0.66]).values if len(prof) > 2 else [0, 0]
prof["frequency"] = prof["per_year"].apply(lambda v: "Elevated" if v > q[1] else "Moderate" if v > q[0] else "Low")
prof = prof.sort_values("reports", ascending=False)

st.markdown("##### Frequently affected locations")
fig = go.Figure()
fig.add_bar(y=prof["area"], x=prof["corroborated"], name="Corroborated", orientation="h", marker_color=C["success"])
fig.add_bar(y=prof["area"], x=prof["reports"] - prof["corroborated"], name="Pending", orientation="h", marker_color=C["warning"])
fig.update_layout(barmode="stack")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(plotly_layout(fig, 420), width="stretch")

st.markdown("##### Location profile")
area = st.selectbox("Location", prof["area"].tolist())
p = prof[prof["area"] == area].iloc[0]
sub = f[f["area"] == area].sort_values("reported_at")
a, b = st.columns([1, 1.4], gap="medium")
with a:
    sev_range = f"{SEVERITY_LABEL[int(sub['sev'].min())]} to {SEVERITY_LABEL[int(sub['sev'].max())]}"
    heavy = sub["rain"].dropna()
    assoc = (f"{int((heavy >= 30).sum())} of {len(heavy)} incidents with rainfall data coincided with 30 mm+ days "
             f"(median {heavy.median():.0f} mm)." if len(heavy) else "No rainfall context recorded.")
    rows = [("Location", f"<b>{esc(area)}</b>"), ("Historical reports", str(int(p["reports"]))),
            ("Corroborated incidents", str(int(p["corroborated"]))),
            ("First recorded", f"{p['first']:%d %b %Y}"), ("Most recent", f"{p['last']:%d %b %Y}"),
            ("Recorded severity", sev_range), ("Years with incidents", str(int(p["years"]))),
            ("Historical frequency", f"{p['frequency']} ({p['per_year']:.1f} reports/yr, relative to other areas)")]
    st.markdown(f"<div class='fp-card'>{kv_rows(rows)}</div>", unsafe_allow_html=True)
    st.markdown(f"**Rainfall association:** {assoc}")
    st.caption("Historical flooding is one input to risk, not proof an area will flood again.")
with b:
    fig = go.Figure()
    for s in ["CORROBORATED", "PENDING"]:
        x = sub[sub["verification_status"] == s]
        fig.add_scatter(x=x["reported_at"], y=x["sev"], mode="markers", name=s.title(),
                        marker=dict(size=11, color=STATUS_COLORS[s], line=dict(width=1, color=C["bg"])),
                        text=x["description"], hovertemplate="%{x|%d %b %Y}<br>%{text}<extra></extra>")
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["None", "Minor", "Moderate", "Severe"], range=[-0.4, 3.4])
    st.plotly_chart(plotly_layout(fig, 300, f"Incident timeline: {area}"), width="stretch")

st.markdown("##### Rainfall and incidents over time")
gs = SAMPLE_DIR / "port_harcourt_gsod_daily.csv"
if gs.exists():
    rain = pd.read_csv(gs, parse_dates=["date"]).dropna(subset=["precipitation_mm"])
    m = rain.set_index("date")["precipitation_mm"].resample("MS").sum(min_count=10)
    inc_m = f.set_index("reported_at").resample("MS").size()
    inc_m.index = inc_m.index.tz_localize(None)
    fig = go.Figure()
    fig.add_bar(x=m.index, y=m.values, name="Monthly rainfall (mm, station)", marker_color="rgba(56,189,248,.45)")
    fig.add_scatter(x=inc_m.index, y=inc_m.values, name="Incidents", yaxis="y2", mode="lines+markers", line=dict(color=C["danger"]))
    fig.update_layout(yaxis=dict(title="mm"), yaxis2=dict(title="incidents", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(plotly_layout(fig, 320, "Port Harcourt Intl station rainfall (NOAA GSOD, real) vs recorded incidents"),
                    width="stretch")
    st.caption("Station months with fewer than 10 valid days are left blank. GSOD records after Aug 2025 were not yet available.")

st.markdown("##### Compare locations")
pick = st.multiselect("Locations", prof["area"].tolist(), prof["area"].tolist()[:4])
if pick:
    cmp = f[f["area"].isin(pick)].copy()
    cmp["year"] = cmp["reported_at"].dt.year
    t = cmp.groupby(["year", "area"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for col in t.columns:
        fig.add_scatter(x=t.index, y=t[col], mode="lines+markers", name=col)
    st.plotly_chart(plotly_layout(fig, 300, "Incidents per year"), width="stretch")
    st.dataframe(prof[prof["area"].isin(pick)][["area", "reports", "corroborated", "avg_sev", "max_sev", "frequency", "last"]]
                 .rename(columns={"avg_sev": "avg severity (0-3)", "max_sev": "max severity"}), hide_index=True,
                 width="stretch")
disclaimer()
