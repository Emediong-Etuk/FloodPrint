"""Flood Intelligence: weather, model risk, explanations, model information, hotspots."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.feature_engineering import FEATURE_DESCRIPTIONS
from ai.hotspot_detection import detect_hotspots
from ai.risk_prediction import RISK_COLORS
from components.state import get_conn, incidents_df, region, region_risk, settings
from components.theme import C, banner, demo_banner, disclaimer, esc, kv_rows, page_header, plotly_layout
from config import MODELS_DIR
from database import repository as repo
from services import risk_service as rs

R, S = region(), settings()
page_header("Flood Intelligence", "Rainfall-driven risk estimates, their drivers, and emerging hotspots.")
df = incidents_df()
demo_banner(int(df["is_demo"].sum()) if not df.empty else 0)
risks, err = region_risk()
tab_risk, tab_hot, tab_model = st.tabs(["Risk estimates", "Hotspots (DBSCAN)", "Model information"])

with tab_risk:
    if err:
        banner(err)
    else:
        rk = sorted(risks, key=lambda r: r.risk_probability, reverse=True)
        snap = rk[0].weather
        if snap.stale:
            banner(f"Weather is stale (last update {snap.fetched_at}).")
        c = st.columns(4)
        c[0].metric("Rain last 24 h", f"{snap.rain_last_24h_mm or 0:.1f} mm")
        c[1].metric("Rain last 7 days", f"{sum(v for v in snap.past_daily[-7:] if v):.0f} mm")
        c[2].metric("Forecast next 7 days", f"{snap.forecast_7d_mm or 0:.0f} mm")
        c[3].metric("Peak hourly rate (next 24 h)", f"{snap.max_intensity_next_24h or 0:.1f} mm/h")

        d = snap.daily.copy()
        d["kind"] = ["Observed" if x <= snap.today else "Forecast" for x in d["date"]]
        fig = go.Figure()
        for k, col in (("Observed", C["accent"]), ("Forecast", "rgba(56,189,248,.45)")):
            s = d[d["kind"] == k]
            fig.add_bar(x=s["date"], y=s["precipitation_mm"], name=k, marker_color=col)
        cum = d[d["kind"] == "Observed"]["precipitation_mm"].fillna(0).cumsum()
        fig.add_scatter(x=d[d["kind"] == "Observed"]["date"], y=cum, name="30-day accumulation", yaxis="y2",
                        line=dict(color=C["warning"], width=2))
        fig.update_layout(yaxis=dict(title="mm/day"), yaxis2=dict(title="cumulative mm", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(plotly_layout(fig, 300, f"Daily rainfall at {rk[0].location_name}: last 30 days and 7-day forecast"),
                        width="stretch")

        a, b = st.columns([1.2, 1], gap="medium")
        with a:
            tbl = pd.DataFrame([{
                "Location": r.location_name, "Risk": r.risk_category, "Estimate": r.risk_probability,
                "Model probability": r.model_probability, "Local adjustment": f"x{r.odds_multiplier:.2f} odds",
                "Corroborated nearby": r.corroborated_nearby} for r in rk])
            st.dataframe(tbl, hide_index=True, width="stretch",
                         column_config={"Estimate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                        "Model probability": st.column_config.NumberColumn(format="%.2f%%")})
            st.caption(f"Window: {rk[0].prediction.window}. Data time {snap.observed_at} ({snap.timezone}). "
                       f"Model {rk[0].prediction.model_version}.")
        with b:
            counts = pd.Series([r.risk_category for r in rk]).value_counts().reindex(list(RISK_COLORS), fill_value=0)
            fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[RISK_COLORS[k] for k in counts.index]))
            st.plotly_chart(plotly_layout(fig, 250, "Risk category breakdown"), width="stretch")

        st.markdown("##### Why this estimate?")
        names = [r.location_name for r in rk]
        pick = st.selectbox("Location", names)
        r = next(x for x in rk if x.location_name == pick)
        p = r.prediction
        x1, x2 = st.columns([1, 1.3], gap="medium")
        with x1:
            rows = [("Risk category", f"<b style='color:{RISK_COLORS[r.risk_category]}'>{r.risk_category}</b>"),
                    ("Location estimate", f"{r.risk_probability:.2%}"),
                    ("ML model probability", f"{r.model_probability:.2%}"),
                    ("Training base rate", f"{p.base_rate:.2%}" if p.base_rate else "n/a"),
                    ("Relative to base rate", f"{p.lift:.1f}x" if p.lift else "n/a"),
                    ("Flood Memory adjustment", f"x{r.odds_multiplier:.2f} odds ({r.corroborated_nearby} corroborated, {r.pending_nearby} pending within {rs.RADIUS_M} m)"),
                    ("Prediction window", esc(p.window)), ("Data timestamp", esc(r.weather.observed_at)),
                    ("Model version", esc(p.model_version))]
            st.markdown(f"<div class='fp-card'>{kv_rows(rows)}</div>", unsafe_allow_html=True)
            st.write(p.explanation)
            for n in r.notes:
                st.caption(n)
        with x2:
            cdf = pd.DataFrame(p.contributions)
            fig = go.Figure(go.Bar(y=cdf["label"], x=cdf["effect"] * 100, orientation="h",
                                   marker_color=[C["danger"] if v > 0 else C["success"] for v in cdf["effect"]]))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="change in probability vs a typical day (percentage points)")
            st.plotly_chart(plotly_layout(fig, 380, "Feature contributions (occlusion)"), width="stretch")

with tab_hot:
    if df.empty:
        st.info("No incidents yet.")
    else:
        c1, c2, c3 = st.columns(3)
        scope = c1.radio("Reports included", ["corroborated", "all_reports"],
                         format_func=lambda s: "Corroborated only" if s == "corroborated" else "Corroborated + pending")
        eps = c2.slider("Neighbourhood radius (eps, m)", 150, 1500, int(S.hotspot_eps_m), 50)
        mins = c3.slider("Minimum reports per hotspot", 2, 10, S.hotspot_min_samples)
        hot = detect_hotspots(df, eps, mins, scope)
        if st.button("Save this clustering run to the database"):
            from datetime import datetime, timezone
            repo.replace_hotspots(get_conn(), datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                                  [h.to_dict() for h in hot])
            st.success(f"Saved {len(hot)} hotspots.")
        if not hot:
            st.info("No clusters found with these settings.")
        else:
            st.dataframe(pd.DataFrame([{
                "Hotspot": h.hotspot_id, "Area": df[df.id.isin(h.incident_ids)]["area"].mode().iat[0],
                "Incidents": h.incident_count, "Unique locations": h.unique_locations,
                "Corroborated": h.verification_count, "Pending": h.pending_count,
                "Avg severity (0-3)": round(h.average_severity, 2), "Radius (m)": round(h.radius_m),
                "First": h.first_incident_at[:10], "Last": h.last_incident_at[:10],
                "Centre": f"{h.center_latitude:.4f}, {h.center_longitude:.4f}"} for h in hot]),
                hide_index=True, width="stretch")
            sel = st.selectbox("Inspect hotspot", [h.hotspot_id for h in hot])
            h = next(x for x in hot if x.hotspot_id == sel)
            members = df[df.id.isin(h.incident_ids)].sort_values("reported_at")
            a, b = st.columns(2)
            sc = pd.Series(h.severity_counts)
            fig = go.Figure(go.Bar(x=sc.index, y=sc.values, marker_color=[C["grey"], C["yellow"], "#F97316", C["danger"]]))
            a.plotly_chart(plotly_layout(fig, 240, "Severity distribution"), width="stretch")
            t = members.set_index("reported_at").resample("QS").size()
            fig = go.Figure(go.Bar(x=t.index, y=t.values, marker_color=C["accent"]))
            b.plotly_chart(plotly_layout(fig, 240, "Reporting history (per quarter)"), width="stretch")
            st.dataframe(members[["id", "reported_at", "location_name", "user_reported_severity", "verification_status", "is_demo"]],
                         hide_index=True, width="stretch")

with tab_model:
    try:
        met = json.loads((MODELS_DIR / "risk_model_metrics.json").read_text())
    except Exception:
        met = None
    if not met:
        st.warning("Risk model metrics not found. Run `python scripts/train_model.py`.")
    else:
        md = met["metadata"]
        st.markdown(f"**Target:** {md['target']}.  \n**Version:** `{md['model_version']}` trained {md['trained_at']}.  \n"
                    f"**Data:** {md['stations_used']} NOAA GSOD stations (West/Central Africa), rainfall = {md.get('rainfall_aggregation', '')}; "
                    f"labels from {md['dfo_events_in_region']} DFO flood events.  \n"
                    f"**Split:** train {md['train_period'][0]} to {md['train_period'][1]}, "
                    f"validation to {md['validation_period'][1]}, test {md['test_period'][0]} to {md['test_period'][1]}.")
        rows = []
        for k, v in met["models"].items():
            t = v["test"]
            rows.append({"Model": k.replace("_", " "), "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"],
                         "Brier skill": t["brier_skill_vs_base_rate"], "Recall": t["recall"], "Precision": t["precision"],
                         "F1": t["f1"], "Test positives": t["positives"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Test-period prevalence {met['models']['random_forest']['test']['prevalence']:.3%}. "
                   "Random Forest is the deployed model (chosen on validation data before test results were seen).")
        sp = met.get("spatial_holdout_nigeria")
        if sp:
            t = sp["test"]
            st.markdown(f"**Spatial hold-out (Nigeria never seen in training):** ROC-AUC {t['roc_auc']:.3f}, "
                        f"PR-AUC {t['pr_auc']:.3f}, {t['positives']} positives over {sp['test_stations']} stations.")
        imp = pd.DataFrame(met["permutation_importance"])
        fig = go.Figure(go.Bar(y=imp["label"], x=imp["auc_drop_mean"], orientation="h", marker_color=C["accent"],
                               error_x=dict(array=imp["auc_drop_std"])))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_layout(fig, 400, "Permutation importance on test data (drop in ROC-AUC)"), width="stretch")
        cal = pd.DataFrame(met["models"]["random_forest"]["test"]["calibration"])
        fig = go.Figure()
        fig.add_scatter(x=cal["mean_predicted"], y=cal["observed_rate"], mode="lines+markers", name="Random Forest",
                        line=dict(color=C["accent"]))
        mx = float(max(cal["mean_predicted"].max(), cal["observed_rate"].max()))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect calibration", line=dict(dash="dot", color=C["muted"]))
        st.plotly_chart(plotly_layout(fig, 300, "Calibration on test data"), width="stretch")
        st.warning("Limitations: trained on large, regionally reported floods (DFO), not street-level flooding. "
                   "It has not been validated on Port Harcourt street incidents. Training used observed station rainfall "
                   "in place of forecasts; live forecasts add error. The local Flood Memory adjustment is rule-based, not learned.")
disclaimer()
