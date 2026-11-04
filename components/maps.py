"""Folium map builder. Observed reports are solid dots; predictions are dashed rings."""

from __future__ import annotations

import folium
import pandas as pd

from components.theme import C, SEVERITY_COLORS, esc
from config import Region

RISK_RING = {"Low": C["success"], "Moderate": C["yellow"], "High": C["warning"], "Very High": C["danger"]}


def incident_color(row) -> str:
    if row.get("verification_status") != "CORROBORATED" or row.get("historical"):
        return C["grey"]
    return SEVERITY_COLORS.get(row.get("user_reported_severity"), C["grey"])


def build_map(region: Region, incidents: pd.DataFrame | None = None, risks=None, hotspots=None,
              height_center=None, zoom: int | None = None, current_days: int = 7) -> folium.Map:
    lat, lon = height_center or (region.center_lat, region.center_lon)
    m = folium.Map(location=[lat, lon], zoom_start=zoom or region.default_zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap (CARTO)", attr="© OpenStreetMap contributors © CARTO").add_to(m)

    if risks:
        g = folium.FeatureGroup(name="Predicted risk (estimate)")
        for r in risks:
            col = RISK_RING[r.risk_category]
            pop = (f"<b>{esc(r.location_name)}</b><br><i>PREDICTED RISK - estimate, not a report</i><br>"
                   f"Risk: <b>{r.risk_category}</b> ({r.risk_probability:.1%}) - {esc(r.prediction.window)}<br>"
                   f"Model probability: {r.model_probability:.1%}<br>"
                   f"Rain last 24 h: {r.weather.rain_last_24h_mm or 0:.1f} mm<br>"
                   f"Forecast 7 days: {r.weather.forecast_7d_mm or 0:.1f} mm<br>"
                   f"Nearby corroborated incidents: {r.corroborated_nearby}<br>"
                   f"Model {esc(r.prediction.model_version)} | data {esc(r.weather.observed_at)}")
            folium.Circle([r.latitude, r.longitude], radius=520, color=col, weight=2, dash_array="6 6",
                          fill=True, fill_opacity=0.06, popup=folium.Popup(pop, max_width=320),
                          tooltip=f"{r.location_name}: predicted {r.risk_category}").add_to(g)
        g.add_to(m)

    if hotspots:
        g = folium.FeatureGroup(name="Hotspots (DBSCAN)")
        for h in hotspots:
            confirmed = h["scope"] == "corroborated"
            col = C["danger"] if confirmed else C["accent"]
            pop = (f"<b>Hotspot {esc(h.get('hotspot_id', h.get('id')))}</b> "
                   f"({'corroborated reports only' if confirmed else 'includes unverified reports'})<br>"
                   f"Incidents: {h['incident_count']} | locations: {h['unique_locations']}<br>"
                   f"Corroborated: {h['verification_count']} | pending: {h['pending_count']}<br>"
                   f"Average severity: {h['average_severity']:.1f} / 3<br>Last incident: {esc(str(h['last_incident_at'])[:16])}")
            folium.Circle([h["center_latitude"], h["center_longitude"]], radius=max(h["radius_m"], 120) + 60,
                          color=col, weight=1.5, fill=True, fill_opacity=0.14,
                          popup=folium.Popup(pop, max_width=300),
                          tooltip=f"Hotspot: {h['incident_count']} incidents").add_to(g)
        g.add_to(m)

    if incidents is not None and not incidents.empty:
        now = pd.Timestamp.now(tz="UTC")
        cur = folium.FeatureGroup(name=f"Reported incidents (last {current_days} days)")
        hist = folium.FeatureGroup(name="Historical / unverified incidents")
        for row in incidents.to_dict("records"):
            recent = (now - row["reported_at"]) <= pd.Timedelta(days=current_days)
            row["historical"] = not recent
            col = incident_color(row)
            ai = row.get("ai_classification") or "not analysed"
            demo = " [DEMO]" if row.get("is_demo") else ""
            pop = (f"<b>#{row['id']} {esc(row['location_name'])}</b>{demo}<br>"
                   f"<i>OBSERVED REPORT</i><br>{esc(row['incident_type'])} - severity {esc(row['user_reported_severity'])}<br>"
                   f"Status: <b>{esc(row['verification_status'])}</b><br>"
                   f"Reported: {row['reported_at'].strftime('%Y-%m-%d %H:%M')} UTC<br>AI photo analysis: {esc(ai)}<br>"
                   f"{esc((row.get('description') or '')[:140])}")
            folium.CircleMarker([row["latitude"], row["longitude"]], radius=7 if recent else 5, color="#0B1220",
                                weight=1, fill=True, fill_color=col, fill_opacity=0.95 if recent else 0.6,
                                popup=folium.Popup(pop, max_width=300),
                                tooltip=f"#{row['id']} {row['location_name']} ({row['verification_status'].lower()})"
                                ).add_to(cur if recent else hist)
        hist.add_to(m)
        cur.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def legend_html() -> str:
    return (
        "<div class='fp-legend'>"
        f"<span><i class='fp-dot' style='background:{C['danger']}'></i>Severe report (corroborated)</span>"
        f"<span><i class='fp-dot' style='background:#F97316'></i>Moderate report</span>"
        f"<span><i class='fp-dot' style='background:{C['yellow']}'></i>Minor report</span>"
        f"<span><i class='fp-dot' style='background:{C['grey']}'></i>Historical or unverified report</span>"
        f"<span><i class='fp-ring' style='border-color:{C['yellow']}'></i>Predicted risk (dashed ring: estimate)</span>"
        f"<span><i class='fp-ring' style='border-color:{C['success']}'></i>Low predicted risk</span>"
        f"<span><i class='fp-dot' style='background:rgba(239,68,68,.35)'></i>Hotspot cluster</span>"
        "</div>"
    )
