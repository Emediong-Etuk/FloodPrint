"""Report Flood: photo upload, AI analysis, location, submission."""

from __future__ import annotations

from datetime import datetime, timezone

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from ai.flood_vision import CLASS_LABELS, heatmap_overlay
from components.state import get_conn, get_vision, get_weather_service, region, settings
from components.theme import C, badge, disclaimer, esc, kv_rows, page_header
from database.models import INCIDENT_TYPES
from services.image_service import ImageValidationError, perceptual_hash, save_image, validate_image
from services.incident_service import IncidentValidationError, submit_incident

R, S = region(), settings()
page_header("Report Flood", "Share what you see. Reports start as PENDING until reviewed.")

st.markdown("##### 1. Photo")
up = st.file_uploader("Upload a photo of the location (JPG, PNG or WEBP, up to "
                      f"{S.max_upload_mb:g} MB)", type=["jpg", "jpeg", "png", "webp"])
vimg = None
vision_result = None
if up is not None:
    data = up.getvalue()
    try:
        vimg = validate_image(data, up.name, S.max_upload_mb)
    except ImageValidationError as exc:
        st.error(str(exc))
    if vimg:
        key = f"vision_{hash(data)}"
        c1, c2 = st.columns([1, 1], gap="medium")
        c1.image(vimg.image, caption=f"{vimg.width}x{vimg.height} {vimg.format}", width="stretch")
        with c2:
            if key not in st.session_state:
                try:
                    with st.spinner("Analysing photo with Flood Vision..."):
                        st.session_state[key] = get_vision().analyze(vimg.image).to_dict()
                except Exception as exc:
                    st.session_state[key] = {"analysis_status": "unavailable", "error": str(exc)}
            vision_result = st.session_state[key]
            if vision_result.get("analysis_status") != "completed":
                st.warning(f"AI analysis unavailable: {vision_result.get('error')}. You can still submit the report.")
            else:
                v = vision_result
                col = {"no_visible_flooding": C["success"], "waterlogging_minor": C["yellow"],
                       "moderate_flooding": "#F97316", "severe_flooding": C["danger"], "uncertain": C["muted"]}[v["classification"]]
                st.markdown(f"**AI photo analysis** {badge(v['label'], col)}", unsafe_allow_html=True)
                rows = [("Confidence", f"{v['confidence']:.0%}{' (low)' if v['low_confidence'] else ''}"),
                        ("Water detected", "Yes" if v["water_detected"] else "No")]
                rows += [(f"P({k.replace('_', ' ')})", f"{p:.0%}") for k, p in v["class_probabilities"].items()]
                if v.get("severity_probabilities"):
                    rows.append(("Severe vs moderate (zero-shot)", f"{v['severity_probabilities']['severe']:.0%} severe"))
                rows.append(("Model", esc(v["model_name"]) + f" / head {esc(v['head_version'])}"))
                st.markdown(f"<div class='fp-card'>{kv_rows(rows)}</div>", unsafe_allow_html=True)
                for n in v["notes"]:
                    st.caption(n)
                if v.get("heatmap"):
                    with st.expander("Where in the photo does water score highest? (coarse patch map)"):
                        st.image(heatmap_overlay(vimg.image, v["heatmap"]), width="stretch")
                        st.caption("Blue = higher water score from 9 overlapping crops. Not pixel segmentation.")
                st.caption("AI analysis never verifies a report on its own.")

st.markdown("##### 2. Location")
mode = st.radio("How would you like to set the location?", ["Choose a neighbourhood", "Click on the map", "Enter coordinates"],
                horizontal=True)
lat, lon, loc_name = R.center_lat, R.center_lon, ""
if mode == "Choose a neighbourhood":
    loc = st.selectbox("Neighbourhood", [l.name for l in R.locations])
    l = next(x for x in R.locations if x.name == loc)
    lat, lon, loc_name = l.lat, l.lon, loc
elif mode == "Click on the map":
    m = folium.Map([R.center_lat, R.center_lon], zoom_start=R.default_zoom, tiles="CartoDB dark_matter")
    pick = st.session_state.get("picked")
    if pick:
        folium.Marker(pick).add_to(m)
    out = st_folium(m, height=360, width="stretch", returned_objects=["last_clicked"], key="pick_map")
    if out and out.get("last_clicked"):
        st.session_state["picked"] = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
    if st.session_state.get("picked"):
        lat, lon = st.session_state["picked"]
        st.caption(f"Selected {lat:.5f}, {lon:.5f}")
    else:
        st.caption("Click the map to drop a pin.")
else:
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=float(R.center_lat), format="%.5f")
    lon = c2.number_input("Longitude", value=float(R.center_lon), format="%.5f")
if vimg and vimg.gps:
    st.caption(f"The photo contains GPS data ({vimg.gps[0]:.4f}, {vimg.gps[1]:.4f}). It is used only for a consistency check "
               "and is stripped from the stored copy.")

st.markdown("##### 3. Details")
with st.form("report"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Street or neighbourhood name", value=loc_name, max_chars=120)
    itype = c2.selectbox("Incident type", INCIDENT_TYPES)
    sev = c1.select_slider("How bad is it?", ["none", "minor", "moderate", "severe"], value="moderate",
                           help="none = no water, minor = puddles, moderate = road partly covered, severe = impassable / homes affected")
    when = c2.time_input("Time observed (today, local time)", value=datetime.now().time().replace(second=0, microsecond=0))
    desc = st.text_area("What do you see?", max_chars=1000, placeholder="e.g. Water covering both lanes near the junction")
    ok = st.form_submit_button("Submit report", type="primary")

if ok:
    try:
        import pandas as pd

        local = pd.Timestamp.combine(pd.Timestamp.now(tz=R.timezone).date(), when).tz_localize(R.timezone)
        rc = None
        try:
            rc = get_weather_service().get(lat, lon).context_summary()
        except Exception:
            rc = {"source": "Open-Meteo", "error": "weather unavailable at submission"}
        image_path = image_hash = None
        if vimg:
            image_path = save_image(vimg.image, S.uploads_dir)
            image_hash = perceptual_hash(vimg.image)
        res = submit_incident(
            get_conn(), R, latitude=lat, longitude=lon, location_name=name, description=desc, severity=sev,
            incident_type=itype, reported_at=local.to_pydatetime(), image_path=image_path, image_hash=image_hash,
            image_gps=vimg.gps if vimg else None, vision=vision_result, rainfall_context=rc,
            duplicate_radius_m=S.duplicate_radius_m, duplicate_window_h=S.duplicate_window_h)
        st.cache_data.clear()
        st.success(f"Report submitted. Incident ID **#{res.incident_id}** - status PENDING. It now appears on the Live Map.")
        if res.duplicate_of:
            st.warning(f"This looks similar to report #{res.duplicate_of}. It has been flagged for review, not deleted.")
        if res.nearby_reports:
            st.info(f"{res.nearby_reports} other report(s) were made nearby around the same time.")
        if res.flags:
            st.caption("Review flags: " + ", ".join(res.flags))
    except (IncidentValidationError, ValueError) as exc:
        st.error(str(exc))
disclaimer()
