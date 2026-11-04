"""Administration: review reports, model status and system statistics."""

from __future__ import annotations

import hmac
import json

import pandas as pd
import streamlit as st

from components.state import get_conn, incidents_df, region, settings
from components.theme import STATUS_COLORS, badge, disclaimer, esc, kv_rows, page_header
from config import MODELS_DIR
from database import repository as repo
from services.image_service import resolve_upload
from services.incident_service import evidence_score, independent_nearby

S = settings()
page_header("Administration", "Review pending reports and system status.")

if not S.admin_password:
    st.warning("Administration is disabled: set `FLOODPRINT_ADMIN_PASSWORD` in the environment or Streamlit secrets.")
    st.stop()
if not st.session_state.get("admin_ok"):
    with st.form("login"):
        pw = st.text_input("Admin password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(pw.encode(), S.admin_password.encode()):
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

conn = get_conn()
if st.button("Sign out"):
    st.session_state["admin_ok"] = False
    st.rerun()

tab_rev, tab_sys = st.tabs(["Review queue", "System and models"])
with tab_rev:
    df = incidents_df()
    status = st.radio("Show", ["PENDING", "CORROBORATED", "REJECTED"], horizontal=True)
    sub = df[df["verification_status"] == status] if not df.empty else df
    if sub.empty:
        st.info(f"No {status.lower()} reports.")
    for r in sub.head(30).itertuples():
        inc = repo.get_incident(conn, int(r.id))
        ev = evidence_score(inc, independent_nearby(df, inc))
        with st.expander(f"#{inc.id} {inc.location_name}{' [DEMO]' if inc.is_demo else ''} - {inc.user_reported_severity} - "
                         f"{inc.reported_at[:16]} - evidence {ev['score']}/100"):
            a, b = st.columns([1, 1.3])
            img = resolve_upload(S.uploads_dir, inc.image_path)
            if img:
                a.image(str(img), width="stretch")
            else:
                a.caption("No photo attached.")
            ai = inc.ai_details or {}
            rows = [("Status", badge(inc.verification_status, STATUS_COLORS[inc.verification_status])),
                    ("Type", esc(inc.incident_type)), ("Description", esc(inc.description)),
                    ("Coordinates", f"{inc.latitude:.5f}, {inc.longitude:.5f}"),
                    ("AI classification", esc(f"{inc.ai_classification} ({inc.ai_confidence:.0%})" if inc.ai_classification else ai.get("error", "not analysed"))),
                    ("Possible duplicate of", f"#{inc.duplicate_of}" if inc.duplicate_of else "none"),
                    ("Quality flags", esc(", ".join(inc.quality_flags) or "none")),
                    ("Evidence score", f"{ev['score']}/100 ({ev['level']}) - rule-based checklist")]
            b.markdown(f"<div class='fp-card'>{kv_rows(rows)}</div>", unsafe_allow_html=True)
            b.caption("; ".join(f"{f['factor']} ({f['points']:+d})" for f in ev["factors"]) or "No evidence factors.")
            if inc.rainfall_context:
                b.caption("Rainfall context: " + json.dumps(inc.rainfall_context)[:300])
            note = b.text_input("Review note", key=f"note{inc.id}")
            c1, c2, c3 = b.columns(3)
            if c1.button("Corroborate", key=f"ok{inc.id}", type="primary"):
                repo.update_verification(conn, inc.id, "CORROBORATED", note or None)
                st.cache_data.clear()
                st.rerun()
            if c2.button("Reject", key=f"no{inc.id}"):
                repo.update_verification(conn, inc.id, "REJECTED", note or None)
                st.cache_data.clear()
                st.rerun()
            if c3.button("Reset to pending", key=f"re{inc.id}"):
                repo.update_verification(conn, inc.id, "PENDING", note or None)
                st.cache_data.clear()
                st.rerun()

with tab_sys:
    c = repo.count_incidents(conn)
    w = conn.execute("SELECT source, COUNT(*) n, MAX(observed_at) last FROM weather_observations GROUP BY source").fetchall()
    rp = conn.execute("SELECT COUNT(*), MAX(created_at) FROM risk_predictions").fetchone()
    hist = conn.execute("SELECT COUNT(*) FROM location_history").fetchone()[0]
    sys_rows = [("Incidents", str(c["TOTAL"])),
                ("Pending / corroborated / rejected", f"{c['PENDING']} / {c['CORROBORATED']} / {c['REJECTED']}"),
                ("Demo incidents", str(c["DEMO"])), ("Flood Memory entries (corroborated)", str(hist)),
                ("Stored risk predictions", f"{rp[0]} (latest {rp[1]})")]
    sys_rows += [(f"Weather rows: {x['source']}", f"{x['n']} (latest {x['last']})") for x in w]
    st.markdown(f"<div class='fp-card'>{kv_rows(sys_rows)}</div>", unsafe_allow_html=True)
    st.write("")
    for name, file in (("Risk model", "risk_model_metrics.json"), ("Flood Vision head", "vision_metrics.json")):
        p = MODELS_DIR / file
        st.markdown(f"**{name}:** {'available' if p.exists() else 'missing'}")
        if p.exists():
            with st.expander(f"{name} metrics file"):
                st.json(json.loads(p.read_text()), expanded=False)
    st.markdown("**Demo data**")
    if st.button("Delete all demonstration records", type="secondary"):
        n = repo.delete_demo_data(conn)
        st.cache_data.clear()
        st.success(f"Deleted {n} demo incidents.")
disclaimer()
