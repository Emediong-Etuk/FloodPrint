"""FLOODPRINT - community flood intelligence and early warning.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FLOODPRINT", page_icon="assets/logo.png" if Path("assets/logo.png").exists() else None,
                   layout="wide", initial_sidebar_state="expanded")

from components.state import get_conn, region  # noqa: E402
from components.theme import apply_theme  # noqa: E402

apply_theme()
get_conn()  # initialise schema (and demo data on an empty database)

PAGES = Path(__file__).parent / "pages"
nav = st.navigation(
    [
        st.Page(PAGES / "1_Dashboard.py", title="Dashboard", default=True),
        st.Page(PAGES / "2_Live_Map.py", title="Live Flood Map", url_path="live-map"),
        st.Page(PAGES / "3_Report_Flood.py", title="Report Flood", url_path="report"),
        st.Page(PAGES / "4_Flood_Intelligence.py", title="Flood Intelligence", url_path="intelligence"),
        st.Page(PAGES / "5_Flood_Memory.py", title="Flood Memory", url_path="memory"),
        st.Page(PAGES / "6_Admin.py", title="Administration", url_path="admin"),
    ]
)
with st.sidebar:
    if Path("assets/logo.png").exists():
        st.image("assets/logo.png", width=180)
    else:
        st.markdown("### FLOODPRINT")
    st.caption("The city remembers where the water goes.")
    st.caption(f"Monitoring: **{region().display_name}**")
nav.run()
