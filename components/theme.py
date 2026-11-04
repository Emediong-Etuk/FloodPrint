"""Design system: CSS, colour tokens, reusable HTML snippets."""

from __future__ import annotations

import html

import streamlit as st

from config import DISCLAIMER

C = {
    "bg": "#0B1220", "bg2": "#111C2E", "card": "#162338", "accent": "#38BDF8",
    "success": "#22C55E", "warning": "#F59E0B", "danger": "#EF4444",
    "text": "#F8FAFC", "muted": "#94A3B8", "yellow": "#EAB308", "grey": "#64748B",
}
SEVERITY_COLORS = {"severe": C["danger"], "moderate": "#F97316", "minor": C["yellow"], "none": C["grey"]}
STATUS_COLORS = {"CORROBORATED": C["success"], "PENDING": C["warning"], "REJECTED": C["danger"]}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Inter', system-ui, sans-serif !important;
}}
.stApp {{ background: {C['bg']}; }}
section[data-testid="stSidebar"] {{ background: {C['bg2']}; border-right: 1px solid #1E2D45; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.fp-title {{ font-size: 1.9rem; font-weight: 700; margin: 0; color: {C['text']}; }}
.fp-sub {{ color: {C['muted']}; font-size: .92rem; margin-top: .15rem; }}
.fp-card {{ background: {C['card']}; border: 1px solid #22324D; border-radius: 12px; padding: 16px 18px; height: 100%; }}
.fp-metric-label {{ color: {C['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.fp-metric-value {{ font-size: 2rem; font-weight: 700; color: {C['text']}; line-height: 1.15; margin-top: 4px; }}
.fp-metric-note {{ color: {C['muted']}; font-size: .8rem; margin-top: 4px; }}
.fp-badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .74rem; font-weight: 600;
  border: 1px solid currentColor; }}
.fp-banner {{ background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.45); color: #FCD34D;
  border-radius: 10px; padding: 8px 14px; font-size: .85rem; margin-bottom: 12px; }}
.fp-info {{ background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.35); color: #BAE6FD;
  border-radius: 10px; padding: 8px 14px; font-size: .85rem; margin-bottom: 12px; }}
.fp-disclaimer {{ color: {C['muted']}; font-size: .76rem; border-top: 1px solid #1E2D45; padding-top: 10px; margin-top: 28px; }}
.fp-header-strip {{ display: flex; gap: 22px; flex-wrap: wrap; color: {C['muted']}; font-size: .85rem; margin: 6px 0 16px; }}
.fp-header-strip b {{ color: {C['text']}; font-weight: 600; }}
.fp-row {{ display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid #1E2D45; font-size: .88rem; }}
.fp-row:last-child {{ border-bottom: none; }}
.fp-legend span {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: .82rem; color: {C['muted']}; }}
.fp-dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
.fp-ring {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; border: 2px dashed; }}
div[data-testid="stMetric"] {{ background: {C['card']}; border-radius: 12px; padding: 12px 16px; border: 1px solid #22324D; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<p class='fp-title'>{esc(title)}</p><p class='fp-sub'>{esc(subtitle)}</p>", unsafe_allow_html=True)


def metric_card(label: str, value, note: str = "", color: str | None = None) -> str:
    style = f"color:{color}" if color else ""
    return (f"<div class='fp-card'><div class='fp-metric-label'>{esc(label)}</div>"
            f"<div class='fp-metric-value' style='{style}'>{esc(value)}</div>"
            f"<div class='fp-metric-note'>{esc(note)}</div></div>")


def badge(text: str, color: str) -> str:
    return f"<span class='fp-badge' style='color:{color}'>{esc(text)}</span>"


def banner(text: str, kind: str = "warn") -> None:
    cls = "fp-banner" if kind == "warn" else "fp-info"
    st.markdown(f"<div class='{cls}'>{esc(text)}</div>", unsafe_allow_html=True)


def demo_banner(n_demo: int) -> None:
    if n_demo:
        banner(f"Demonstration data: {n_demo} records shown here are synthetic examples labelled [DEMO]. "
               "They are not real incidents. Rainfall context on historical demo records is real NOAA station data.")


def disclaimer() -> None:
    st.markdown(f"<div class='fp-disclaimer'>{esc(DISCLAIMER)}</div>", unsafe_allow_html=True)


def kv_rows(rows: list[tuple[str, str]]) -> str:
    return "".join(f"<div class='fp-row'><span style='color:{C['muted']}'>{esc(k)}</span><span>{v}</span></div>"
                   for k, v in rows)


def plotly_layout(fig, height: int = 320, title: str | None = None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=C["card"], plot_bgcolor=C["card"], height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10), font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        title=dict(text=title, font=dict(size=14)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#22324D", zeroline=False)
    fig.update_yaxes(gridcolor="#22324D", zeroline=False)
    return fig
