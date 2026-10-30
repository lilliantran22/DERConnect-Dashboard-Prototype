"""Styling helpers: CSS, KPI cards, banner, pills."""
import streamlit as st

CAT_COLOR = {"Unexplained": "#E4572E", "Explained": "#3A86FF", "Data issue": "#9AA0A6"}
STATUS_COLOR = {"Check": "#E4572E", "OK": "#2E9E5B", "Explained": "#3A86FF", "Data issue": "#9AA0A6"}

CSS = """<style>
.block-container{padding-top:1.2rem;max-width:1280px}
.hero{background:linear-gradient(120deg,#182B49 0%,#00629B 100%);color:#fff;padding:22px 28px;border-radius:16px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.7rem;color:#fff;padding:0}
.hero p{margin:4px 0 0;opacity:.85}
.banner{padding:14px 18px;border-radius:12px;margin-bottom:14px;border-left:6px solid;color:#1D2433;font-size:1.02rem}
.kpi{background:#fff;border:1px solid #E6E9EF;border-radius:14px;padding:14px 16px;box-shadow:0 1px 3px rgba(24,43,73,.06);border-top:4px solid var(--c,#00629B)}
.kpi .l{font-size:.74rem;color:#5b6577;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:1.85rem;font-weight:700;color:#182B49;line-height:1.2}
.kpi .s{font-size:.8rem;color:#7a8497}
.card{background:#fff;border:1px solid #E6E9EF;border-radius:14px;padding:14px 16px;margin-bottom:10px;box-shadow:0 1px 3px rgba(24,43,73,.06)}
.card .t{font-weight:600;color:#182B49}
.card .m{color:#5b6577;font-size:.88rem}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.74rem;font-weight:600;color:#fff}
</style>"""

BANNERS = {
    "attention": ("#FDECEA", "#D64530", "<b>Needs attention.</b> Something unexplained happened in the last 48 hours."),
    "watch": ("#FFF6E0", "#E0A100", "<b>Keep an eye on it.</b> Unexplained activity in the past 7 days, nothing urgent right now."),
    "healthy": ("#E8F6EE", "#2E9E5B", "<b>All clear.</b> No unexplained activity in the past 7 days."),
}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{sub}</p></div>", unsafe_allow_html=True)


def banner(status):
    bg, border, text = BANNERS[status]
    st.markdown(f"<div class='banner' style='background:{bg};border-color:{border}'>{text}</div>", unsafe_allow_html=True)


def kpi(col, label, value, sub="", color="#00629B"):
    col.markdown(f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div><div class='s'>{sub}&nbsp;</div></div>", unsafe_allow_html=True)


def pill(text, color):
    return f"<span class='pill' style='background:{color}'>{text}</span>"
