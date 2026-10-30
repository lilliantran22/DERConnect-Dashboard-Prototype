"""Home Grid Health: prototype dashboard for a homeowner/tenant.
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.data_loader import load_csv
from src.detect import evaluate, run_detection
from src.indicators import (CATS, KIND_LABEL, appliance_status, classify_events,
                            explain_event, home_status, kpis)
from src.knowledge import INDICATORS, TYPE_NOTES, example_figure
from src.quality import completeness, find_quality_issues
from src.simulate import simulate_home, simulate_peers

st.set_page_config(page_title="Home Grid Health", page_icon="⚡", layout="wide")
ui.inject_css()
EMPTY_CTX = pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([]), "label": []})


@st.cache_data
def get_sim():
    df, truth, ctx = simulate_home(seed=1)
    return df, truth, ctx, simulate_peers()


def style(fig, h=400):
    fig.update_layout(height=h, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.1), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------- sidebar ----------
st.sidebar.title("Settings")
source = st.sidebar.radio("Data source", ["Simulated home", "Upload CSV"])
sensitivity = st.sidebar.slider("Sensitivity (lower = more alerts)", 4.0, 10.0, 6.0, 0.5)
price = st.sidebar.number_input("Electricity price ($/kWh)", 0.05, 1.50, 0.40, 0.01)

truth, peers, ctx = None, None, EMPTY_CTX
if source == "Simulated home":
    df, truth, ctx, peers = get_sim()
else:
    up = st.sidebar.file_uploader("CSV (timestamp, submeter, kw)", type="csv")
    if up is None:
        ui.hero("Home Grid Health", "Upload a CSV in the sidebar, or switch to the simulated home.")
        st.stop()
    df = load_csv(up)
    df = df[df.building_id == st.sidebar.selectbox("Building", sorted(df.building_id.unique()))]

with st.sidebar.expander("Add a known event (explains alerts)"):
    lab = st.text_input("What happened?", "Guests over")
    d0 = st.date_input("From", df.timestamp.max().date())
    d1 = st.date_input("To", df.timestamp.max().date())
    if st.button("Add event"):
        st.session_state.setdefault("extra_ctx", []).append(
            dict(start=pd.Timestamp(d0), end=pd.Timestamp(d1) + pd.Timedelta("1D"), label=lab))
if st.session_state.get("extra_ctx"):
    ctx = pd.concat([ctx, pd.DataFrame(st.session_state["extra_ctx"])], ignore_index=True)

# ---------- pipeline ----------
scored, events = run_detection(df, sensitivity)
quality = find_quality_issues(df)
events = classify_events(events, quality, ctx)
now = df.timestamp.max()
subs = sorted(df.submeter.unique())
chosen = st.sidebar.multiselect("Appliance categories", subs, default=subs)
scored = scored[scored.submeter.isin(chosen)]
if len(events):
    events = events[events.submeter.isin(chosen)].reset_index(drop=True)
if len(quality):
    quality = quality[quality.submeter.isin(chosen)].reset_index(drop=True)

# ---------- header ----------
ui.hero("Home Grid Health", f"Is anything unusual happening with your electricity? · data through {now:%b %d, %Y}")
ui.banner(home_status(events, now))
k = kpis(events, quality, now, price)
c = st.columns(5)
ui.kpi(c[0], "Unexplained events", k["unexplained"], "last 7 days", ui.CAT_COLOR["Unexplained"])
ui.kpi(c[1], "Explained events", k["explained"], "known activity", ui.CAT_COLOR["Explained"])
ui.kpi(c[2], "Data issues", k["data_issues"], "meter / data problems", ui.CAT_COLOR["Data issue"])
ui.kpi(c[3], "Extra energy", f"{k['excess_kwh']:.1f} kWh", f"≈ ${k['est_cost']:.2f}", "#182B49")
ui.kpi(c[4], "Appliances affected", k["devices"], "unexplained only", "#FFCD00")
st.write("")

tabs = st.tabs(["Overview", "Appliances", "Neighborhood", "Events", "Data quality",
                "Anomaly guide", "Coverage & checks"])

# ---------- overview ----------
with tabs[0]:
    scope = st.selectbox("Show", ["Whole home"] + list(chosen))
    if scope == "Whole home":
        g = scored.groupby("timestamp")[["kw", "expected"]].sum().reset_index()
        ev, q = events, quality
    else:
        g = scored[scored.submeter == scope]
        ev = events[events.submeter == scope] if len(events) else events
        q = quality[quality.submeter == scope] if len(quality) else quality
    fig = go.Figure()
    fig.add_scatter(x=g.timestamp, y=g.kw, name="Actual", line=dict(color="#182B49", width=1.6))
    fig.add_scatter(x=g.timestamp, y=g.expected, name="Expected", line=dict(color="#8A94A6", dash="dash"))
    for _, e in (ev.iterrows() if len(ev) else []):
        fig.add_vrect(x0=e.start - pd.Timedelta("8min"), x1=e.end + pd.Timedelta("8min"),
                      fillcolor=ui.CAT_COLOR[e.category], opacity=0.28, line_width=0)
    for _, r in (q.iterrows() if len(q) else []):
        fig.add_vrect(x0=r.start, x1=r.end, fillcolor=ui.CAT_COLOR["Data issue"], opacity=0.28, line_width=0)
    st.plotly_chart(style(fig, 420), width="stretch")
    st.markdown(" ".join(ui.pill(n, col) for n, col in ui.CAT_COLOR.items()) +
                "&nbsp; shaded areas on the chart", unsafe_allow_html=True)
    st.subheader("Latest alerts")
    unx = events[events.category == "Unexplained"].sort_values("start", ascending=False).head(3) if len(events) else events
    if len(unx) == 0:
        st.write("Nothing unexplained to show.")
    for _, e in (unx.iterrows() if len(unx) else []):
        x = explain_event(e, price)
        st.markdown(f"<div class='card'><div class='t'>{x['title']}</div>"
                    f"<div class='m'>{x['what']}<br>{x['cost']}</div></div>", unsafe_allow_html=True)

# ---------- appliances ----------
with tabs[1]:
    stat = appliance_status(scored, events, quality, now)
    cols = st.columns(max(len(stat), 1))
    for col, (_, r) in zip(cols, stat.iterrows()):
        col.markdown(f"<div class='card'><div class='t'>{r.submeter}</div>"
                     f"<div style='font-size:1.6rem;font-weight:700;color:#182B49'>{r.now_kw:.2f} kW</div>"
                     f"<div class='m'>expected {r.expected_kw:.2f} kW</div><br>"
                     f"{ui.pill(r.status, ui.STATUS_COLOR[r.status])}</div>", unsafe_allow_html=True)
    st.caption("Status covers the last 48 hours. Check = unexplained event; Data issue = meter problem.")

# ---------- neighborhood ----------
with tabs[2]:
    st.subheader("Your home vs similar homes")
    if peers is None:
        st.info("Peer comparison needs multiple buildings (simulated here; DERConnect later).")
    else:
        mine = df.groupby("timestamp")["kw"].sum().mul(0.25).resample("D").sum().iloc[:-1]
        p = peers.iloc[:-1]
        fig = go.Figure()
        fig.add_scatter(x=p.index, y=p.min(axis=1), line=dict(width=0), showlegend=False)
        fig.add_scatter(x=p.index, y=p.max(axis=1), name="Peer range", fill="tonexty",
                        fillcolor="rgba(138,148,166,0.25)", line=dict(width=0))
        fig.add_scatter(x=p.index, y=p.mean(axis=1), name="Peer average", line=dict(color="#8A94A6"))
        fig.add_scatter(x=mine.index, y=mine.values, name="Your home", line=dict(color="#E4572E", width=2.5))
        fig.update_yaxes(title="kWh per day")
        st.plotly_chart(style(fig, 380), width="stretch")
        ratio = mine.iloc[-7:].sum() / p.iloc[-7:].mean(axis=1).sum()
        (st.warning if abs(ratio - 1) > 0.3 else st.success)(
            f"Last 7 days: your home used {ratio:.0%} of the peer average "
            f"({'outside' if abs(ratio - 1) > 0.3 else 'within'} the ±30% band).")

# ---------- events ----------
with tabs[3]:
    if events.empty:
        st.write("No events to show.")
    else:
        pick_cat = st.multiselect("Show", CATS, default=CATS)
        ev = events[events.category.isin(pick_cat)]
        show = ev.assign(type=ev.kind.map(KIND_LABEL))[
            ["id", "start", "submeter", "type", "category", "reason", "duration_h", "excess_kwh", "peak_z"]]
        st.dataframe(show.astype({"start": str}).round(2), hide_index=True, width="stretch")
        if len(ev):
            pick = st.selectbox("Explain event #", ev.id, index=len(ev) - 1)
            e = ev[ev.id == pick].iloc[0]
            x = explain_event(e, price)
            st.markdown(f"### {x['title']} &nbsp;{ui.pill(e.category, ui.CAT_COLOR[e.category])}", unsafe_allow_html=True)
            if e.category == "Explained":
                st.info(f"Matches a known event: **{e.reason}**. Unusual, but expected.")
            elif e.category == "Data issue":
                st.info(f"Overlaps a data problem: **{e.reason}**. Likely the meter or data, not your home.")
            st.write(f"**What happened:** {x['what']}")
            st.write(f"**Why it was flagged:** {x['why']}")
            st.write(f"**Cost impact:** {x['cost']}")
            st.write(f"**Possibly harmless:** {x['harmless']}")
            st.write(f"**What to do:** {x['action']}")

# ---------- data quality ----------
with tabs[4]:
    st.subheader("Is the data itself trustworthy?")
    comp = completeness(df[df.submeter.isin(chosen)])
    cc = st.columns(max(len(comp), 1))
    for col, (name, v) in zip(cc, comp.items()):
        ui.kpi(col, name, f"{v:.1%}", "samples present", "#2E9E5B" if v > 0.99 else "#E0A100")
    st.write("")
    if quality.empty:
        st.success("No missing or stuck-meter problems found.")
    else:
        st.dataframe(quality.astype({"start": str, "end": str}), hide_index=True, width="stretch")
    st.caption("Missing data = no readings for 2+ intervals. Stuck meter = identical value for 2+ hours. "
               "Alerts that overlap these are labeled 'Data issue' instead of 'Unexplained'.")

# ---------- anomaly guide ----------
with tabs[5]:
    st.subheader("What kinds of unusual activity can the dashboard show?")
    st.markdown("Anomaly types follow Lai et al. (2021). Red shading marks the anomaly in each example.")
    for key, n in TYPE_NOTES.items():
        with st.expander(n["name"], expanded=(key == "global")):
            a, b = st.columns([3, 2])
            a.markdown(f"**What it looks like:** {n['looks']}  \n**At home:** {n['home']}  \n"
                       f"**Dashboard label today:** {n['label']}  \n**Known gap:** {n['gap']}")
            b.plotly_chart(example_figure(key), width="stretch", key=f"ex_{key}")
    st.subheader("Reading vs activity")
    st.markdown("""
| Bucket | Meaning | Counted in the banner? |
|---|---|---|
| **Data issue** | The reading is odd because the meter or data is (missing, stuck) | No |
| **Explained** | Real, unusual activity that matches a known event (e.g. guests) | No |
| **Unexplained** | Real, unusual activity with no known reason | **Yes** |
""")
    st.markdown("**How detection works:** expected power = the typical value for that time of week; "
                "an alert fires when a reading, or the 2-hour average, is far from expected. "
                "**Limits:** simple baseline, 14 days of data, no weather or occupancy input yet.")

# ---------- coverage & checks ----------
with tabs[6]:
    st.subheader("Coverage: anomaly types")
    if truth is not None and not truth.empty:
        unx_ev = events[events.category == "Unexplained"] if len(events) else events
        m_raw, tab_raw = evaluate(events, truth)
        m_alert, _ = evaluate(unx_ev, truth)
        res = dict(zip(tab_raw.type, tab_raw.detected))
        lat = dict(zip(tab_raw.type, tab_raw.latency_h))
    else:
        res, lat, m_raw, m_alert = {}, {}, None, None
    cov = pd.DataFrame([dict(
        Type=n["name"], **{"Looks like": n["looks"],
                           "Test result": ("✅ Caught" if res.get(k) else "❌ Missed") if k in res else "-",
                           "Latency (h)": round(lat[k], 2) if k in lat and lat[k] == lat[k] else None,
                           "Label today": n["label"], "Gap": n["gap"]}) for k, n in TYPE_NOTES.items()])
    st.dataframe(cov, hide_index=True, width="stretch")
    st.subheader("Coverage: dashboard indicators")
    st.dataframe(pd.DataFrame(INDICATORS, columns=["Indicator", "Source", "Status", "Where"]),
                 hide_index=True, width="stretch")
    st.subheader("Detector check (simulated data only)")
    if m_raw is None:
        st.info("Ground truth only exists for the simulated home.")
    else:
        a, b, c3, d = st.columns(4)
        ui.kpi(a, "Recall", f"{m_raw['recall']:.0%}", "true events caught")
        ui.kpi(b, "Precision: raw detector", f"{m_raw['precision']:.0%}", f"{m_raw['false_alarms']} false alarm(s)")
        ui.kpi(c3, "Precision: after explaining", f"{m_alert['precision']:.0%}", f"{m_alert['false_alarms']} false alarm(s)")
        ui.kpi(d, "Recall after explaining", f"{m_alert['recall']:.0%}", "")
        st.caption("Event-based scoring (Si et al.): each injected anomaly counts once, 1 h tolerance. "
                   "The injected anomalies are large, so this is a sanity check, not a benchmark.")
