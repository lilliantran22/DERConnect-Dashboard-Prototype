"""Home Grid Health: prototype dashboard for time series anomaly detection on electricity data.
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.data_loader import load_csv
from src.detect import evaluate, false_alarm_list, run_detection
from src.indicators import (CATS, KIND_LABEL, appliance_status, classify_events,
                            explain_event, home_status, kpis)
from src.knowledge import INDICATORS, TYPE_NOTES, example_figure
from src.quality import completeness, find_quality_issues
from src.simulate import simulate_home, simulate_peers, simulate_training
from src.zeroshot import ZeroShotAR

st.set_page_config(page_title="Home Grid Health", layout="wide")
ui.inject_css()

BL, ZS = "Per-home baseline", "Zero-shot (AR)"
METHOD_COLOR = {BL: "#182B49", ZS: "#E4572E"}
EMPTY_CTX = pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([]), "label": []})


@st.cache_data
def get_sim():
    end = pd.Timestamp.now().floor("D")
    df, truth, ctx = simulate_home(seed=1, end=end)
    return df, truth, ctx, simulate_peers(end=end)


@st.cache_resource
def get_model():
    """Trained once on simulated homes that are never the home being scored."""
    return ZeroShotAR().fit(simulate_training(end=pd.Timestamp("2026-01-01")))


def style(fig, h=400):
    fig.update_layout(height=h, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def status_style(v):
    t = str(v)
    if t.startswith(("Built", "Caught")):
        return "background-color:#E3F4EA;color:#17653B"
    if t.startswith("Partial"):
        return "background-color:#FFF3D6;color:#8A6100"
    if t.startswith("Missed"):
        return "background-color:#FDE7E3;color:#A32B16"
    if t.startswith("Planned"):
        return "background-color:#ECEFF3;color:#4A5568"
    return ""


def styled(df, cols):
    s = df.style
    return (getattr(s, "map", None) or s.applymap)(status_style, subset=cols)


def analyze(d, ctx, model, sens):
    scored, ev = run_detection(d, sens, model=model)
    q = find_quality_issues(d)
    return scored, classify_events(ev, q, ctx), q


def alerts_only(ev):
    return ev[ev.category == "Unexplained"] if len(ev) else ev


@st.cache_data(show_spinner=False)
def sweep(d, truth, ctx, _model, method, grid):
    rows = []
    for s in grid:
        _, ev, _ = analyze(d, ctx, _model, s)
        m, _t = evaluate(alerts_only(ev), truth)
        rows.append(dict(sensitivity=s, method=method, precision=m["precision"], recall=m["recall"]))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cold_start(d, truth, ctx, _model, sens, days_list):
    rows, end = [], d.timestamp.max()
    for n in days_list:
        sub = d[d.timestamp > end - pd.Timedelta(days=n)]
        tr = truth[truth.start >= sub.timestamp.min()]
        for name, mdl in [(BL, None), (ZS, _model)]:
            _, ev, _ = analyze(sub, ctx, mdl, sens)
            m, _t = evaluate(alerts_only(ev), tr)
            rows.append(dict(days=n, method=name, recall=m["recall"], false_alarms=m["false_alarms"]))
    return pd.DataFrame(rows)


# ---------- sidebar ----------
st.sidebar.title("Settings")
source = st.sidebar.radio("Data source", ["Simulated home", "Upload CSV"])
method = st.sidebar.selectbox("Detection method", [BL, ZS])
sensitivity = st.sidebar.slider("Sensitivity (lower = more alerts)", 4.0, 10.0, 6.0, 0.5)
price = st.sidebar.number_input("Electricity price ($/kWh)", 0.05, 1.50, 0.40, 0.01)

truth, peers, ctx, full_df = None, None, EMPTY_CTX, None
if source == "Simulated home":
    df, truth, ctx, peers = get_sim()
    full_df = df
    hist = st.sidebar.slider("Days of history available", 1, 14, 14,
                             help="Shrink this to mimic a newly installed home.")
    df = df[df.timestamp > df.timestamp.max() - pd.Timedelta(days=hist)]
    truth = truth[truth.start >= df.timestamp.min()]
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
model = get_model()
cur_model = model if method == ZS else None
scored, events, quality = analyze(df, ctx, cur_model, sensitivity)
now = df.timestamp.max()
subs = sorted(df.submeter.unique())
chosen = st.sidebar.multiselect("Appliance categories", subs, default=subs)
scored = scored[scored.submeter.isin(chosen)]
if len(events):
    events = events[events.submeter.isin(chosen)].reset_index(drop=True)
if len(quality):
    quality = quality[quality.submeter.isin(chosen)].reset_index(drop=True)

# ---------- header ----------
ui.hero("Home Grid Health", f"Detecting unusual electricity activity · {method} · data through {now:%b %d, %Y}")
ui.banner(home_status(events, now))
k = kpis(events, quality, now, price)
c = st.columns(5)
ui.kpi(c[0], "Unexplained events", k["unexplained"], "last 7 days", ui.CAT_COLOR["Unexplained"])
ui.kpi(c[1], "Explained events", k["explained"], "known activity", ui.CAT_COLOR["Explained"])
ui.kpi(c[2], "Data issues", k["data_issues"], "meter / data problems", ui.CAT_COLOR["Data issue"])
ui.kpi(c[3], "Extra energy", f"{k['excess_kwh']:.1f} kWh", f"about ${k['est_cost']:.2f}", "#182B49")
ui.kpi(c[4], "Appliances affected", k["devices"], "unexplained only", "#FFCD00")
st.write("")

tabs = st.tabs(["Overview", "Appliances", "Neighborhood", "Events", "Data quality",
                "Detection errors", "Anomaly guide", "Coverage & checks"])

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
    if method == ZS:
        st.caption("Zero-shot: the model was trained on other homes only. Its 'Expected' line is a one-step-ahead "
                   "forecast, so it follows the actual line closely; alerts come from the forecast error.")
        if source == "Simulated home":
            st.caption(f"Trained on {model.n_series_} series from 6 simulated homes. This home was never in training.")
    st.subheader("Latest alerts")
    unx = alerts_only(events).sort_values("start", ascending=False).head(3) if len(events) else events
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
    mine = df.groupby("timestamp")["kw"].sum().mul(0.25).resample("D").sum().iloc[:-1] if peers is not None else None
    if peers is None:
        st.info("Peer comparison needs multiple buildings (simulated here; DERConnect later).")
    elif len(mine) < 2:
        st.info("Not enough history yet for a peer comparison.")
    else:
        p = peers.reindex(mine.index)
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
            f"Last {min(7, len(mine))} days: your home used {ratio:.0%} of the peer average "
            f"({'outside' if abs(ratio - 1) > 0.3 else 'within'} the +/-30% band).")

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

# ---------- detection errors ----------
with tabs[5]:
    st.subheader("False positives, false negatives, and zero-shot performance")
    if truth is None or full_df is None:
        st.info("Error analysis needs ground-truth labels, which only exist for the simulated home.")
    elif truth.empty:
        st.info("No true anomalies fall inside the selected history window. Increase the days of history.")
    else:
        unx = alerts_only(events)
        m, tab = evaluate(unx, truth)
        fa = false_alarm_list(unx, truth)
        a, b, c3, d = st.columns(4)
        ui.kpi(a, "Recall", f"{m['recall']:.0%}", f"{int(tab.detected.sum())} of {len(tab)} true events caught")
        ui.kpi(b, "Precision", "n/a" if m["precision"] != m["precision"] else f"{m['precision']:.0%}",
               f"{m['alerts']} alerts raised")
        ui.kpi(c3, "False positives", m["false_alarms"], "alerts with no true anomaly", "#E4572E")
        ui.kpi(d, "False negatives", int((~tab.detected).sum()), "true anomalies missed", "#E0A100")
        st.write("")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**False negatives (missed), {method}**")
            miss = tab[~tab.detected][["type", "submeter", "note"]]
            st.dataframe(miss, hide_index=True, width="stretch") if len(miss) else st.success("None missed.")
        with right:
            st.markdown(f"**False positives (no true anomaly), {method}**")
            if len(fa):
                st.dataframe(fa.assign(kind=fa.kind.map(KIND_LABEL))[["submeter", "start", "kind", "peak_z"]]
                             .astype({"start": str}).round(2), hide_index=True, width="stretch")
            else:
                st.success("None.")

        st.markdown("**Sensitivity trade-off** (alerts only; each method evaluated on this history window)")
        grid = tuple(x / 2 for x in range(8, 25))
        sw = pd.concat([sweep(df, truth, ctx, None, BL, grid), sweep(df, truth, ctx, model, ZS, grid)])
        fig = go.Figure()
        for meth, dash_ in [(BL, "solid"), (ZS, "solid")]:
            sub = sw[sw.method == meth]
            fig.add_scatter(x=sub.sensitivity, y=sub.recall, name=f"{meth}: recall",
                            line=dict(color=METHOD_COLOR[meth], dash="solid"))
            fig.add_scatter(x=sub.sensitivity, y=sub.precision, name=f"{meth}: precision",
                            line=dict(color=METHOD_COLOR[meth], dash="dot"))
        fig.add_vline(x=sensitivity, line_dash="dash", line_color="#8A94A6")
        fig.update_xaxes(title="Sensitivity threshold (higher = fewer alerts)")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Solid = recall (fewer false negatives), dotted = precision (fewer false positives). "
                   "Dashed vertical line = current sensitivity.")

        st.markdown("**Detection by anomaly type** (current sensitivity)")
        cmp_ = {}
        for meth, mdl in [(BL, None), (ZS, model)]:
            _, ev_m, _ = analyze(df, ctx, mdl, sensitivity)
            _, t_m = evaluate(alerts_only(ev_m), truth)
            cmp_[meth] = t_m.set_index("type")
        rows = []
        for t_ in cmp_[BL].index:
            r = dict(Type=TYPE_NOTES[t_]["name"])
            for meth in (BL, ZS):
                x = cmp_[meth].loc[t_]
                r[meth] = "Caught" if x.detected else "Missed"
                r[f"{meth} latency (h)"] = round(x.latency_h, 2) if x.detected else None
            rows.append(r)
        st.dataframe(styled(pd.DataFrame(rows), [BL, ZS]), hide_index=True, width="stretch")

        st.markdown("**Cold start: how much history does each method need?**")
        cs = cold_start(full_df, get_sim()[1], ctx, model, sensitivity, (1, 2, 3, 5, 7, 10, 14))
        c1, c2 = st.columns(2)
        for col, metric, title in [(c1, "recall", "Recall by days of history"),
                                   (c2, "false_alarms", "False positives by days of history")]:
            f = go.Figure()
            for meth in (BL, ZS):
                sub = cs[cs.method == meth]
                f.add_scatter(x=sub.days, y=sub[metric], name=meth, line=dict(color=METHOD_COLOR[meth]))
            f.update_xaxes(title="Days of history available")
            f.update_layout(title=dict(text=title, font=dict(size=14)))
            col.plotly_chart(style(f, 300), width="stretch")
        st.caption("Zero-shot needs no history of the new home beyond the window it scores, because its parameters "
                   "come from other homes. The per-home baseline learns the time-of-week pattern from the home's own "
                   "days, so it has little to compare against early on. Short windows contain fewer true anomalies, "
                   "so recall there rests on few events. Simulated data; the injected anomalies are large.")

# ---------- anomaly guide ----------
with tabs[6]:
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
    st.subheader("Detection methods")
    st.markdown("""
| Method | How it works | Needs history of the new home? |
|---|---|---|
| **Per-home baseline** | Expected = typical value for that time of week in this home's own history; alert when a reading, or its 2-hour average, is far from expected | Yes |
| **Zero-shot (AR)** | Linear autoregression on first differences, trained on other homes and applied unchanged to a new one; alert on large forecast error | No |
""")
    st.markdown("**Limits:** 14 days of simulated data, no weather or occupancy input yet, and the zero-shot model is a "
                "placeholder (AR) until the team picks a detector.")

# ---------- coverage & checks ----------
with tabs[7]:
    st.subheader("Coverage: anomaly types")
    if truth is not None and not truth.empty:
        m_raw, tab_raw = evaluate(events, truth)
        m_alert, _ = evaluate(alerts_only(events), truth)
        res = dict(zip(tab_raw.type, tab_raw.detected))
        lat = dict(zip(tab_raw.type, tab_raw.latency_h))
    else:
        res, lat, m_raw, m_alert = {}, {}, None, None
    cov = pd.DataFrame([dict(
        Type=n["name"], **{"Looks like": n["looks"],
                           "Test result": ("Caught" if res.get(k) else "Missed") if k in res else "-",
                           "Latency (h)": round(lat[k], 2) if k in lat and lat[k] == lat[k] else None,
                           "Label today": n["label"], "Gap": n["gap"]}) for k, n in TYPE_NOTES.items()])
    st.dataframe(styled(cov, ["Test result"]), hide_index=True, width="stretch")
    st.subheader("Coverage: dashboard indicators")
    st.dataframe(styled(pd.DataFrame(INDICATORS, columns=["Indicator", "Source", "Status", "Where"]), ["Status"]),
                 hide_index=True, width="stretch")
    st.subheader(f"Detector check: {method} (simulated data only)")
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
