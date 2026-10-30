"""Turn detected events into homeowner-readable indicators.

Every event is put in one of three buckets, which answers 'unusual reading vs unusual activity':
  Data issue  = overlaps a missing/stuck-meter problem (the data is unusual, not the home)
  Explained   = overlaps a known event such as guests (unusual, but expected)
  Unexplained = needs the homeowner's attention
"""
import pandas as pd

KIND_LABEL = {"spike": "Power spike", "drop": "Sudden drop",
              "sustained_high": "Running high", "sustained_low": "Running low"}
CATS = ["Unexplained", "Explained", "Data issue"]

HARMLESS = {
    "HVAC / air handler": "Heat wave or cold snap, a thermostat change, guests, or a scheduled defrost cycle.",
    "Lighting": "Someone was up late, a motion light, or a holiday/party.",
    "Plug loads": "A new device, charging an EV or laptop, or a one-off appliance use.",
    "Other / misc": "A new appliance, a space heater, or a seasonal change in use.",
}
ACTION = {
    "spike": "Check what was running at that moment. Repeated spikes can point to a failing compressor or loose wiring.",
    "drop": "Check whether a device tripped off or lost power.",
    "sustained_high": "Look for something left on or a device working harder than usual; an inspection may be worthwhile if it continues.",
    "sustained_low": "Check whether a device has stopped working.",
}


def classify_events(events, quality, context, tol="30min"):
    if events.empty:
        return events.assign(category=[], reason=[])
    tol = pd.Timedelta(tol)
    cat, why = [], []
    for _, e in events.iterrows():
        q = quality[(quality.submeter == e.submeter) & (quality.end >= e.start - tol) &
                    (quality.start <= e.end + tol)] if len(quality) else quality
        c = context[(context.end >= e.start - tol) & (context.start <= e.end + tol)] if len(context) else context
        if len(q):
            cat.append("Data issue"); why.append(q.issue.iloc[0])
        elif len(c):
            cat.append("Explained"); why.append(c.label.iloc[0])
        else:
            cat.append("Unexplained"); why.append("")
    return events.assign(category=cat, reason=why)


def _recent(events, now, span):
    return events[events.end >= now - pd.Timedelta(span)] if len(events) else events


def home_status(events, now):
    unx = events[events.category == "Unexplained"] if len(events) else events
    if unx.empty:
        return "healthy"
    r = _recent(unx, now, "48h")
    if len(r) and ((r.peak_z >= 8).any() or (r.duration_h >= 2).any()):
        return "attention"
    return "watch" if len(_recent(unx, now, "7D")) else "healthy"


def kpis(events, quality, now, price):
    ev = _recent(events, now, "7D")
    unx = ev[ev.category == "Unexplained"] if len(ev) else ev
    real = ev[ev.category != "Data issue"] if len(ev) else ev
    extra = float(real.excess_kwh.clip(lower=0).sum()) if len(real) else 0.0
    return dict(
        unexplained=len(unx), explained=int((ev.category == "Explained").sum()) if len(ev) else 0,
        data_issues=int((quality.end >= now - pd.Timedelta("7D")).sum()) if len(quality) else 0,
        excess_kwh=extra, est_cost=extra * price,
        devices=unx.submeter.nunique() if len(unx) else 0)


def appliance_status(scored, events, quality, now):
    rows = []
    for sub, g in scored.groupby("submeter"):
        g = g.sort_values("timestamp"); last = g.dropna(subset=["kw"]).iloc[-1]
        e = _recent(events[events.submeter == sub], now, "48h") if len(events) else events
        q = quality[(quality.submeter == sub) & (quality.end >= now - pd.Timedelta("48h"))] if len(quality) else quality
        if len(e) and (e.category == "Unexplained").any(): s = "Check"
        elif len(q): s = "Data issue"
        elif len(e): s = "Explained"
        else: s = "OK"
        rows.append(dict(submeter=sub, now_kw=last.kw, expected_kw=last.expected, status=s))
    return pd.DataFrame(rows)


def explain_event(e, price):
    return dict(
        title=f"{KIND_LABEL[e.kind]} on {e.submeter}",
        what=f"{e.start:%a %b %d, %I:%M %p} for about {e.duration_h:.2g} h. "
             f"Peak {e.peak_kw:.2f} kW vs about {e.expected_kw:.2f} kW expected.",
        why=f"Reading was {e.peak_z:.1f} robust standard deviations from this device's usual pattern for that time of week.",
        cost=f"About {abs(e.excess_kwh):.2f} kWh {'extra' if e.excess_kwh >= 0 else 'less'} (~${abs(e.excess_kwh) * price:.2f}).",
        harmless=HARMLESS.get(e.submeter, "Normal changes in how the space is used."),
        action=ACTION[e.kind])
