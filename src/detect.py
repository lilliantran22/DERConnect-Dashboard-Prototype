"""Simple, explainable detector: expected = typical value for that time of week
(median), anomaly = large robust z-score of the residual, either at a single
point or averaged over ~2 hours (catches sustained shifts)."""
import numpy as np
import pandas as pd

STEP_H = 0.25


def _robust_scale(x, floor):
    x = x.dropna()
    return max(1.4826 * np.median(np.abs(x - np.median(x))), floor)


def _expected(s):
    key = [s.index.weekday >= 5, s.index.hour * 60 + s.index.minute]
    return s.groupby(key).transform("median")


def score_series(s, sensitivity, model=None):
    """model=None -> per-home baseline (time-of-week median). Otherwise model.expected(s)."""
    exp = _expected(s) if model is None else model.expected(s)
    r = s - exp
    z = r / _robust_scale(r, 0.02)
    rr = r.rolling(8, center=True, min_periods=4).mean()
    z2 = rr / _robust_scale(rr, 0.01)
    flag = (z.abs() > sensitivity) | (z2.abs() > 0.9 * sensitivity)
    return pd.DataFrame({"kw": s, "expected": exp, "resid": r, "z": z, "z_roll": z2, "flag": flag})


def _events(sc, sub, sens, gap):
    idx = np.flatnonzero(sc["flag"].to_numpy())
    out = []
    if idx.size == 0:
        return out
    for grp in np.split(idx, np.where(np.diff(idx) > gap + 1)[0] + 1):
        seg = sc.iloc[grp[0]:grp[-1] + 1]
        pts = seg[seg["z"].abs() > sens]
        up = seg["resid"].sum() >= 0
        if 1 <= len(pts) <= 2:               # isolated point(s): spike / drop
            seg, kind = pts, ("spike" if pts["resid"].sum() >= 0 else "drop")
        else:                                # sustained shift
            kind = "sustained_high" if up else "sustained_low"
        out.append(dict(
            submeter=sub, start=seg.index[0], end=seg.index[-1], duration_h=len(seg) * STEP_H,
            kind=kind, peak_z=float(np.maximum(seg["z"].abs(), seg["z_roll"].abs()).max()),
            excess_kwh=float(seg["resid"].sum() * STEP_H), peak_kw=float(seg["kw"].max()),
            expected_kw=float(seg["expected"].mean())))
    return out


def run_detection(df, sensitivity=4.0, merge_gap=2, model=None):
    """df columns: timestamp, submeter, kw. Returns (scored_df, events_df)."""
    parts, events = [], []
    for sub, g in df.groupby("submeter"):
        s = g.set_index("timestamp")["kw"].sort_index()
        sc = score_series(s, sensitivity, model)
        events += _events(sc, sub, sensitivity, merge_gap)
        parts.append(sc.assign(submeter=sub).reset_index())
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev = ev.sort_values("start").reset_index(drop=True)
        ev.insert(0, "id", range(1, len(ev) + 1))
    return pd.concat(parts, ignore_index=True), ev


def false_alarm_list(events, truth, tol="1h"):
    """Events that overlap no true anomaly."""
    if events.empty:
        return events
    tol = pd.Timedelta(tol)
    keep = []
    for _, e in events.iterrows():
        m = truth[(truth.submeter == e.submeter) & (truth.end + tol >= e.start) & (truth.start - tol <= e.end)] \
            if len(truth) else truth
        keep.append(len(m) == 0)
    return events[keep]


def evaluate(events, truth, tol="1h"):
    """Event-based check (Si et al.): was each true event caught, how late, how many false alarms."""
    tol = pd.Timedelta(tol)
    cols = ["type", "submeter", "note", "detected", "latency_h"]
    rows = []
    for _, t in truth.iterrows():
        m = events[(events.submeter == t.submeter) & (events.end >= t.start - tol) & (events.start <= t.end + tol)] \
            if len(events) else events
        rows.append(dict(type=t.type, submeter=t.submeter, note=t.note, detected=len(m) > 0,
                         latency_h=(m.start.min() - t.start).total_seconds() / 3600 if len(m) else np.nan))
    table = pd.DataFrame(rows, columns=cols)
    fa = len(false_alarm_list(events, truth, tol))
    n_ev = len(events)
    return dict(recall=table.detected.mean() if len(table) else np.nan,
                precision=(n_ev - fa) / n_ev if n_ev else np.nan,
                false_alarms=int(fa), alerts=n_ev), table
