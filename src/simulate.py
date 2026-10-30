"""Simulated home data with injected anomalies (ground truth known).

Mirrors the assumed DERConnect shape: one row per (timestamp, submeter) with kW.
Anomaly types follow Lai et al. (2021): global, contextual, shapelet, seasonal, trend.
"""
import numpy as np
import pandas as pd

STEP = "15min"
PER_DAY = 96
SUBMETERS = ["HVAC / air handler", "Lighting", "Plug loads", "Other / misc"]


def _base(sub, hours, weekend):
    if sub == "HVAC / air handler":
        y = 0.7 + 0.8 * np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    elif sub == "Lighting":
        y = 0.08 + 0.45 * np.exp(-((hours - 19.5) / 2.2) ** 2) + 0.15 * np.exp(-((hours - 7) / 1.2) ** 2)
    elif sub == "Plug loads":
        y = 0.25 + 0.35 * np.exp(-((hours - 20) / 3) ** 2)
    else:
        y = 0.12 + 0 * hours
    return y * np.where(weekend, 1.15, 1.0)


def simulate_home(name="My home", seed=0, days=14, inject=True, scale=1.0, end=None):
    """Return (df, truth, context). truth/context are empty when inject=False.
    context = known benign events (e.g. guests) the dashboard can use to explain alerts."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().floor("D")
    n = days * PER_DAY
    ts = pd.date_range(end=end, periods=n, freq=STEP)
    hours = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    weekend = ts.weekday.to_numpy() >= 5
    data = {s: scale * _base(s, hours, weekend) + rng.normal(0, 0.04, n) for s in SUBMETERS}
    truth, context = [], []

    def at(days_before_end, hour):
        return n - days_before_end * PER_DAY + int(hour * 4)

    def mark(sub, i, length, kind, note):
        truth.append(dict(submeter=sub, start=ts[i], end=ts[i + length - 1], type=kind, note=note))

    if inject:
        i = at(6, 13); data["HVAC / air handler"][i] += 3.5
        mark("HVAC / air handler", i, 1, "global", "One-sample power surge")
        i = at(5, 3); data["Lighting"][i] += 0.5
        mark("Lighting", i, 1, "contextual", "Lights briefly on at 3 AM")
        i = at(4, 10); L = 12
        data["Plug loads"][i:i + L] = np.where(np.arange(L) // 2 % 2 == 0, 1.2, 0.1)
        mark("Plug loads", i, L, "shapelet", "On/off flapping instead of a smooth cycle")
        i = at(3, 12); L = 16
        data["HVAC / air handler"][i:i + L] = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(L) / 4)
        mark("HVAC / air handler", i, L, "seasonal", "Short-cycling (rhythm change)")
        i = at(2, 0); L = 96
        data["Other / misc"][i:i + L] += np.linspace(0, 1.0, L)
        mark("Other / misc", i, L, "trend", "Load creeping upward for a day")
        # benign, known activity: guests over (looks unusual, is explained)
        i = at(1, 17); L = 16
        for sub, bump in [("Lighting", 0.4), ("Plug loads", 0.5), ("HVAC / air handler", 0.3)]:
            data[sub][i:i + L] += bump
        context.append(dict(start=ts[i], end=ts[i + L - 1], label="Guests over (known event)"))
        # data-quality problems (not home activity)
        i = at(4, 0); data["Lighting"][i:i + 8] = np.nan           # meter stopped reporting
        i = at(1, 2); data["Plug loads"][i:i + 16] = data["Plug loads"][i]  # stuck / flat-lined

    rows = [pd.DataFrame({"timestamp": ts, "building_id": name, "submeter": s,
                          "kw": np.clip(v, 0, None)}) for s, v in data.items()]
    return (pd.concat(rows, ignore_index=True), pd.DataFrame(truth),
            pd.DataFrame(context, columns=["start", "end", "label"]))


def simulate_peers(n=6, days=14, end=None):
    """Daily kWh per peer home (normal behaviour only) for the neighborhood view."""
    out = {}
    for k in range(n):
        d, _, _ = simulate_home(f"Home {k + 1}", seed=100 + k, days=days, inject=False,
                             scale=float(np.random.default_rng(k).uniform(0.8, 1.25)), end=end)
        t = d.groupby(["timestamp"])["kw"].sum() * 0.25
        out[f"Home {k + 1}"] = t.resample("D").sum()
    return pd.DataFrame(out)
