"""Data adapters. The dashboard only needs: timestamp, submeter, kw (building_id optional).

TODO (DERConnect): once access is confirmed, implement load_derconnect() so it returns
that same shape, resampled from 2-second data to 15-minute means. Nothing else changes.
"""
import pandas as pd

REQUIRED = {"timestamp", "submeter", "kw"}


def load_csv(file):
    df = pd.read_csv(file, parse_dates=["timestamp"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if "building_id" not in df.columns:
        df["building_id"] = "Uploaded"
    out = (df.set_index("timestamp").groupby(["building_id", "submeter"])["kw"]
             .resample("15min").mean().reset_index())
    return out.dropna(subset=["kw"])


def load_derconnect(*args, **kwargs):
    raise NotImplementedError("Fill in once DERConnect access/schema is known.")
