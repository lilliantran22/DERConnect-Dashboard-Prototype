"""Data-quality checks: is it the meter/data that is unusual, not the home?"""
import pandas as pd

MIN_GAP = 2     # missing samples in a row
MIN_FLAT = 8    # identical samples in a row (~2 h at 15 min)
COLS = ["submeter", "start", "end", "issue", "samples"]


def _runs(mask, sub, min_len, label, pad=0):
    grp = (mask != mask.shift()).cumsum()
    out = []
    for _, m in mask.groupby(grp):
        if m.iloc[0] and len(m) >= min_len:
            out.append(dict(submeter=sub, start=m.index[0] - pad * pd.Timedelta("15min"),
                            end=m.index[-1], issue=label, samples=len(m) + pad))
    return out


def _full(g):
    s = g.set_index("timestamp")["kw"].sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="15min"))


def find_quality_issues(df):
    rows = []
    for sub, g in df.groupby("submeter"):
        s = _full(g)
        rows += _runs(s.isna(), sub, MIN_GAP, "Missing data")
        rows += _runs(s.diff().eq(0), sub, MIN_FLAT - 1, "Stuck / flat-lined meter", pad=1)
    return pd.DataFrame(rows, columns=COLS).sort_values("start").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=COLS)


def completeness(df):
    """Share of expected 15-min samples that are present, per submeter."""
    return pd.Series({sub: float(_full(g).notna().mean()) for sub, g in df.groupby("submeter")})
