# Home Grid Health: dashboard prototype

Homeowner/tenant view of unusual activity in home electricity data (DSC 180A dashboard subteam).
Built on simulated data first; swap in DERConnect when access is confirmed.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
app.py                 # Streamlit UI (8 tabs)
.streamlit/config.toml # light theme
src/simulate.py        # simulated home: 5 injected anomaly types, a known benign event, 2 data problems, peer homes
src/detect.py          # per-home baseline detector, pluggable model hook, event-based evaluation, false-alarm list
src/zeroshot.py        # zero-shot detector: AR trained on other homes, applied to an unseen home (placeholder model)
src/quality.py         # missing-data / stuck-meter checks, completeness
src/indicators.py      # classify events (Unexplained / Explained / Data issue), status, KPIs, explanations
src/knowledge.py       # anomaly guide content, indicator coverage table, example charts
src/ui.py              # CSS and card helpers
src/data_loader.py     # CSV loader + load_derconnect() stub (the only file to change for real data)
data/                  # local CSVs (git-ignored)
```

## Data contract
`timestamp, submeter, kw` (optional `building_id`), 15-minute means. DERConnect's 2-second data
should be resampled to this in `load_derconnect()`. The schema is an assumption until we see the real data.

## Team questions this prototype addresses
1. Unusual reading vs unusual activity: every alert is bucketed as Data issue / Explained / Unexplained; only Unexplained drives the status banner.
2. Covering all unusual activity: the Coverage & checks tab lists every anomaly type and slide-6 indicator as built, partial, or planned.

## Zero-shot and error analysis
- `src/zeroshot.py` trains an AR model on simulated peer homes and scores a held-out home with no refitting (Si et al.).
  AR is a placeholder; any class with `fit(train_df)` and `expected(series)` can replace it.
- The Detection errors tab shows false positives and negatives, a sensitivity trade-off, results by anomaly type,
  and a cold-start comparison against the per-home baseline. These use simulated ground truth only.
