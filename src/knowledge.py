"""Static content for the Anomaly guide + coverage tabs, and example charts."""
import numpy as np
import plotly.graph_objects as go

TYPE_NOTES = {
    "global": dict(name="Global (spike)", looks="One reading far outside the normal range.",
                   home="A short power surge when the AC compressor starts.",
                   label="Power spike", gap="None for single spikes."),
    "contextual": dict(name="Contextual", looks="A normal-sized value at the wrong moment.",
                       home="Lights on at 3 AM.",
                       label="Power spike", gap="Labeled as a spike; it is only odd for the time of day."),
    "shapelet": dict(name="Shapelet", looks="The shape of each usage cycle changes.",
                     home="A plugged-in device switching on and off rapidly.",
                     label="Running high / low", gap="Detected but not labeled as a shape change."),
    "seasonal": dict(name="Seasonal (rhythm)", looks="Cycles run faster or slower than usual.",
                     home="AC short-cycling instead of long steady runs.",
                     label="Running high / low", gap="Hardest type (Si et al.); may be missed when the average stays the same."),
    "trend": dict(name="Trend", looks="Usage steadily climbs or falls.",
                  home="An aging appliance slowly drawing more power.",
                  label="Running high", gap="Detected once the shift is large; not labeled as a trend."),
}

# (indicator, source, status, where)
INDICATORS = [
    ("Actual vs expected demand", "Si / Burgos", "Built", "Overview"),
    ("Power surge", "Slide 6", "Built", "Overview, Events"),
    ("Sustained load growth / running high", "Slide 6", "Built", "Overview, Events"),
    ("Appliance status (OK / Check)", "Slide 6", "Built", "Appliances"),
    ("Excess energy (kWh) and cost", "Slide 3", "Built", "Summary cards"),
    ("Explain each alert + possible harmless reasons", "Burgos", "Built", "Events"),
    ("Missing or stuck meter data", "Burgos (suspicious readings)", "Built", "Data quality"),
    ("Event-based precision, recall, latency", "Si", "Built (simulated data)", "Detection errors"),
    ("Zero-shot detection (train on other homes, score an unseen home)", "Si", "Built (simulated data)", "Detection errors"),
    ("False positive / false negative breakdown", "Open question", "Built (simulated data)", "Detection errors"),
    ("Cold start: performance vs days of history", "Si (zero-shot)", "Built (simulated data)", "Detection errors"),
    ("Explained vs unexplained activity", "Open question 1", "Partial: known-event list only", "Events"),
    ("Appliance behaving differently (shape / rhythm)", "Slide 6", "Partial: detected, not labeled", "Events"),
    ("Home deviates from neighbors", "Slide 6", "Partial: chart + callout", "Neighborhood"),
    ("Appliance running nonstop / at odd hours", "Slide 6", "Planned", "-"),
    ("Weather / occupancy context for explanations", "Open question 1", "Planned", "-"),
    ("Peak-demand event counts", "Slide 6", "Planned", "-"),
    ("Real DERConnect data loader", "Project", "Planned: needs data access", "-"),
]


def example_figure(kind):
    x = np.arange(200)
    y = 1 + 0.5 * np.sin(2 * np.pi * x / 25)
    a, b = 100, 140
    if kind == "global":
        y[120] += 3; a, b = 117, 123
    elif kind == "contextual":
        y[119] += 1.0; a, b = 116, 122
    elif kind == "shapelet":
        y[100:126] = 1 + 0.5 * np.sign(np.sin(2 * np.pi * x[100:126] / 25)); b = 126
    elif kind == "seasonal":
        y[100:140] = 1 + 0.5 * np.sin(2 * np.pi * x[100:140] / 12.5)
    elif kind == "trend":
        y[100:] += np.clip(0.03 * (x[100:] - 100), 0, 1.2); b = 200
    fig = go.Figure(go.Scatter(x=x, y=y, line=dict(color="#182B49", width=2)))
    fig.add_vrect(x0=a, x1=b, fillcolor="#E4572E", opacity=0.25, line_width=0)
    fig.update_layout(height=190, margin=dict(l=0, r=0, t=6, b=0), template="plotly_white",
                      xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)")
    return fig
