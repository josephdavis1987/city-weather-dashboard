# streamlit_app.py

import streamlit as st
import datetime
import calendar
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess

st.set_page_config(page_title="City Weather Analysis", layout="wide")

# ── LOESS-cyclic helper ──────────────────────────────────────────────────────
def loess_cyclic(x, y, frac=0.1):
    """
    LOESS smoothing on a cyclic domain.
    - x: 1D array of day-of-year (1…n)
    - y: measurements (same length)
    - frac: LOESS span (fraction of points)
    Returns y_smoothed of length n.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    k = max(int(frac * n), 1)

    x_ext = np.concatenate((x[-k:] - n, x, x[:k] + n))
    y_ext = np.concatenate((y[-k:], y, y[:k]))

    sm_ext = lowess(y_ext, x_ext, frac=frac)
    return sm_ext[k : k + n, 1]


# ── Sidebar: City selection ──────────────────────────────────────────────────
st.sidebar.title("Settings")
all_agg = pd.read_csv("data/precompute_weather.csv")
cities_list = all_agg["city"].unique().tolist()
city = st.sidebar.selectbox("Select a city:", cities_list, index=cities_list.index("Chattanooga"))
my_frac = st.sidebar.slider("LOESS smoothing span (fraction)", min_value=0.01, max_value=0.2, value=0.025, step=0.005)

# ── Load & prepare precomputed data for the selected city ───────────────────
agg = all_agg[all_agg["city"] == city].sort_values("day_of_year").copy()

# Convert temperature columns from °C → °F
for col in ["avg_tmin", "avg_tmax", "min_t", "max_t"]:
    agg[col] = agg[col] * 9/5 + 32
if "avg_t" in agg.columns:
    agg["avg_t"] = agg["avg_t"] * 9/5 + 32

# Re-run LOESS on the now-°F columns
agg["tmin_loess"] = loess_cyclic(agg["day_of_year"], agg["avg_tmin"], frac=my_frac)
agg["tmax_loess"] = loess_cyclic(agg["day_of_year"], agg["avg_tmax"], frac=my_frac)
agg["precip_loess"] = loess_cyclic(agg["day_of_year"], agg["avg_precip"], frac=my_frac)
agg["cum_precip_loess"] = loess_cyclic(agg["day_of_year"], agg["avg_cum_precip"], frac=my_frac)

# ── Fetch live 2025 data from NASA POWER ───────────────────────────────────────
cities_meta = pd.read_csv("data/Southeast_US_Cities.csv")
row = cities_meta.loc[cities_meta["city"] == city].iloc[0]
lat, lon = float(row["lat"]), float(row["lon"])

start_2025 = "20250101"
today = datetime.date.today()
end_2025 = today.strftime("%Y%m%d")

url_2025 = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR"
    f"&community=AG"
    f"&longitude={lon}&latitude={lat}"
    f"&start={start_2025}&end={end_2025}"
    f"&format=JSON"
)
r2025 = requests.get(url_2025)
data2025 = r2025.json()["properties"]["parameter"]

dates_2025 = list(data2025["T2M_MAX"].keys())
df2025 = pd.DataFrame({
    "date": pd.to_datetime(dates_2025, format="%Y%m%d"),
    "tmax": list(data2025["T2M_MAX"].values()),
    "tmin": list(data2025["T2M_MIN"].values()),
    "precip": list(data2025["PRECTOTCORR"].values()),
})
df2025[["tmax", "tmin"]] = df2025[["tmax", "tmin"]] * 9/5 + 32
df2025["day_of_year"] = df2025["date"].dt.dayofyear
df2025["cum_precip"] = df2025["precip"].cumsum()
df2025 = df2025.replace({
    "tmax": {-1766.2: np.nan},
    "tmin": {-1766.2: np.nan},
    "precip": {-999: np.nan}
})
df2025 = df2025.dropna(subset=["tmax", "tmin", "precip"], how="all")

# ── Prepare month-start ticks ─────────────────────────────────────────────────
month_vals = []
month_names = []
for m in range(1, 13):
    dt = datetime.date(2021, m, 1)  # non-leap anchor
    month_vals.append(dt.timetuple().tm_yday)
    month_names.append(calendar.month_abbr[m])

# ── Title ─────────────────────────────────────────────────────────────────────
st.title(f"{city}: Weather Analysis")

# ── 1) Temperature Plot ───────────────────────────────────────────────────────
fig_temp = go.Figure()

# Historic min → max ribbon
fig_temp.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["min_t"],
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_temp.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["max_t"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(72,194,216,0.3)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Historic Record (°F)"
))

# Historic LOESS ribbon (tmin_loess → tmax_loess)
fig_temp.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["tmin_loess"],
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_temp.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["tmax_loess"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(25,50,195,0.2)",
    line=dict(color="rgba(25,116,195,0)"),
    name=f"Historic Avg Range (°F, LOESS frac={my_frac})"
))

# Live 2025 temperature ribbon
fig_temp.add_trace(go.Scatter(
    x=df2025["day_of_year"], y=df2025["tmin"],
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_temp.add_trace(go.Scatter(
    x=df2025["day_of_year"], y=df2025["tmax"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(255,0,0,0.36)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Live 2025 Range (°F)"
))

fig_temp.update_layout(
    title="Temperature by Day of Year (°F)",
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45,
        showgrid=False
    ),
    yaxis_title="Temperature (°F)",
    height=500,
    legend=dict(
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1
    )
)
st.plotly_chart(fig_temp, use_container_width=True)

# ── 2) Cumulative Precipitation Plot ──────────────────────────────────────────
fig_cum = go.Figure()

# Historic full range (min_cum_precip → max_cum_precip)
fig_cum.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["min_cum_precip"],
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_cum.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["max_cum_precip"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(72,194,216,0.2)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Historic Range (min→max)"
))

# ±1 Std ribbon around avg
fig_cum.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["avg_cum_precip"] - agg["std_cum_precip"],
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_cum.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["avg_cum_precip"] + agg["std_cum_precip"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(38,124,186,0.3)",
    line=dict(color="rgba(38,124,186,0)"),
    name="Historic ±1 Std Dev"
))

# Historic average line
fig_cum.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["avg_cum_precip"],
    mode="lines", line=dict(color="rgb(38,124,186)", width=3),
    name="Historic Average"
))

# Live 2025 cumulative
fig_cum.add_trace(go.Scatter(
    x=df2025["day_of_year"], y=df2025["cum_precip"],
    mode="lines", line=dict(color="rgb(195,25,25)", width=3),
    name="Live 2025 Cumulative"
))

fig_cum.update_layout(
    title="Cumulative Precipitation by Day of Year (mm)",
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(title="Cumulative Precipitation (mm)", gridcolor="rgba(200,200,200,0.2)"),
    height=500,
    legend=dict(
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1
    )
)
st.plotly_chart(fig_cum, use_container_width=True)

# ── 3) Daily Precipitation Plot ──────────────────────────────────────────────
fig_daily = go.Figure()

# Historic max_precip ribbon (0 → max_precip)
fig_daily.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["max_precip"],
    mode="lines", line=dict(color="rgba(72,194,216,0)"),
    fill="tozeroy", fillcolor="rgba(72,194,216,0.3)",
    name="Historic Max Precip"
))

# LOESS-smoothed historic average precip
fig_daily.add_trace(go.Scatter(
    x=agg["day_of_year"], y=agg["precip_loess"],
    mode="lines", line=dict(color="rgb(38,124,186)", width=3, dash="dash"),
    name="Historic Avg Precip (LOESS)"
))

# Live 2025 precip ribbon (0 → precip)
fig_daily.add_trace(go.Scatter(
    x=df2025["day_of_year"], y=df2025["precip"],
    mode="lines", line=dict(color="rgba(195,25,25,0)"),
    fill="tozeroy", fillcolor="rgba(195,25,25,0.64)",
    name="Live 2025 Precip"
))

max_all = max(agg["max_precip"].max(), df2025["precip"].max()) * 1.05
fig_daily.update_layout(
    title="Daily Precipitation by Day of Year (mm/day)",
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(title="Precipitation (mm/day)", range=[0, max_all], gridcolor="rgba(200,200,200,0.2)"),
    height=500,
    legend=dict(
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1
    )
)
st.plotly_chart(fig_daily, use_container_width=True)
