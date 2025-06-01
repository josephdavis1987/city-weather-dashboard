# streamlit_app.py

import streamlit as st
import datetime, calendar
import requests
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS AND UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_city_list():
    """
    Reads Southeast_US_Cities.csv and returns a DataFrame with columns ['city','lat','lon'].
    """
    path = Path(__file__).parent / "data" / "Southeast_US_Cities.csv"
    df = pd.read_csv(path)
    return df

@st.cache_data
def fetch_historic(city):
    """
    Queries DuckDB for all daily_weather rows for the given city,
    converts temps to °F, adds day_of_year, and returns a DataFrame.
    """
    con = duckdb.connect(str(Path(__file__).parent / "data" / "weather.duckdb"))
    query = f"""
        SELECT *
        FROM daily_weather
        WHERE city = '{city}'
    """
    df = con.execute(query).fetchdf()
    con.close()

    # Convert temps to °F
    df[['t','tmax','tmin']] = df[['t','tmax','tmin']] * 9/5 + 32

    # Compute day_of_year
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear

    return df

def loess_cyclic(x, y, frac=0.1):
    """
    LOESS smoothing on a cyclic domain (1…n). Returns smoothed y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    k = max(int(frac * n), 1)

    x_ext = np.concatenate((x[-k:] - n, x, x[:k] + n))
    y_ext = np.concatenate((y[-k:], y, y[:k]))

    sm_ext = lowess(y_ext, x_ext, frac=frac)
    return sm_ext[k:k+n, 1]

@st.cache_data
def fetch_live_2025(city):
    """
    Hits NASA POWER for T2M_MAX, T2M_MIN, PRECTOTCORR from 2025-01-01 to today,
    returns a DataFrame with columns ['date','tmax','tmin','precip','day_of_year','cum_precip'].
    """
    cities_df = load_city_list()
    row = cities_df.loc[cities_df["city"] == city].iloc[0]
    lat, lon = float(row["lat"]), float(row["lon"])

    start_2025 = "20250101"
    today = datetime.date.today()
    end_2025 = today.strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_2025}&end={end_2025}"
        f"&format=JSON"
    )
    r = requests.get(url)
    data = r.json()["properties"]["parameter"]

    dates = list(data["T2M_MAX"].keys())
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "tmax": list(data["T2M_MAX"].values()),
        "tmin": list(data["T2M_MIN"].values()),
        "precip": list(data["PRECTOTCORR"].values()),
    })
    # Convert temps to °F
    df[["tmax","tmin"]] = df[["tmax","tmin"]] * 9/5 + 32

    # day_of_year + cumulative precip
    df["day_of_year"] = df["date"].dt.dayofyear
    df = df.sort_values("day_of_year").copy()
    df["cum_precip"] = df["precip"].fillna(0).cumsum()

    # Replace missing‐data sentinels
    df = df.replace({
        "tmax": {-1766.2: np.nan},
        "tmin": {-1766.2: np.nan},
        "precip": {-999: np.nan}
    })
    df = df.dropna(subset=["tmax","tmin","precip"], how="all")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 2) STREAMLIT APP LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="City Weather Dashboard", layout="wide")
st.title("🌤️ City Weather Analysis")

# Sidebar city selector
cities_df = load_city_list()
city_names = cities_df["city"].tolist()
selected_city = st.sidebar.selectbox("Choose a city:", city_names, index=city_names.index("Chattanooga"))

st.sidebar.markdown("---")
st.sidebar.markdown("Data from NASA POWER (daily T2M/T2M_MAX/T2M_MIN/ precipitation)")

# Fetch historic + live 2025
city_df = fetch_historic(selected_city)
df2025  = fetch_live_2025(selected_city)

# Compute monthly tick positions once
month_vals = []
month_names = []
for m in range(1,13):
    dt = datetime.date(2021, m, 1)  # non-leap year
    doy = dt.timetuple().tm_yday
    month_vals.append(doy)
    month_names.append(calendar.month_abbr[m])

# ──────────────────────────────────────────────────────────────────────────────
# 3) PREPARE AGGREGATED HISTORIC DATA (LOESS, STD) FOR PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

# day_of_year already in city_df
agg = (
    city_df
      .groupby("day_of_year")
      .agg(
        avg_t=("t","mean"),
        avg_tmin=("tmin","mean"),
        avg_tmax=("tmax","mean"),
        min_t=("tmin","min"),
        max_t=("tmax","max"),
        avg_precip=("precip","mean"),
        min_precip=("precip","min"),
        max_precip=("precip","max"),
        avg_cum_precip=("cum_precip","mean"),
        std_cum_precip=("cum_precip","std"),
        min_cum_precip=("cum_precip","min"),
        max_cum_precip=("cum_precip","max"),
      )
      .reset_index()
)

# LOESS smoothing
my_frac = 0.025
agg["tmin_loess"]       = loess_cyclic(agg["day_of_year"], agg["avg_tmin"],  frac=my_frac)
agg["tmax_loess"]       = loess_cyclic(agg["day_of_year"], agg["avg_tmax"],  frac=my_frac)
agg["precip_loess"]     = loess_cyclic(agg["day_of_year"], agg["avg_precip"], frac=my_frac)
agg["cum_precip_loess"] = loess_cyclic(agg["day_of_year"], agg["avg_cum_precip"], frac=my_frac)

# ──────────────────────────────────────────────────────────────────────────────
# 4) TEMPERATURE PLOT
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🌡️ Temperature: Historic LOESS vs Live 2025 Range")

fig_temp = go.Figure()

# Historic LOESS ribbon (avg_tmin → avg_tmax)
fig_temp.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg["avg_tmin"], 
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_temp.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg["avg_tmax"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(72,194,216,0.3)", line=dict(color="rgba(0,0,0,0)"),
    name="Historic Avg Temp Range (LOESS)"
))

# Live 2025 range ribbon (tmin → tmax)
fig_temp.add_trace(go.Scatter(
    x=df2025.day_of_year, y=df2025.tmin,
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_temp.add_trace(go.Scatter(
    x=df2025.day_of_year, y=df2025.tmax,
    mode="lines", fill="tonexty",
    fillcolor="rgba(255,0,0,0.36)", line=dict(color="rgba(0,0,0,0)"),
    name="Live 2025 Temp Range"
))

fig_temp.update_layout(
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45
    ),
    yaxis_title="Temperature (°F)",
    legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    margin=dict(t=50, b=50)
)

st.plotly_chart(fig_temp, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) CUMULATIVE PRECIPITATION PLOT
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📈 Cumulative Precipitation: Historic Range + ±1 Std + Live 2025")

fig_cum = go.Figure()

# Historic range (min_cum_precip → max_cum_precip)
fig_cum.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg.min_cum_precip,
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_cum.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg.max_cum_precip,
    mode="lines", fill="tonexty",
    fillcolor="rgba(72,194,216,0.2)", line=dict(color="rgba(0,0,0,0)"),
    name="Historic Range (min→max)"
))

# ±1 Std ribbon around avg_cum_precip
fig_cum.add_trace(go.Scatter(
    x=agg.day_of_year,
    y=agg.avg_cum_precip - agg.std_cum_precip,
    mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False
))
fig_cum.add_trace(go.Scatter(
    x=agg.day_of_year,
    y=agg.avg_cum_precip + agg.std_cum_precip,
    mode="lines", fill="tonexty",
    fillcolor="rgba(38,124,186,0.3)", line=dict(color="rgba(38,124,186,0)"),
    name="Historic ±1 Std Dev"
))

# Historic average line
fig_cum.add_trace(go.Scatter(
    x=agg.day_of_year,
    y=agg.avg_cum_precip,
    mode="lines",
    line=dict(color="rgb(38,124,186)", width=3),
    name="Historic Average"
))

# Live 2025 cumulative
fig_cum.add_trace(go.Scatter(
    x=df2025.day_of_year,
    y=df2025.cum_precip,
    mode="lines",
    line=dict(color="rgb(195,25,25)", width=3),
    name="Live 2025 Cumulative"
))

fig_cum.update_layout(
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(title="Cumulative Precipitation (mm)", gridcolor="rgba(200,200,200,0.2)"),
    margin=dict(t=50, b=50)
)

st.plotly_chart(fig_cum, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6) DAILY PRECIPITATION PLOT (MOVED TO BOTTOM)
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("☔ Daily Precipitation: Historic Max Ribbon + Live 2025 + LOESS")

fig_daily = go.Figure()

# Historic max_precip ribbon
fig_daily.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg.max_precip,
    mode="lines", line=dict(color="rgba(72,194,216,0)"),
    fill="tozeroy", fillcolor="rgba(72,194,216,0.3)",
    name="Historic Max Precip"
))

# Live 2025 precip ribbon
fig_daily.add_trace(go.Scatter(
    x=df2025.day_of_year, y=df2025.precip,
    mode="lines", line=dict(color="rgba(195,25,25,0)"),
    fill="tozeroy", fillcolor="rgba(195,25,25,0.64)",
    name="Live 2025 Precip"
))

# LOESS historic average precip
fig_daily.add_trace(go.Scatter(
    x=agg.day_of_year, y=agg.precip_loess,
    mode="lines", line=dict(color="rgb(38,124,186)", width=3, dash="dash"),
    name="Historic Avg Precip (LOESS)"
))

fig_daily.update_layout(
    xaxis=dict(
        title="Day of Year",
        tickmode="array",
        tickvals=month_vals,
        ticktext=month_names,
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(title="Precipitation (mm/day)", gridcolor="rgba(200,200,200,0.2)"),
    margin=dict(t=50, b=50)
)

st.plotly_chart(fig_daily, use_container_width=True)
