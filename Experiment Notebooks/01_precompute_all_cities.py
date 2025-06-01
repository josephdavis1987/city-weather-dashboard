
# precompute_all_cities

import os
import duckdb
import pandas as pd
from pathlib import Path

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Path to your DuckDB file (already populated with daily_weather)
DB_PATH = os.path.join("data", "weather.duckdb")

# Path to the CSV listing all Southeast US cities and their lat/lon
CITIES_CSV_PATH = os.path.join("data", "Southeast_US_Cities.csv")

# Where to write the final, combined precomputed CSV:
OUT_CSV_PATH = os.path.join("data", "precompute_weather.csv")


# ── HELPER FUNCTION ────────────────────────────────────────────────────────────

def aggregate_for_city(con: duckdb.DuckDBPyConnection, city_name: str) -> pd.DataFrame:
    """
    Run a DuckDB query that:
     - Filters daily_weather for the given city_name,
     - Computes day_of_year = CAST(strftime('%j', date) AS INTEGER),
     - Aggregates mean/min/max/stddev for T, TMIN, TMAX, precip, cum_precip,
     - Returns a DataFrame with a new column "city" prepended.
    """
    sql = f"""
    SELECT
      '{city_name}' AS city,
      CAST(strftime('%j', date) AS INTEGER) AS day_of_year,

      -- temperature aggregates
      AVG(t)    AS avg_t,
      AVG(tmin) AS avg_tmin,
      AVG(tmax) AS avg_tmax,
      MIN(tmin) AS min_t,
      MAX(tmax) AS max_t,

      -- precipitation aggregates (daily)
      AVG(precip)    AS avg_precip,
      MIN(precip)    AS min_precip,
      MAX(precip)    AS max_precip,

      -- cumulative precipitation aggregates
      AVG(cum_precip)        AS avg_cum_precip,
      STDDEV_POP(cum_precip) AS std_cum_precip,
      MIN(cum_precip)        AS min_cum_precip,
      MAX(cum_precip)        AS max_cum_precip

    FROM daily_weather
    WHERE city = '{city_name}'
    GROUP BY day_of_year
    ORDER BY day_of_year
    """
    return con.execute(sql).df()


# ── MAIN SCRIPT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Read the list of cities:
    cities_df = pd.read_csv(CITIES_CSV_PATH)
    city_names = cities_df["city"].tolist()

    # 2) Open DuckDB connection once:
    con = duckdb.connect(DB_PATH)

    # 3) For each city, run the aggregation and collect results:
    agg_frames = []
    for city_name in city_names:
        print(f"→ Aggregating for city: {city_name}")
        city_agg = aggregate_for_city(con, city_name)
        agg_frames.append(city_agg)

    con.close()

    # 4) Concatenate all city‐level aggregates into one big DataFrame:
    all_cities_agg = pd.concat(agg_frames, axis=0, ignore_index=True)

    # 5) Write to CSV (no index):
    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
    all_cities_agg.to_csv(OUT_CSV_PATH, index=False)

    print(f"\n✅ Done! Wrote {len(all_cities_agg)} rows to '{OUT_CSV_PATH}'.")
