"""
Run this once from inside the 'archive' folder:
    python generate_stations.py

It reads the benchmark data you already have and builds stations.json,
a small file listing each known monitoring location with its most recent
historical and regional stats. The frontend map reads this file directly.
"""

import pandas as pd
import json

# Combine all splits so we get every station that ever appeared
train = pd.read_csv("early_warning_train.csv")
val = pd.read_csv("early_warning_validation.csv")
test = pd.read_csv("early_warning_test.csv")
all_rows = pd.concat([train, val, test], ignore_index=True)

all_rows["date"] = pd.to_datetime(all_rows["date"])

# For each location, keep only the most recent row -- that's the latest
# known history/regional snapshot for that station
latest_per_station = (
    all_rows.sort_values("date")
    .groupby("location_id", as_index=False)
    .tail(1)
)

stations = []
for _, row in latest_per_station.iterrows():
    stations.append({
        "location_id": row["location_id"],
        "state": row.get("state", ""),
        "location_description": row.get("location_description", ""),
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "as_of_date": row["date"].strftime("%Y-%m-%d"),
        "hist_7d_max_cell_count": row["hist_7d_max_cell_count"],
        "hist_7d_bloom_fraction": row["hist_7d_bloom_fraction"],
        "hist_30d_max_cell_count": row["hist_30d_max_cell_count"],
        "hist_30d_bloom_fraction": row["hist_30d_bloom_fraction"],
        "hist_90d_max_cell_count": row["hist_90d_max_cell_count"],
        "hist_90d_bloom_fraction": row["hist_90d_bloom_fraction"],
        "regional_7d_max_cell_count": row["regional_7d_max_cell_count"],
        "regional_30d_max_cell_count": row["regional_30d_max_cell_count"],
        "regional_90d_max_cell_count": row["regional_90d_max_cell_count"],
    })

# Drop any station missing coordinates
stations = [s for s in stations if pd.notna(s["latitude"]) and pd.notna(s["longitude"])]

import math

def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

stations_clean = [clean_nans(s) for s in stations]

with open("stations.json", "w") as f:
    json.dump(stations_clean, f, indent=2, default=str)

print(f"Wrote {len(stations)} stations to stations.json")