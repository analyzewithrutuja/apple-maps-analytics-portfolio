import glob
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2*R*asin(sqrt(a))

def load_plt(path):
    df = pd.read_csv(path, skiprows=6, header=None,
                      names=["lat","lon","zero","alt_ft","days","date","time"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df

files = sorted(glob.glob("Geolife Trajectories 1.3/Data/*/Trajectory/*.plt"))
print(f"{len(files)} trajectory files")

results = []
for f in files:
    try:
        df = load_plt(f)
    except Exception:
        continue
    if len(df) < 20:
        continue
    dur_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    if dur_min <= 0 or dur_min > 90:
        continue
    dist = sum(haversine(df["lat"].iloc[i], df["lon"].iloc[i], df["lat"].iloc[i+1], df["lon"].iloc[i+1])
               for i in range(len(df)-1))
    dist_km = dist / 1000
    avg_speed_kmh = dist_km / (dur_min/60) if dur_min > 0 else 0
    straight_line_km = haversine(df["lat"].iloc[0], df["lon"].iloc[0], df["lat"].iloc[-1], df["lon"].iloc[-1]) / 1000

    # Full-path bounding box (not just start/end) -- this is what determines graph coverage needed
    lat_span_km = (df["lat"].max() - df["lat"].min()) * 111
    lon_span_km = (df["lon"].max() - df["lon"].min()) * 111 * cos(radians(df["lat"].mean()))
    path_extent_km = max(lat_span_km, lon_span_km)

    # Detect large mid-trip gaps (sign of GPS dropout / a pause that shouldn't count as one continuous drive)
    max_gap_s = df["timestamp"].diff().dt.total_seconds().max()

    results.append({
        "file": f, "points": len(df), "duration_min": round(dur_min,1),
        "distance_km": round(dist_km,2), "straight_line_km": round(straight_line_km,2),
        "path_extent_km": round(path_extent_km,2),
        "avg_speed_kmh": round(avg_speed_kmh,1), "max_gap_s": max_gap_s,
        "start_lat": df["lat"].iloc[0], "start_lon": df["lon"].iloc[0],
        "end_lat": df["lat"].iloc[-1], "end_lon": df["lon"].iloc[-1],
        "lat_min": df["lat"].min(), "lat_max": df["lat"].max(),
        "lon_min": df["lon"].min(), "lon_max": df["lon"].max(),
    })

summary = pd.DataFrame(results)
# Real, clean point-to-point urban drives: reasonable speed, compact full-path extent, no big gaps
candidates = summary[
    (summary["avg_speed_kmh"].between(12, 70)) &
    (summary["straight_line_km"].between(3, 12)) &
    (summary["path_extent_km"] < 15) &   # entire path (not just endpoints) stays compact
    (summary["max_gap_s"] < 120)          # no long pauses/dropouts mid-trip
]
print(f"{len(candidates)} clean candidates after full-path filtering")

# Keep to a core Beijing area for a tractable shared road network
core = candidates[(candidates["lat_min"] > 39.75) & (candidates["lat_max"] < 40.15) &
                   (candidates["lon_min"] > 116.15) & (candidates["lon_max"] < 116.55)]
print(f"{len(core)} within core Beijing bbox")

sel = core.sort_values("straight_line_km", ascending=False).head(15)
print(sel[["file","distance_km","straight_line_km","path_extent_km","avg_speed_kmh"]].to_string())
sel.to_csv("final_selected_trips.csv", index=False)
