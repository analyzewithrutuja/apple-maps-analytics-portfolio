import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from math import radians, sin, cos, sqrt, asin

ox.settings.log_console = False
ox.settings.use_cache = True

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

# ---------- Load selected trips ----------
trips = pd.read_csv("final_selected_trips.csv")
print(f"Analyzing {len(trips)} real driving trips")

lat_min = min(trips["start_lat"].min(), trips["end_lat"].min()) - 0.02
lat_max = max(trips["start_lat"].max(), trips["end_lat"].max()) + 0.02
lon_min = min(trips["start_lon"].min(), trips["end_lon"].min()) - 0.02
lon_max = max(trips["start_lon"].max(), trips["end_lon"].max()) + 0.02
print(f"Downloading road network for bbox: lat [{lat_min:.3f},{lat_max:.3f}] lon [{lon_min:.3f},{lon_max:.3f}]")

G = ox.graph_from_bbox((lon_min, lat_min, lon_max, lat_max), network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

def route_polyline(G, route):
    """Return list of (lat, lon) points along the route for deviation calc."""
    pts = []
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        edge = min(data.values(), key=lambda d: d.get("length", 1e9))
        if "geometry" in edge:
            xs, ys = edge["geometry"].xy
            pts.extend(zip(ys, xs))  # (lat, lon)
        else:
            pts.append((G.nodes[u]["y"], G.nodes[u]["x"]))
            pts.append((G.nodes[v]["y"], G.nodes[v]["x"]))
    return pts

def point_to_polyline_dist(lat, lon, poly):
    """Min haversine distance (m) from a point to any vertex of the polyline (fast approx)."""
    dists = [haversine(lat, lon, p[0], p[1]) for p in poly]
    return min(dists) if dists else np.nan

results = []
examples = []
for _, row in trips.iterrows():
    f = row["file"]
    try:
        gps = load_plt(f)
    except Exception as e:
        print("skip", f, e)
        continue

    try:
        orig_node = ox.nearest_nodes(G, row["start_lon"], row["start_lat"])
        dest_node = ox.nearest_nodes(G, row["end_lon"], row["end_lat"])
        route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    except Exception as e:
        print("routing failed for", f, e)
        continue

    def route_edge_attr(G, route, attr):
        vals = []
        for u, v in zip(route[:-1], route[1:]):
            data = G.get_edge_data(u, v)
            edge = min(data.values(), key=lambda d: d.get("length", 1e9))
            vals.append(edge.get(attr, 0))
        return vals

    routed_dist_m = sum(route_edge_attr(G, route, "length"))
    routed_time_s = sum(route_edge_attr(G, route, "travel_time"))
    poly = route_polyline(G, route)

    actual_dist_km = row["distance_km"]
    actual_dur_min = row["duration_min"]

    # Deviation: sample every 5th GPS point to keep it fast, measure distance to routed polyline
    sample = gps.iloc[::5]
    deviations = [point_to_polyline_dist(r["lat"], r["lon"], poly) for _, r in sample.iterrows()]
    avg_dev_m = float(np.mean(deviations))
    max_dev_m = float(np.max(deviations))

    detour_ratio = actual_dist_km / (routed_dist_m / 1000) if routed_dist_m > 0 else np.nan
    eta_error_pct = (actual_dur_min*60 - routed_time_s) / routed_time_s * 100 if routed_time_s > 0 else np.nan

    results.append({
        "file": f.split("\\")[-1].split("/")[-1],
        "actual_distance_km": actual_dist_km,
        "routed_distance_km": round(routed_dist_m/1000, 2),
        "detour_ratio": round(detour_ratio, 2),
        "actual_duration_min": round(actual_dur_min, 1),
        "routed_eta_min": round(routed_time_s/60, 1),
        "eta_error_pct": round(eta_error_pct, 1),
        "avg_deviation_m": round(avg_dev_m, 1),
        "max_deviation_m": round(max_dev_m, 1),
    })
    examples.append((f, gps, poly, row))

res_df = pd.DataFrame(results)
res_df["quality_flag"] = np.where((res_df["avg_deviation_m"] > 150) | (res_df["detour_ratio"] > 1.3),
                                   "REVIEW", "OK")
res_df.to_csv("routing_quality_results.csv", index=False)

print("\n--- Routing Quality Summary ---")
print(res_df.to_string())
print(f"\nMean detour ratio: {res_df['detour_ratio'].mean():.2f}")
print(f"Mean |ETA error|: {res_df['eta_error_pct'].abs().mean():.1f}%")
print(f"Mean deviation: {res_df['avg_deviation_m'].mean():.1f} m")
print(f"Flagged for review: {(res_df['quality_flag']=='REVIEW').sum()} / {len(res_df)}")

# ---------- Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(15,4))
axes[0].hist(res_df["detour_ratio"].dropna(), bins=12, color="steelblue")
axes[0].axvline(1.0, color="black", linestyle="--", label="perfect match")
axes[0].set_title("Detour Ratio (actual / routed distance)")
axes[0].legend()

axes[1].hist(res_df["eta_error_pct"].dropna(), bins=12, color="darkorange")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_title("ETA Error (%)")

axes[2].hist(res_df["avg_deviation_m"].dropna(), bins=12, color="seagreen")
axes[2].set_title("Avg. Path Deviation (m)")
plt.tight_layout()
plt.savefig("routing_quality_summary.png", dpi=130)
print("Saved routing_quality_summary.png")

# ---------- Map examples: best match and worst match ----------
res_sorted = res_df.sort_values("avg_deviation_m")
best_idx = res_sorted.index[0]
worst_idx = res_sorted.index[-1]

fig, axes = plt.subplots(1, 2, figsize=(13,6))
for ax, idx, label in [(axes[0], best_idx, "Best Match"), (axes[1], worst_idx, "Worst Match (flagged)")]:
    f, gps, poly, row = examples[idx]
    poly_lat = [p[0] for p in poly]
    poly_lon = [p[1] for p in poly]
    ax.plot(gps["lon"], gps["lat"], color="black", linewidth=2, label="Actual GPS trace")
    ax.plot(poly_lon, poly_lat, color="red", linewidth=1.5, linestyle="--", label="Routed path")
    ax.scatter([row["start_lon"]], [row["start_lat"]], color="green", zorder=5, label="Start")
    ax.scatter([row["end_lon"]], [row["end_lat"]], color="blue", zorder=5, label="End")
    ax.set_title(f"{label}\navg dev={res_df.loc[idx,'avg_deviation_m']:.0f}m, "
                 f"detour={res_df.loc[idx,'detour_ratio']:.2f}x")
    ax.legend(fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("routing_quality_examples.png", dpi=130)
print("Saved routing_quality_examples.png")
