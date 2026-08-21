import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

grid = pd.read_csv("grid_stats_top2000.csv")
users = pd.read_csv("user_stats.csv")

# The Geolife dataset has a handful of users who traveled internationally, which squishes
# the real signal (Beijing, where ~95%+ of the density lives) into an unreadable dot on a
# world-scale plot. Zoom into the Beijing metro area where the actual activity is.
beijing = grid[(grid["grid_lat"].between(39.7, 40.2)) & (grid["grid_lon"].between(116.0, 116.7))]
print(f"{len(beijing)}/{len(grid)} top grid cells are within the Beijing metro area")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1. Spatial heatmap: point density by grid cell (traffic/activity density)
sc = axes[0].scatter(beijing["grid_lon"], beijing["grid_lat"], c=beijing["point_count"],
                      cmap="inferno", s=40, norm=matplotlib.colors.LogNorm())
axes[0].set_title("GPS Point Density by Grid Cell\n(Beijing metro, top cells, log scale)")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
plt.colorbar(sc, ax=axes[0], label="Point count")

# 2. Spatial heatmap: average speed by grid cell (congestion proxy)
sc2 = axes[1].scatter(beijing["grid_lon"], beijing["grid_lat"], c=beijing["avg_speed_kmh"],
                       cmap="RdYlGn", s=40, vmin=0, vmax=40)
axes[1].set_title("Average Speed by Grid Cell\n(Beijing metro, lower = likely congestion)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(sc2, ax=axes[1], label="Avg speed (km/h)")

# 3. Data volume distribution across users (shows real-world skew)
axes[2].hist(users["total_points"], bins=40, color="steelblue")
axes[2].set_title("GPS Points per User\n(182 users, full dataset)")
axes[2].set_xlabel("Total points recorded")
axes[2].set_ylabel("Number of users")

plt.tight_layout()
plt.savefig("spark_scale_results.png", dpi=130)
print("Saved spark_scale_results.png")
