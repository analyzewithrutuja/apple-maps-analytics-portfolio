import os
import time
import glob

# Requires JAVA_HOME set to a JDK 17+ install (PySpark runs on the JVM).
# On Windows, also requires HADOOP_HOME pointing to a directory containing bin/winutils.exe +
# hadoop.dll (e.g. from https://github.com/cdarlint/winutils) -- Spark's file-listing code calls
# into native Hadoop I/O even in pure local mode. Both must be set before the SparkSession starts.
if "JAVA_HOME" not in os.environ:
    raise EnvironmentError("Set JAVA_HOME to a JDK 17+ install before running this script.")

from pyspark.sql import SparkSession, Window, Row
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

spark = (
    SparkSession.builder.appName("GeolifeScaleAnalysis")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ---------- Discover all trajectory files across all 182 users ----------
files = glob.glob("Geolife Trajectories 1.3/Data/*/Trajectory/*.plt")
print(f"Discovered {len(files)} trajectory files across all users")

# ---------- Read all raw GPS points, correctly skipping each file's 6-line header ----------
# Each .plt file has 6 metadata header lines before the actual GPS data (format, altitude
# units, a track-definition line, etc). A plain spark.read.csv() has no per-file header-skip
# option, so those header lines get parsed as if they were data -- corrupting the date/time
# columns with garbage like "0 2" and crashing the timestamp cast downstream. Fixed by reading
# whole files (path, content) and explicitly dropping each file's first 6 lines before parsing.
def parse_file(path_content):
    path, content = path_content
    user_id = path.split(os.sep)[-3] if os.sep in path else path.split("/")[-3]
    traj_id = path.split(os.sep)[-1] if os.sep in path else path.split("/")[-1]
    rows = []
    for line in content.splitlines()[6:]:
        parts = line.strip().split(",")
        if len(parts) != 7:
            continue
        try:
            lat, lon, _zero, alt_ft, _days = (float(parts[i]) for i in range(5))
        except ValueError:
            continue
        rows.append(Row(lat=lat, lon=lon, alt_ft=alt_ft, date=parts[5], time=parts[6],
                         user_id=user_id, traj_id=traj_id))
    return rows

file_rdd = spark.sparkContext.wholeTextFiles(
    "Geolife Trajectories 1.3/Data/*/Trajectory/*.plt", minPartitions=64
)
# NOTE: .toDF() without an explicit schema calls .first() internally to infer types, and on this
# machine .first()/.take() on a wholeTextFiles-derived RDD reliably crashes the Python worker
# (Windows-specific socket issue transferring larger buffered payloads back to the driver).
# Passing an explicit schema to createDataFrame() skips that internal .first() call entirely.
raw_schema = StructType([
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("alt_ft", DoubleType(), True),
    StructField("date", StringType(), True),
    StructField("time", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("traj_id", StringType(), True),
])
raw = spark.createDataFrame(file_rdd.flatMap(parse_file), schema=raw_schema)

raw = raw.withColumn("timestamp", F.to_timestamp(F.concat_ws(" ", F.col("date"), F.col("time"))))
raw = raw.filter(F.col("timestamp").isNotNull())

n_points = raw.count()
print(f"Total raw GPS points across full dataset: {n_points:,}")
n_users = raw.select("user_id").distinct().count()
print(f"Distinct users: {n_users}")

# ---------- Distributed window computation: speed & distance between consecutive points ----------
# Partitioned per (user, trajectory file) so consecutive-point deltas never cross between trips.
w = Window.partitionBy("user_id", "traj_id").orderBy("timestamp")

pts = raw.withColumn("prev_lat", F.lag("lat").over(w)) \
          .withColumn("prev_lon", F.lag("lon").over(w)) \
          .withColumn("prev_ts", F.lag("timestamp").over(w))

# Haversine distance (meters) between consecutive points, vectorized across the cluster
R = 6371000.0
pts = pts.withColumn("dlat", F.radians(F.col("lat") - F.col("prev_lat"))) \
          .withColumn("dlon", F.radians(F.col("lon") - F.col("prev_lon"))) \
          .withColumn(
              "a",
              F.sin(F.col("dlat")/2)**2 +
              F.cos(F.radians(F.col("prev_lat"))) * F.cos(F.radians(F.col("lat"))) *
              F.sin(F.col("dlon")/2)**2
          ) \
          .withColumn("dist_m", 2 * R * F.asin(F.sqrt(F.col("a")))) \
          .withColumn("dt_s", F.col("timestamp").cast("long") - F.col("prev_ts").cast("long")) \
          .withColumn("speed_kmh", F.when(F.col("dt_s") > 0, F.col("dist_m") / F.col("dt_s") * 3.6))

pts = pts.filter(F.col("speed_kmh").isNotNull() & (F.col("speed_kmh") < 200))  # drop GPS glitches

# ---------- Spatial aggregation: grid-cell speed/traffic density across the whole dataset ----------
# 0.01-degree grid (~1.1km cells) -- classic large-scale mapping/traffic aggregation pattern
grid = pts.withColumn("grid_lat", F.round(F.col("lat") / 0.01) * 0.01) \
          .withColumn("grid_lon", F.round(F.col("lon") / 0.01) * 0.01)

grid_stats = grid.groupBy("grid_lat", "grid_lon").agg(
    F.count("*").alias("point_count"),
    F.avg("speed_kmh").alias("avg_speed_kmh"),
    F.stddev("speed_kmh").alias("std_speed_kmh"),
    F.countDistinct("user_id").alias("distinct_users"),
).orderBy(F.desc("point_count"))

grid_stats.cache()
n_cells = grid_stats.count()
print(f"\nSpatial grid cells with GPS activity: {n_cells:,}")

print("\nTop 15 busiest grid cells (by point density):")
grid_stats.show(15, truncate=False)

# ---------- Per-user summary ----------
user_stats = pts.groupBy("user_id").agg(
    F.count("*").alias("total_points"),
    F.countDistinct("traj_id").alias("n_trajectories"),
    F.avg("speed_kmh").alias("avg_speed_kmh"),
    (F.sum("dist_m")/1000).alias("total_distance_km"),
).orderBy(F.desc("total_points"))

print("\nTop 10 users by data volume:")
user_stats.show(10, truncate=False)

elapsed = time.time() - t0
print(f"\nTotal processing time: {elapsed:.1f}s for {n_points:,} raw points -> "
      f"{n_cells:,} spatial cells across {n_users} users")

# Export results for writeup/plotting
grid_stats.orderBy(F.desc("point_count")).limit(2000).toPandas().to_csv("grid_stats_top2000.csv", index=False)
user_stats.toPandas().to_csv("user_stats.csv", index=False)

with open("run_summary.txt", "w") as f:
    f.write(f"total_raw_points={n_points}\n")
    f.write(f"distinct_users={n_users}\n")
    f.write(f"spatial_grid_cells={n_cells}\n")
    f.write(f"processing_time_seconds={elapsed:.1f}\n")

spark.stop()
print("Done.")
