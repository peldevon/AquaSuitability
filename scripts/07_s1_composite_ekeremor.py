import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray  # IMPORTANT: registers the .rio accessor

from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
from odc.stac import load


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = "/config/Desktop/AquacultureProject"

PILOT_FILE = f"{PROJECT_ROOT}/data/pilot_regions.geojson"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"

NAME_FIELD = "adm2nm"
PILOT_NAME = "Ekeremor"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

TARGET_RESOLUTION = 30


# =========================================================
# 1. LOAD PILOT REGION
# =========================================================

print("Loading pilot region...")

pilots = gpd.read_file(PILOT_FILE)

pilot_match = pilots[
    pilots[NAME_FIELD].str.lower() == PILOT_NAME.lower()
]

if len(pilot_match) == 0:
    raise ValueError(
        f"Pilot '{PILOT_NAME}' not found in {PILOT_FILE}"
    )

pilot = pilot_match.iloc[0]

print("Pilot:", PILOT_NAME)


# =========================================================
# 2. DETERMINE LOCAL UTM CRS
# =========================================================

centroid = pilot.geometry.centroid

lon = centroid.x
lat = centroid.y

utm_zone = int((lon + 180) // 6) + 1

epsg = (
    32600 + utm_zone
    if lat >= 0
    else 32700 + utm_zone
)

print("Local UTM:", epsg)


# =========================================================
# 3. CONNECT TO PLANETARY COMPUTER
# =========================================================

print("Connecting to Planetary Computer...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)


# =========================================================
# 4. SEARCH SENTINEL-1 RTC
# =========================================================

print(
    f"Searching Sentinel-1 RTC from "
    f"{START_DATE} to {END_DATE}..."
)

search = catalog.search(
    collections=["sentinel-1-rtc"],
    intersects=mapping(pilot.geometry),
    datetime=f"{START_DATE}/{END_DATE}",
)

items = list(search.items())

print("Scenes found:", len(items))

if len(items) == 0:
    raise RuntimeError(
        "No Sentinel-1 RTC scenes found."
    )


# =========================================================
# 5. LOAD VV AND VH FROM CLOUD
# =========================================================

print("Loading Sentinel-1 data lazily...")

ds = load(
    items,
    bands=["vv", "vh"],
    bbox=pilot.geometry.bounds,
    crs=f"EPSG:{epsg}",
    resolution=TARGET_RESOLUTION,
    chunks={
        "time": 1,
        "x": 1024,
        "y": 1024,
    },
)

print(ds)


# =========================================================
# 6. MASK INVALID / FILL VALUES
# =========================================================

print("Masking invalid values...")

vv_linear = ds.vv.where(
    (ds.vv != -32768)
    & (ds.vv > 0)
)

vh_linear = ds.vh.where(
    (ds.vh != -32768)
    & (ds.vh > 0)
)


# =========================================================
# 7. CONVERT LINEAR POWER TO DECIBELS
# =========================================================

print("Converting linear backscatter to dB...")

vv_db = 10.0 * np.log10(vv_linear)
vh_db = 10.0 * np.log10(vh_linear)


# =========================================================
# 8. BUILD TEMPORAL MEDIAN
# =========================================================

print("Building temporal median composites...")

vv_med_lazy = vv_db.median(
    dim="time",
    skipna=True
)

vh_med_lazy = vh_db.median(
    dim="time",
    skipna=True
)

print("Executing computation...")

# Compute both together
vv_med, vh_med = __import__("dask").compute(
    vv_med_lazy,
    vh_med_lazy
)

print("Median composites completed.")


# =========================================================
# 9. RESTORE SPATIAL METADATA
# =========================================================

for arr in [vv_med, vh_med]:

    arr.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
        inplace=True
    )

    arr.rio.write_crs(
        f"EPSG:{epsg}",
        inplace=True
    )


# =========================================================
# 10. PRINT STATISTICS
# =========================================================

def print_stats(name, arr):

    values = arr.values
    values = values[np.isfinite(values)]

    print(f"\n{name}")
    print("Min    :", np.min(values))
    print("1%     :", np.quantile(values, 0.01))
    print("Median :", np.median(values))
    print("99%    :", np.quantile(values, 0.99))
    print("Max    :", np.max(values))


print_stats(
    "VV median composite",
    vv_med
)

print_stats(
    "VH median composite",
    vh_med
)


# =========================================================
# 11. OUTPUT PATHS
# =========================================================

vv_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_vv_median_2023_2025.tif"
)

vh_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_vh_median_2023_2025.tif"
)

figure_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_s1_median_composite_2023_2025.png"
)


# =========================================================
# 12. SAVE GEOTIFFS
# =========================================================

print("\nSaving VV GeoTIFF...")

vv_med.rio.to_raster(
    vv_output,
    compress="DEFLATE",
    dtype="float32"
)

print("Saving VH GeoTIFF...")

vh_med.rio.to_raster(
    vh_output,
    compress="DEFLATE",
    dtype="float32"
)


# =========================================================
# 13. CREATE DIAGNOSTIC IMAGE
# =========================================================

print("Creating diagnostic figure...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

vv_med.plot(
    ax=axes[0],
    cmap="gray",
    vmin=-25,
    vmax=0
)

axes[0].set_title(
    "Ekeremor Sentinel-1 VV Median\n2023â2025"
)

vh_med.plot(
    ax=axes[1],
    cmap="gray",
    vmin=-30,
    vmax=-5
)

axes[1].set_title(
    "Ekeremor Sentinel-1 VH Median\n2023â2025"
)

for ax in axes:
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

plt.tight_layout()

plt.savefig(
    figure_output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 14. FINAL REPORT
# =========================================================

print("\n======================================")
print("STEP 7 COMPLETE")
print("======================================")

print("VV:")
print(vv_output)

print("\nVH:")
print(vh_output)

print("\nFigure:")
print(figure_output)

print("\nCRS:")
print(f"EPSG:{epsg}")

print("\nResolution:")
print(f"{TARGET_RESOLUTION} m")

print("\nScenes:")
print(len(items))

print("======================================")