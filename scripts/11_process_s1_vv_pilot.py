import sys
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import dask

from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
from odc.stac import load
from skimage.filters import threshold_otsu


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = "/config/Desktop/AquacultureProject"

PILOT_FILE = f"{PROJECT_ROOT}/data/pilot_regions.geojson"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"

NAME_FIELD = "adm2nm"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

TARGET_RESOLUTION = 30


# =========================================================
# 1. READ PILOT NAME FROM COMMAND LINE
# =========================================================

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage:\n"
        "python scripts/11_process_s1_vv_pilot.py Borgu\n"
        "or\n"
        "python scripts/11_process_s1_vv_pilot.py Kukawa"
    )

PILOT_NAME = sys.argv[1]

pilot_slug = (
    PILOT_NAME
    .lower()
    .replace(" ", "_")
    .replace("-", "_")
)

print("======================================")
print("SENTINEL-1 VV PILOT PROCESSING")
print("======================================")
print("Pilot:", PILOT_NAME)


# =========================================================
# 2. LOAD PILOT REGION
# =========================================================

pilots = gpd.read_file(PILOT_FILE)

subset = pilots[
    pilots[NAME_FIELD].str.lower()
    == PILOT_NAME.lower()
]

if len(subset) == 0:
    raise ValueError(
        f"{PILOT_NAME} not found in pilot_regions.geojson"
    )

pilot = subset.iloc[0]


# =========================================================
# 3. DETERMINE LOCAL UTM
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
# 4. CONNECT TO PLANETARY COMPUTER
# =========================================================

print("Connecting to Planetary Computer...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace
)


# =========================================================
# 5. SEARCH SENTINEL-1 RTC
# =========================================================

print("Searching Sentinel-1 RTC...")

items = list(
    catalog.search(
        collections=["sentinel-1-rtc"],
        intersects=mapping(pilot.geometry),
        datetime=f"{START_DATE}/{END_DATE}"
    ).items()
)

print("Scenes found:", len(items))

if len(items) == 0:
    raise RuntimeError(
        f"No Sentinel-1 scenes found for {PILOT_NAME}"
    )


# =========================================================
# 6. LOAD VV ONLY
#
# We no longer load VH because Ekeremor testing showed
# VV Otsu gave the strongest cross-sensor agreement.
# This halves the SAR data volume for the remaining pilots.
# =========================================================

print("Loading VV lazily...")

ds = load(
    items,
    bands=["vv"],
    bbox=pilot.geometry.bounds,
    crs=f"EPSG:{epsg}",
    resolution=TARGET_RESOLUTION,
    chunks={
        "time": 1,
        "x": 1024,
        "y": 1024
    }
)

print(ds)


# =========================================================
# 7. MASK INVALID VALUES
# =========================================================

print("Masking invalid/fill values...")

vv_linear = ds.vv.where(
    (ds.vv != -32768)
    & (ds.vv > 0)
)


# =========================================================
# 8. CONVERT LINEAR POWER TO dB
# =========================================================

print("Converting VV to dB...")

vv_db = 10.0 * np.log10(vv_linear)


# =========================================================
# 9. TEMPORAL MEDIAN
# =========================================================

print("Building 2023-2025 VV median composite...")
print("Executing cloud computation...")

vv_med_lazy = vv_db.median(
    dim="time",
    skipna=True
)

vv_med = dask.compute(
    vv_med_lazy
)[0]

print("VV median completed.")


# =========================================================
# 10. SPATIAL METADATA
# =========================================================

vv_med.rio.set_spatial_dims(
    x_dim="x",
    y_dim="y",
    inplace=True
)

vv_med.rio.write_crs(
    f"EPSG:{epsg}",
    inplace=True
)


# =========================================================
# 11. STATISTICS
# =========================================================

values = vv_med.values
values = values[np.isfinite(values)]

print("\nVV median statistics")
print("Min    :", np.min(values))
print("1%     :", np.quantile(values, 0.01))
print("Median :", np.median(values))
print("99%    :", np.quantile(values, 0.99))
print("Max    :", np.max(values))


# =========================================================
# 12. OTSU THRESHOLD
# =========================================================

print("\nCalculating Otsu threshold...")

threshold = threshold_otsu(values)

print("VV Otsu threshold:", threshold)


# =========================================================
# 13. BINARY WATER MASK
#
# SAR open water generally has lower backscatter.
# =========================================================

water = (
    vv_med < threshold
).astype("uint8")

water = water.where(
    np.isfinite(vv_med),
    other=255
)


# =========================================================
# 14. PIXEL COUNTS
# =========================================================

arr = water.values

water_pixels = np.sum(arr == 1)
nonwater_pixels = np.sum(arr == 0)
nodata_pixels = np.sum(arr == 255)

print("\nClassification")
print("Water pixels    :", int(water_pixels))
print("Non-water pixels:", int(nonwater_pixels))
print("NoData pixels   :", int(nodata_pixels))

valid_pixels = (
    water_pixels
    + nonwater_pixels
)

if valid_pixels > 0:

    water_fraction = (
        water_pixels
        / valid_pixels
    )

    print(
        "Mapped water fraction:",
        water_fraction
    )


# =========================================================
# 15. OUTPUT PATHS
# =========================================================

vv_output = (
    f"{OUTPUT_DIR}/"
    f"{pilot_slug}_vv_median_2023_2025.tif"
)

water_output = (
    f"{OUTPUT_DIR}/"
    f"{pilot_slug}_water_vv_otsu.tif"
)

figure_output = (
    f"{OUTPUT_DIR}/"
    f"{pilot_slug}_s1_vv_water.png"
)


# =========================================================
# 16. SAVE RASTERS
# =========================================================

print("\nSaving VV median raster...")

vv_med.rio.to_raster(
    vv_output,
    compress="DEFLATE",
    dtype="float32"
)

print("Saving water mask...")

water.rio.write_nodata(
    255,
    inplace=True
)

water.rio.to_raster(
    water_output,
    compress="DEFLATE",
    dtype="uint8",
    nodata=255
)


# =========================================================
# 17. DIAGNOSTIC FIGURE
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
    f"{PILOT_NAME} Sentinel-1 VV Median\n"
    f"2023â2025"
)

water.plot(
    ax=axes[1],
    add_colorbar=False
)

axes[1].set_title(
    f"VV Otsu Water Mask\n"
    f"Threshold = {threshold:.3f} dB"
)

for ax in axes:

    ax.set_xlabel(
        "Easting (m)"
    )

    ax.set_ylabel(
        "Northing (m)"
    )

plt.tight_layout()

plt.savefig(
    figure_output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 18. FINAL REPORT
# =========================================================

print("\n======================================")
print("STEP 11 COMPLETE")
print("======================================")

print("Pilot:")
print(PILOT_NAME)

print("\nScenes:")
print(len(items))

print("\nOtsu threshold:")
print(threshold)

print("\nVV raster:")
print(vv_output)

print("\nWater mask:")
print(water_output)

print("\nFigure:")
print(figure_output)

print("======================================")