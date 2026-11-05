import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray

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
PILOT_NAME = "Ekeremor"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# B11 is 20 m, so use 20 m for MNDWI
TARGET_RESOLUTION = 20


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
        f"Pilot '{PILOT_NAME}' not found."
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
# 4. SEARCH SENTINEL-2 L2A
# =========================================================

print(
    f"Searching Sentinel-2 L2A from "
    f"{START_DATE} to {END_DATE}..."
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=mapping(pilot.geometry),
    datetime=f"{START_DATE}/{END_DATE}",
    query={
        "eo:cloud_cover": {
            "lt": 60
        }
    }
)

items = list(search.items())

print("Scenes found:", len(items))

if len(items) == 0:
    raise RuntimeError(
        "No Sentinel-2 scenes found."
    )

print(
    "Available assets:",
    list(items[0].assets.keys())
)


# =========================================================
# 5. LOAD REQUIRED BANDS
# =========================================================

print("Loading B03, B11 and SCL lazily...")

ds = load(
    items,
    bands=[
        "B03",
        "B11",
        "SCL"
    ],
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
# 6. MASK CLOUDS / SHADOWS / INVALID PIXELS
# =========================================================

print("Applying Sentinel-2 SCL mask...")

# SCL classes to reject:
# 0  = no data
# 1  = saturated / defective
# 3  = cloud shadow
# 8  = cloud medium probability
# 9  = cloud high probability
# 10 = thin cirrus
# 11 = snow / ice

bad_classes = [
    0,
    1,
    3,
    8,
    9,
    10,
    11
]

valid = ~ds.SCL.isin(bad_classes)

green = ds.B03.where(valid)
swir = ds.B11.where(valid)


# =========================================================
# 7. CALCULATE MNDWI
# =========================================================

print("Calculating MNDWI...")

denominator = green + swir

mndwi = (green - swir) / denominator

# remove invalid denominator cases
mndwi = mndwi.where(
    np.isfinite(mndwi)
)


# =========================================================
# 8. TEMPORAL MEDIAN
# =========================================================

print("Building temporal median MNDWI composite...")
print("This may take some time because scenes are cloud-hosted.")

mndwi_med = mndwi.median(
    dim="time",
    skipna=True
)

print("Executing computation...")

mndwi_med = mndwi_med.compute()

print("MNDWI median composite completed.")


# =========================================================
# 9. RESTORE SPATIAL METADATA
# =========================================================

mndwi_med.rio.set_spatial_dims(
    x_dim="x",
    y_dim="y",
    inplace=True
)

mndwi_med.rio.write_crs(
    f"EPSG:{epsg}",
    inplace=True
)


# =========================================================
# 10. BASIC MNDWI STATISTICS
# =========================================================

values = mndwi_med.values
values = values[np.isfinite(values)]

print("\nMNDWI statistics")
print("Min    :", np.min(values))
print("1%     :", np.quantile(values, 0.01))
print("Median :", np.median(values))
print("99%    :", np.quantile(values, 0.99))
print("Max    :", np.max(values))


# =========================================================
# 11. OTSU WATER THRESHOLD
# =========================================================

print("\nCalculating Otsu threshold...")

mndwi_threshold = threshold_otsu(values)

print(
    "MNDWI Otsu threshold:",
    mndwi_threshold
)

# For optical MNDWI:
# higher values = more likely water
water = (
    mndwi_med > mndwi_threshold
).astype("uint8")

water = water.where(
    np.isfinite(mndwi_med),
    other=255
)


# =========================================================
# 12. COUNT PIXELS
# =========================================================

arr = water.values

water_pixels = np.sum(arr == 1)
nonwater_pixels = np.sum(arr == 0)
nodata_pixels = np.sum(arr == 255)

print("\nSentinel-2 MNDWI mask")
print(
    "Water pixels    :",
    int(water_pixels)
)
print(
    "Non-water pixels:",
    int(nonwater_pixels)
)
print(
    "NoData pixels   :",
    int(nodata_pixels)
)


# =========================================================
# 13. OUTPUT PATHS
# =========================================================

mndwi_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_mndwi_median_2023_2025.tif"
)

water_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_water_mndwi_otsu.tif"
)

figure_output = (
    f"{OUTPUT_DIR}/"
    "ekeremor_s2_mndwi_2023_2025.png"
)


# =========================================================
# 14. SAVE RASTERS
# =========================================================

print("\nSaving MNDWI raster...")

mndwi_med.rio.to_raster(
    mndwi_output,
    compress="DEFLATE",
    dtype="float32"
)

print("Saving MNDWI water mask...")

water.rio.to_raster(
    water_output,
    compress="DEFLATE",
    dtype="uint8",
    nodata=255
)


# =========================================================
# 15. CREATE DIAGNOSTIC FIGURE
# =========================================================

print("Creating diagnostic figure...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

mndwi_med.plot(
    ax=axes[0],
    cmap="RdYlBu",
    vmin=-0.5,
    vmax=0.7
)

axes[0].set_title(
    f"Ekeremor MNDWI Median\n"
    f"2023â2025 | Otsu={mndwi_threshold:.3f}"
)

water.plot(
    ax=axes[1],
    add_colorbar=False
)

axes[1].set_title(
    "Sentinel-2 MNDWI Water Mask"
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
# 16. FINAL REPORT
# =========================================================

print("\n======================================")
print("STEP 9 COMPLETE")
print("======================================")

print("MNDWI threshold:")
print(mndwi_threshold)

print("\nMNDWI raster:")
print(mndwi_output)

print("\nWater mask:")
print(water_output)

print("\nDiagnostic figure:")
print(figure_output)

print("\nScenes used:")
print(len(items))

print("\nCRS:")
print(f"EPSG:{epsg}")

print("\nResolution:")
print(f"{TARGET_RESOLUTION} m")

print("======================================")