import numpy as np
import matplotlib.pyplot as plt
import rioxarray
from skimage.filters import threshold_otsu


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = "/config/Desktop/AquacultureProject"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"

VV_FILE = f"{OUTPUT_DIR}/ekeremor_vv_median_2023_2025.tif"
VH_FILE = f"{OUTPUT_DIR}/ekeremor_vh_median_2023_2025.tif"

VV_MASK_OUT = f"{OUTPUT_DIR}/ekeremor_water_vv_otsu.tif"
VH_MASK_OUT = f"{OUTPUT_DIR}/ekeremor_water_vh_otsu.tif"
COMBINED_MASK_OUT = f"{OUTPUT_DIR}/ekeremor_water_combined_otsu.tif"

COMBINED_RASTER_OUT = f"{OUTPUT_DIR}/ekeremor_vv_plus_vh_median_2023_2025.tif"

FIGURE_OUT = f"{OUTPUT_DIR}/ekeremor_sar_classifier_comparison.png"


# =========================================================
# 1. LOAD VV AND VH COMPOSITES
# =========================================================

print("Loading VV and VH median rasters...")

vv = rioxarray.open_rasterio(VV_FILE, masked=True).squeeze()
vh = rioxarray.open_rasterio(VH_FILE, masked=True).squeeze()

print("VV shape:", vv.shape)
print("VH shape:", vh.shape)
print("CRS:", vv.rio.crs)
print("Resolution:", vv.rio.resolution())


# =========================================================
# 2. BUILD COMBINED RASTER
# =========================================================

print("Building combined raster (VV + VH)...")

combined = vv + vh


# =========================================================
# 3. EXTRACT VALID VALUES FOR OTSU
# =========================================================

def get_valid_values(arr):
    values = arr.values
    values = values[np.isfinite(values)]
    return values

vv_vals = get_valid_values(vv)
vh_vals = get_valid_values(vh)
combined_vals = get_valid_values(combined)

if len(vv_vals) == 0:
    raise RuntimeError("VV raster has no valid values.")

if len(vh_vals) == 0:
    raise RuntimeError("VH raster has no valid values.")

if len(combined_vals) == 0:
    raise RuntimeError("Combined raster has no valid values.")


# =========================================================
# 4. CALCULATE OTSU THRESHOLDS
# =========================================================

print("Calculating Otsu thresholds...")

vv_thresh = threshold_otsu(vv_vals)
vh_thresh = threshold_otsu(vh_vals)
combined_thresh = threshold_otsu(combined_vals)

print("\nOtsu thresholds")
print("VV threshold       :", vv_thresh)
print("VH threshold       :", vh_thresh)
print("Combined threshold :", combined_thresh)


# =========================================================
# 5. CREATE WATER MASKS
#    Rule: lower backscatter = more likely water
# =========================================================

print("Generating binary water masks...")

vv_water = (vv < vv_thresh).astype("uint8")
vh_water = (vh < vh_thresh).astype("uint8")
combined_water = (combined < combined_thresh).astype("uint8")

# Apply valid-data masks so nodata does not become false land
vv_water = vv_water.where(np.isfinite(vv), other=255)
vh_water = vh_water.where(np.isfinite(vh), other=255)
combined_water = combined_water.where(np.isfinite(combined), other=255)


# =========================================================
# 6. RESTORE SPATIAL METADATA
# =========================================================

for arr in [combined, vv_water, vh_water, combined_water]:
    arr.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    arr.rio.write_crs(vv.rio.crs, inplace=True)


# =========================================================
# 7. PRINT WATER PIXEL SUMMARY
# =========================================================

def count_pixels(mask, name):
    vals = mask.values
    water_count = np.sum(vals == 1)
    nonwater_count = np.sum(vals == 0)
    nodata_count = np.sum(vals == 255)

    print(f"\n{name}")
    print("Water pixels   :", int(water_count))
    print("Non-water pixels:", int(nonwater_count))
    print("NoData pixels  :", int(nodata_count))

count_pixels(vv_water, "VV mask")
count_pixels(vh_water, "VH mask")
count_pixels(combined_water, "Combined mask")


# =========================================================
# 8. SAVE OUTPUT RASTERS
# =========================================================

print("\nSaving output rasters...")

combined.rio.to_raster(
    COMBINED_RASTER_OUT,
    compress="DEFLATE",
    dtype="float32"
)

vv_water.rio.to_raster(
    VV_MASK_OUT,
    compress="DEFLATE",
    dtype="uint8",
    nodata=255
)

vh_water.rio.to_raster(
    VH_MASK_OUT,
    compress="DEFLATE",
    dtype="uint8",
    nodata=255
)

combined_water.rio.to_raster(
    COMBINED_MASK_OUT,
    compress="DEFLATE",
    dtype="uint8",
    nodata=255
)


# =========================================================
# 9. CREATE DIAGNOSTIC FIGURE
# =========================================================

print("Creating diagnostic comparison figure...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top row: continuous rasters
vv.plot(ax=axes[0, 0], cmap="gray", vmin=-25, vmax=0, add_colorbar=True)
axes[0, 0].set_title(f"VV median\nOtsu = {vv_thresh:.3f}")

vh.plot(ax=axes[0, 1], cmap="gray", vmin=-30, vmax=-5, add_colorbar=True)
axes[0, 1].set_title(f"VH median\nOtsu = {vh_thresh:.3f}")

combined.plot(ax=axes[0, 2], cmap="gray", add_colorbar=True)
axes[0, 2].set_title(f"VV + VH median\nOtsu = {combined_thresh:.3f}")

# Bottom row: binary masks
vv_water.plot(ax=axes[1, 0], add_colorbar=False)
axes[1, 0].set_title("Water mask from VV")

vh_water.plot(ax=axes[1, 1], add_colorbar=False)
axes[1, 1].set_title("Water mask from VH")

combined_water.plot(ax=axes[1, 2], add_colorbar=False)
axes[1, 2].set_title("Water mask from VV + VH")

for ax in axes.ravel():
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

plt.tight_layout()

plt.savefig(
    FIGURE_OUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 10. FINAL REPORT
# =========================================================

print("\n======================================")
print("STEP 8 COMPLETE")
print("======================================")
print("Combined raster:")
print(COMBINED_RASTER_OUT)

print("\nVV mask:")
print(VV_MASK_OUT)

print("\nVH mask:")
print(VH_MASK_OUT)

print("\nCombined mask:")
print(COMBINED_MASK_OUT)

print("\nComparison figure:")
print(FIGURE_OUT)
print("======================================")