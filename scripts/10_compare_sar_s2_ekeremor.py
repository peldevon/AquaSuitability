import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray

from rasterio.enums import Resampling


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = "/config/Desktop/AquacultureProject"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"

SAR_VV = f"{OUTPUT_DIR}/ekeremor_water_vv_otsu.tif"
SAR_VH = f"{OUTPUT_DIR}/ekeremor_water_vh_otsu.tif"
SAR_COMBINED = f"{OUTPUT_DIR}/ekeremor_water_combined_otsu.tif"

S2_FILE = f"{OUTPUT_DIR}/ekeremor_water_mndwi_otsu.tif"

CSV_OUT = f"{OUTPUT_DIR}/ekeremor_sar_s2_agreement.csv"
FIGURE_OUT = f"{OUTPUT_DIR}/ekeremor_sar_s2_agreement.png"

AGREEMENT_VV_OUT = f"{OUTPUT_DIR}/ekeremor_agreement_vv_s2.tif"
AGREEMENT_VH_OUT = f"{OUTPUT_DIR}/ekeremor_agreement_vh_s2.tif"
AGREEMENT_COMBINED_OUT = (
    f"{OUTPUT_DIR}/ekeremor_agreement_combined_s2.tif"
)


# =========================================================
# 1. LOAD RASTERS
# =========================================================

print("Loading SAR water masks...")

vv = rioxarray.open_rasterio(
    SAR_VV,
    masked=False
).squeeze()

vh = rioxarray.open_rasterio(
    SAR_VH,
    masked=False
).squeeze()

combined = rioxarray.open_rasterio(
    SAR_COMBINED,
    masked=False
).squeeze()

print("Loading Sentinel-2 water mask...")

s2 = rioxarray.open_rasterio(
    S2_FILE,
    masked=False
).squeeze()


# =========================================================
# 2. DISPLAY ORIGINAL GRID INFORMATION
# =========================================================

print("\nOriginal grids")

print("\nSAR:")
print("CRS:", combined.rio.crs)
print("Resolution:", combined.rio.resolution())
print("Shape:", combined.shape)

print("\nSentinel-2:")
print("CRS:", s2.rio.crs)
print("Resolution:", s2.rio.resolution())
print("Shape:", s2.shape)


# =========================================================
# 3. REPROJECT SENTINEL-2 TO SAR GRID
#
# Binary/categorical data -> nearest neighbour only
# =========================================================

print("\nAligning Sentinel-2 to the 30 m SAR grid...")

s2_30m = s2.rio.reproject_match(
    combined,
    resampling=Resampling.nearest
)

print("Aligned Sentinel-2 grid:")
print("Resolution:", s2_30m.rio.resolution())
print("Shape:", s2_30m.shape)


# =========================================================
# 4. HELPER FUNCTION FOR VALID MASKS
# =========================================================

def valid_binary(arr):
    """
    Valid values must be exactly 0 or 1.
    Value 255 is NoData.
    """
    return (
        np.isfinite(arr.values)
        & np.isin(arr.values, [0, 1])
    )


# =========================================================
# 5. AGREEMENT ANALYSIS FUNCTION
# =========================================================

def compare_masks(sar, optical, name):

    sar_arr = sar.values
    opt_arr = optical.values

    valid = (
        valid_binary(sar)
        & valid_binary(optical)
    )

    s = sar_arr[valid].astype(np.uint8)
    o = opt_arr[valid].astype(np.uint8)

    # Agreement categories:
    #
    # 0 = both non-water
    # 1 = both water
    # 2 = SAR water only
    # 3 = Sentinel-2 water only
    # 255 = NoData

    agreement = np.full(
        sar_arr.shape,
        255,
        dtype=np.uint8
    )

    both_nonwater = (
        valid
        & (sar_arr == 0)
        & (opt_arr == 0)
    )

    both_water = (
        valid
        & (sar_arr == 1)
        & (opt_arr == 1)
    )

    sar_only = (
        valid
        & (sar_arr == 1)
        & (opt_arr == 0)
    )

    s2_only = (
        valid
        & (sar_arr == 0)
        & (opt_arr == 1)
    )

    agreement[both_nonwater] = 0
    agreement[both_water] = 1
    agreement[sar_only] = 2
    agreement[s2_only] = 3

    total = len(s)

    n_both_nonwater = np.sum((s == 0) & (o == 0))
    n_both_water = np.sum((s == 1) & (o == 1))
    n_sar_only = np.sum((s == 1) & (o == 0))
    n_s2_only = np.sum((s == 0) & (o == 1))

    overall_agreement = (
        (n_both_nonwater + n_both_water)
        / total
    )

    water_intersection = n_both_water

    water_union = (
        n_both_water
        + n_sar_only
        + n_s2_only
    )

    water_iou = (
        water_intersection / water_union
        if water_union > 0
        else np.nan
    )

    sar_water = np.sum(s == 1)
    s2_water = np.sum(o == 1)

    # Dice / F1-style cross-sensor overlap
    dice = (
        2 * n_both_water
        / (sar_water + s2_water)
        if (sar_water + s2_water) > 0
        else np.nan
    )

    print("\n======================================")
    print(name)
    print("======================================")
    print("Valid pixels        :", total)
    print("Both non-water      :", n_both_nonwater)
    print("Both water          :", n_both_water)
    print("SAR water only      :", n_sar_only)
    print("Sentinel-2 water only:", n_s2_only)

    print("\nOverall agreement   :", overall_agreement)
    print("Water IoU           :", water_iou)
    print("Water Dice/F1 overlap:", dice)

    stats = {
        "classifier": name,
        "valid_pixels": total,
        "both_nonwater": n_both_nonwater,
        "both_water": n_both_water,
        "sar_water_only": n_sar_only,
        "s2_water_only": n_s2_only,
        "overall_agreement": overall_agreement,
        "water_iou": water_iou,
        "water_dice": dice,
    }

    return agreement, stats


# =========================================================
# 6. RUN COMPARISONS
# =========================================================

agreement_vv, stats_vv = compare_masks(
    vv,
    s2_30m,
    "VV Otsu"
)

agreement_vh, stats_vh = compare_masks(
    vh,
    s2_30m,
    "VH Otsu"
)

agreement_combined, stats_combined = compare_masks(
    combined,
    s2_30m,
    "VV + VH Otsu"
)


# =========================================================
# 7. SAVE STATISTICS
# =========================================================

df = pd.DataFrame([
    stats_vv,
    stats_vh,
    stats_combined
])

df.to_csv(
    CSV_OUT,
    index=False
)

print("\nAgreement statistics saved:")
print(CSV_OUT)

print("\nSummary table:")
print(
    df[
        [
            "classifier",
            "overall_agreement",
            "water_iou",
            "water_dice"
        ]
    ].to_string(index=False)
)


# =========================================================
# 8. SAVE AGREEMENT RASTERS
# =========================================================

def save_agreement(
    template,
    arr,
    output_path
):

    out = template.copy(
        data=arr
    )

    out.rio.write_nodata(
        255,
        inplace=True
    )

    out.rio.to_raster(
        output_path,
        dtype="uint8",
        compress="DEFLATE",
        nodata=255
    )


print("\nSaving agreement rasters...")

save_agreement(
    combined,
    agreement_vv,
    AGREEMENT_VV_OUT
)

save_agreement(
    combined,
    agreement_vh,
    AGREEMENT_VH_OUT
)

save_agreement(
    combined,
    agreement_combined,
    AGREEMENT_COMBINED_OUT
)


# =========================================================
# 9. CREATE COMPARISON FIGURE
# =========================================================

print("Creating comparison figure...")

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

vv.plot(
    ax=axes[0, 0],
    add_colorbar=False
)
axes[0, 0].set_title(
    "Sentinel-1 VV Water"
)

vh.plot(
    ax=axes[0, 1],
    add_colorbar=False
)
axes[0, 1].set_title(
    "Sentinel-1 VH Water"
)

combined.plot(
    ax=axes[0, 2],
    add_colorbar=False
)
axes[0, 2].set_title(
    "Sentinel-1 VV + VH Water"
)

s2_30m.plot(
    ax=axes[1, 0],
    add_colorbar=False
)
axes[1, 0].set_title(
    "Sentinel-2 MNDWI Water\naligned to 30 m"
)

agreement_da = combined.copy(
    data=agreement_combined
)

agreement_da.plot(
    ax=axes[1, 1],
    add_colorbar=True
)
axes[1, 1].set_title(
    "Combined SAR vs S2 Agreement"
)

axes[1, 2].axis("off")

summary_text = (
    "Cross-sensor comparison\n\n"
    f"VV IoU: "
    f"{stats_vv['water_iou']:.3f}\n"
    f"VH IoU: "
    f"{stats_vh['water_iou']:.3f}\n"
    f"VV+VH IoU: "
    f"{stats_combined['water_iou']:.3f}\n\n"
    f"VV Dice: "
    f"{stats_vv['water_dice']:.3f}\n"
    f"VH Dice: "
    f"{stats_vh['water_dice']:.3f}\n"
    f"VV+VH Dice: "
    f"{stats_combined['water_dice']:.3f}"
)

axes[1, 2].text(
    0.05,
    0.95,
    summary_text,
    va="top",
    fontsize=13
)

for ax in axes.ravel():
    if ax.has_data():
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
# 10. IDENTIFY BEST CROSS-SENSOR MATCH
# =========================================================

best = df.sort_values(
    "water_iou",
    ascending=False
).iloc[0]

print("\n======================================")
print("STEP 10 COMPLETE")
print("======================================")

print("Best SAR candidate by Water IoU:")
print(best["classifier"])

print("\nWater IoU:")
print(best["water_iou"])

print("\nWater Dice:")
print(best["water_dice"])

print("\nIMPORTANT:")
print(
    "This is cross-sensor agreement, "
    "NOT independent validation."
)

print("\nCSV:")
print(CSV_OUT)

print("\nFigure:")
print(FIGURE_OUT)

print("======================================")