import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
from odc.stac import load

NAME_FIELD = "adm2nm"

pilots = gpd.read_file(
    "/config/Desktop/AquacultureProject/data/pilot_regions.geojson"
)

pilot = pilots[
    pilots[NAME_FIELD].str.lower() == "ekeremor"
].iloc[0]

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

items = list(
    catalog.search(
        collections=["sentinel-1-rtc"],
        intersects=mapping(pilot.geometry),
        datetime="2023-01-01/2025-12-31",
    ).items()
)

centroid = pilot.geometry.centroid
lon = centroid.x
lat = centroid.y

zone = int((lon + 180) // 6) + 1
epsg = (32600 if lat >= 0 else 32700) + zone

print("Local UTM:", epsg)

ds = load(
    items,
    bands=["vv", "vh"],
    bbox=pilot.geometry.bounds,
    crs=f"EPSG:{epsg}",
    resolution=30,
    chunks={"x": 2048, "y": 2048},
)

print(ds)

ds_stats = ds.chunk({
    "time": -1,
    "x": 512,
    "y": 512
})

import numpy as np

# Use only a few scenes for quick diagnostics
small = ds.isel(time=[0, 20, 40])

# Take a small central spatial window
y0 = small.sizes["y"] // 2
x0 = small.sizes["x"] // 2

small = small.isel(
    y=slice(y0 - 100, y0 + 100),
    x=slice(x0 - 100, x0 + 100),
)

print("\nComputing small diagnostic sample...")

sample = small.compute()

vv = sample.vv.values
vh = sample.vh.values

vv = vv[np.isfinite(vv)]
vh = vh[np.isfinite(vh)]

print("\nVV statistics")
print("Min :", vv.min())
print("1%  :", np.quantile(vv, 0.01))
print("50% :", np.quantile(vv, 0.50))
print("99% :", np.quantile(vv, 0.99))
print("Max :", vv.max())

print("\nVH statistics")
print("Min :", vh.min())
print("1%  :", np.quantile(vh, 0.01))
print("50% :", np.quantile(vh, 0.50))
print("99% :", np.quantile(vh, 0.99))
print("Max :", vh.max())