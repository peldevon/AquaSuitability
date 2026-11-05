import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc

PILOT_FILE = "data/pilot_regions.geojson"
NAME_FIELD = "adm2nm"

names = [
    "Ekeremor",
    "Borgu",
    "Kukawa"
]

gdf = gpd.read_file(PILOT_FILE)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace
)

for name in names:

    subset = gdf[
        gdf[NAME_FIELD].str.lower() == name.lower()
    ]

    if len(subset) == 0:
        print(name, "NOT FOUND")
        continue

    geom = subset.iloc[0].geometry

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(geom),
        datetime="2023-01-01/2025-12-31",
        query={
            "eo:cloud_cover": {
                "lt": 60
            }
        }
    )

    items = list(search.items())

    print(
        name,
        "Sentinel-2 scenes:",
        len(items)
    )

    if items:
        print(
            "Assets:",
            list(items[0].assets.keys())
        )