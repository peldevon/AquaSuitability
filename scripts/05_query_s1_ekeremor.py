import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc

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

search = catalog.search(
    collections=["sentinel-1-rtc"],
    intersects=mapping(pilot.geometry),
    datetime="2023-01-01/2025-12-31",
)

items = list(search.items())

print("Scenes:", len(items))

if items:
    print("First scene:", items[0].id)
    print("Assets:")
    print(list(items[0].assets.keys()))