from pystac_client import Client
import planetary_computer as pc

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

for cid in ["sentinel-1-rtc", "sentinel-2-l2a"]:
    c = catalog.get_collection(cid)
    print(cid, "->", c.title)

print("Catalog connection OK")