import geopandas as gpd

INPUT = "/config/Desktop/AquacultureProject/data/nigeria_adm2.geojson"
OUTPUT = "/config/Desktop/AquacultureProject/data/pilot_regions.geojson"

NAME_FIELD = "adm2nm"

pilot_names = [
    "Ekeremor",
    "Epe",
    "Borgu",
    "Lokoja",
    "Kukawa",
    "Nguru",
]

gdf = gpd.read_file(INPUT)

print("Available names:")
print(sorted(gdf[NAME_FIELD].dropna().unique()))

pilots = gdf[
    gdf[NAME_FIELD]
    .str.lower()
    .isin([x.lower() for x in pilot_names])
].copy()

print("\nSelected:")
print(pilots[NAME_FIELD].tolist())

# 10 km buffer
pilots = pilots.to_crs("EPSG:6933")
pilots["geometry"] = pilots.geometry.buffer(10_000)
pilots = pilots.to_crs("EPSG:4326")

pilots.to_file(OUTPUT, driver="GeoJSON")

print("Saved:", OUTPUT)