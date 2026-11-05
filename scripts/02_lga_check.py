import geopandas as gpd

path = "/config/Desktop/AquacultureProject/data/nigeria_adm2.geojson"

gdf = gpd.read_file(path)

print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:")
print(gdf.columns.tolist())

print("\nInvalid geometries:")
print((~gdf.geometry.is_valid).sum())

print("\nFirst rows:")
print(gdf.head())

# Repair if needed
gdf["geometry"] = gdf.geometry.make_valid()

# Equal-area national check
gdf_eq = gdf.to_crs("EPSG:6933")
gdf_eq["area_km2"] = gdf_eq.geometry.area / 1_000_000

print("\nApproximate total polygon area:")
print(gdf_eq["area_km2"].sum())