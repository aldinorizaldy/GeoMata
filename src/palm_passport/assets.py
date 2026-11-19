"""Earth Engine catalog asset IDs used by Palm Passport.

Some datasets get new versions every year (Hansen) or have a V1/V2 split
(JRC GFC2020). Each entry lists candidates newest first; ``resolve`` returns
the first one that exists in the catalog, so the notebooks keep working when
a new version lands.
"""

from __future__ import annotations

from functools import lru_cache

import ee

LANDSAT_C2_L2 = {
    "LT05": "LANDSAT/LT05/C02/T1_L2",  # Landsat 5 TM, 1984-2012
    "LE07": "LANDSAT/LE07/C02/T1_L2",  # Landsat 7 ETM+, 1999-2022 (SLC-off after May 2003)
    "LC08": "LANDSAT/LC08/C02/T1_L2",  # Landsat 8 OLI, 2013-
    "LC09": "LANDSAT/LC09/C02/T1_L2",  # Landsat 9 OLI-2, 2021-
}

SENTINEL2_SR = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
SENTINEL1_GRD = "COPERNICUS/S1_GRD"

# Descals et al. global oil palm map (2019, 10 m). Band "classification":
# 1 = industrial closed-canopy, 2 = smallholder closed-canopy, 3 = other.
GLOBAL_OIL_PALM = "BIOPAMA/GlobalOilPalm/v1"

GAUL_LEVEL2 = "FAO/GAUL/2015/level2"

CANDIDATES = {
    # EU Forest Observatory forest cover for 31 Dec 2020 (EUDR cut-off date).
    "jrc_gfc2020": ["JRC/GFC2020/V2", "JRC/GFC2020/V1"],
    # Hansen Global Forest Change; band "lossyear" counts years since 2000.
    "hansen": [
        "UMD/hansen/global_forest_change_2025_v1_13",
        "UMD/hansen/global_forest_change_2024_v1_12",
        "UMD/hansen/global_forest_change_2023_v1_11",
    ],
}


@lru_cache(maxsize=None)
def resolve(key: str) -> str:
    """Return the newest asset ID for ``key`` that exists in the catalog."""
    for asset_id in CANDIDATES[key]:
        try:
            ee.data.getAsset(asset_id)
            return asset_id
        except ee.EEException:
            continue
    raise RuntimeError(f"None of {CANDIDATES[key]} exist; update palm_passport.assets.")


def jrc_forest_2020() -> ee.Image:
    """Forest cover on 31 Dec 2020 (1 = forest), masked elsewhere."""
    image = ee.ImageCollection(resolve("jrc_gfc2020")).mosaic().select("Map")
    return image.eq(1).selfMask().rename("forest2020")


def hansen() -> ee.Image:
    return ee.Image(resolve("hansen"))


def sentinel2_rgb(region: ee.Geometry, start: str, end: str, clear_threshold: float = 0.6) -> ee.Image:
    """Cloud-free Sentinel-2 median (B4, B3, B2 reflectance 0-1) for a basemap."""
    s2 = (
        ee.ImageCollection(SENTINEL2_SR)
        .filterBounds(region)
        .filterDate(start, end)
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS), ["cs_cdf"])
        .map(lambda img: img.updateMask(img.select("cs_cdf").gte(clear_threshold)))
    )
    return s2.select(["B4", "B3", "B2"]).median().divide(10000).clip(region)
