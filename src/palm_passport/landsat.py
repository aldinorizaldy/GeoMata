"""Harmonised yearly Landsat composites, 1985 to now (notebook 02).

- Collection 2 Level-2 surface reflectance from Landsat 5, 7, 8 and 9.
- Clouds, shadows and saturated pixels masked with QA_PIXEL / QA_RADSAT.
- TM and ETM+ reflectance mapped onto OLI with the Roy et al. (2016)
  OLS coefficients, so the 40-year series has no sensor jumps.
- One median composite per calendar year.
"""

from __future__ import annotations

import ee

from . import assets

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

_TM_ETM_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
_OLI_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

# Roy et al. 2016, Remote Sensing of Environment 185, Table 2 (ETM+ -> OLI, OLS).
ROY_SLOPES = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
ROY_INTERCEPTS = [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]

# Bits in QA_PIXEL to mask: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
_QA_MASK_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def _prep(image: ee.Image, sr_bands: list[str], harmonise: bool) -> ee.Image:
    qa_ok = image.select("QA_PIXEL").bitwiseAnd(_QA_MASK_BITS).eq(0)
    sat_ok = image.select("QA_RADSAT").eq(0)
    sr = image.select(sr_bands, BANDS).multiply(0.0000275).add(-0.2)
    if harmonise:
        sr = sr.multiply(ee.Image.constant(ROY_SLOPES)).add(ee.Image.constant(ROY_INTERCEPTS))
    sr = sr.updateMask(sr.reduce(ee.Reducer.min()).gt(0))  # drop negative reflectance
    return (
        sr.updateMask(qa_ok.And(sat_ok))
        .float()
        .copyProperties(image, ["system:time_start"])
    )


def surface_reflectance(region: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """All clear-sky Landsat 5/7/8/9 pixels over ``region`` in OLI-equivalent reflectance."""

    def load(key, sr_bands, harmonise):
        return (
            ee.ImageCollection(assets.LANDSAT_C2_L2[key])
            .filterBounds(region)
            .filterDate(start, end)
            .map(lambda img: _prep(img, sr_bands, harmonise))
        )

    return (
        load("LT05", _TM_ETM_BANDS, True)
        .merge(load("LE07", _TM_ETM_BANDS, True))
        .merge(load("LC08", _OLI_BANDS, False))
        .merge(load("LC09", _OLI_BANDS, False))
    )


def nbr(image: ee.Image) -> ee.Image:
    """Normalised Burn Ratio, (NIR - SWIR2) / (NIR + SWIR2)."""
    return image.normalizedDifference(["nir", "swir2"]).rename("NBR")


def ndvi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["nir", "red"]).rename("NDVI")


def annual_composites(region: ee.Geometry, start_year: int, end_year: int) -> ee.ImageCollection:
    """One median composite per year (all six bands plus NBR and NDVI).

    Each image carries ``year``, ``n_scenes`` and ``system:time_start``
    (1 July). Years without any clear pixel are left masked; LandTrendr
    treats masked years as missing.
    """
    sr = surface_reflectance(region, f"{start_year}-01-01", f"{end_year + 1}-01-01")

    def composite(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        scenes = sr.filterDate(start, start.advance(1, "year"))
        median = scenes.median()
        median = ee.Image(
            ee.Algorithms.If(
                scenes.size().gt(0),
                median,
                ee.Image.constant([0] * len(BANDS)).rename(BANDS).float().updateMask(0),
            )
        )
        out = median.addBands(nbr(median)).addBands(ndvi(median))
        return out.set(
            {
                "year": year,
                "n_scenes": scenes.size(),
                "system:time_start": ee.Date.fromYMD(year, 7, 1).millis(),
            }
        )

    return ee.ImageCollection(ee.List.sequence(start_year, end_year).map(composite))


def scene_counts(region: ee.Geometry, start_year: int, end_year: int) -> ee.FeatureCollection:
    """How many Landsat scenes touch ``region`` each year (a quick data-gap check)."""
    sr = surface_reflectance(region, f"{start_year}-01-01", f"{end_year + 1}-01-01")

    def count(year):
        start = ee.Date.fromYMD(year, 1, 1)
        n = sr.filterDate(start, start.advance(1, "year")).size()
        return ee.Feature(None, {"year": year, "n_scenes": n})

    return ee.FeatureCollection(ee.List.sequence(start_year, end_year).map(count))
