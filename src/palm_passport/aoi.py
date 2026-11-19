"""Study areas for the Siak prototype.

GAUL has no kecamatan (level 3) polygons, so the first test area is a
20 x 20 km box inside Siak. ``densest_smallholder_box`` picks the box with the
most smallholder palm automatically, so we don't have to guess coordinates.
Swap in BIG / Ina-Geoportal boundaries later.
"""

from __future__ import annotations

import ee

from . import assets
from .palm_mask import SMALLHOLDER, oil_palm_classes

# Riau lies in UTM zone 47N.
UTM_47N = "EPSG:32647"


def district(name: str = "Siak", province: str = "Riau") -> ee.FeatureCollection:
    """District (kabupaten) polygon from FAO GAUL level 2."""
    fc = (
        ee.FeatureCollection(assets.GAUL_LEVEL2)
        .filter(ee.Filter.eq("ADM1_NAME", province))
        .filter(ee.Filter.eq("ADM2_NAME", name))
    )
    return fc


def box(lon: float, lat: float, size_km: float = 20) -> ee.Geometry:
    """Square of ``size_km`` centred on (lon, lat)."""
    return ee.Geometry.Point([lon, lat]).buffer(size_km * 500).bounds()


def densest_smallholder_box(region: ee.FeatureCollection, size_km: float = 20) -> ee.Feature:
    """The ``size_km`` grid cell inside ``region`` with the most smallholder palm.

    Returns a Feature with properties ``smallholder_frac`` (0-1) and
    ``center_lon`` / ``center_lat``.
    """
    grid = region.geometry().coveringGrid(ee.Projection(UTM_47N).atScale(size_km * 1000))
    smallholder = oil_palm_classes().eq(SMALLHOLDER).unmask(0).rename("smallholder_frac")
    scored = smallholder.reduceRegions(
        collection=grid, reducer=ee.Reducer.mean(), scale=100, tileScale=4
    )
    best = ee.Feature(scored.sort("smallholder_frac", False).first())
    center = best.geometry().centroid(maxError=10).coordinates()
    return best.set({"center_lon": center.get(0), "center_lat": center.get(1)})
