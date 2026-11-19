"""Planting year and replanting from the Landsat archive (notebook 02).

Idea: a palm plot starts with land clearing, which shows up as a sharp drop
in NBR, followed by several years of regrowth as the canopy closes. LandTrendr
fits straight-line segments to each pixel's yearly NBR; the segment with the
biggest drop is the clearing, and its first year is the clearing year. A
second big drop on the same pixel means the plot was cleared again
(replanted), so we keep both the first and the most recent clearing.

LandTrendr expects disturbance to raise the index, so NBR is flipped (x -1)
and scaled by 1000 before fitting. Magnitudes below are in those units:
forest (NBR ~0.6) cleared to bare soil (~0.1) is a magnitude of about 500.
"""

from __future__ import annotations

import datetime as _dt

import ee

from .ages import AGE_CLASSES

DEFAULT_LT_PARAMS = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.05,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}

SEGMENT_FIELDS = ["clear_year", "end_year", "start_val", "end_val", "magnitude", "duration"]


def _lt_series(composites: ee.ImageCollection, index: str = "NBR") -> ee.ImageCollection:
    """Single-band series in LandTrendr's orientation (disturbance = increase)."""

    def flip(img):
        return (
            img.select(index)
            .multiply(-1000)
            .rename(index)
            .set("system:time_start", img.get("system:time_start"))
        )

    return composites.map(flip)


def run_landtrendr(composites: ee.ImageCollection, index: str = "NBR", params: dict | None = None):
    """Run LandTrendr on the yearly composites. Returns the raw output image."""
    lt_params = dict(DEFAULT_LT_PARAMS, **(params or {}))
    lt_params["timeSeries"] = _lt_series(composites, index)
    return ee.Algorithms.TemporalSegmentation.LandTrendr(**lt_params)


def _segments(lt: ee.Image) -> ee.Image:
    """6 x N array per pixel: one column per fitted segment (rows = SEGMENT_FIELDS)."""
    arr = lt.select("LandTrendr")
    vertices = arr.arrayMask(arr.arraySlice(0, 3, 4))  # keep vertex columns
    left = vertices.arraySlice(1, 0, -1)
    right = vertices.arraySlice(1, 1)
    start_year = left.arraySlice(0, 0, 1)
    end_year = right.arraySlice(0, 0, 1)
    start_val = left.arraySlice(0, 2, 3)
    end_val = right.arraySlice(0, 2, 3)
    magnitude = end_val.subtract(start_val)
    duration = end_year.subtract(start_year)
    return ee.Image.cat(
        [start_year.add(1), end_year, start_val, end_val, magnitude, duration]
    ).toArray(0)


def _pick(segments: ee.Image, sort_key: ee.Image, prefix: str) -> ee.Image:
    """Column with the smallest ``sort_key``, flattened to named bands."""
    first = segments.arraySort(sort_key).arraySlice(1, 0, 1)
    return first.arrayProject([0]).arrayFlatten([[f"{prefix}_{f}" for f in SEGMENT_FIELDS]])


def clearing_events(
    lt: ee.Image,
    min_magnitude: float = 200,
    max_duration: int = 4,
) -> ee.Image:
    """Clearing dates from LandTrendr output.

    Output bands (masked where there was no qualifying clearing):
      first_clear_year, first_magnitude   earliest big drop
      last_clear_year,  last_magnitude    most recent big drop (the current planting)
      n_clearings                         how many big drops (2+ = replanted)
      replanted                           1 if n_clearings >= 2
    """
    seg = _segments(lt)
    mag = seg.arraySlice(0, 4, 5)
    dur = seg.arraySlice(0, 5, 6)
    year = seg.arraySlice(0, 0, 1)
    is_clearing = mag.gte(min_magnitude).And(dur.lte(max_duration))

    # Sort keys put qualifying segments first; non-qualifying ones get a large
    # key, so even if none qualify the arrays never go empty (we mask below).
    big = ee.Image.constant(1e6)
    first_key = ee.Image(big).subtract(is_clearing.multiply(big.subtract(year)))
    last_key = ee.Image(big).subtract(is_clearing.multiply(big.add(year)))
    # first_key: qualifying -> year (earliest wins); otherwise 1e6.
    # last_key:  qualifying -> -year (latest wins);  otherwise 1e6.

    first = _pick(seg, first_key, "first")
    last = _pick(seg, last_key, "last")
    n = is_clearing.arrayReduce(ee.Reducer.sum(), [1]).arrayGet([0, 0]).rename("n_clearings")

    out = ee.Image.cat(
        [
            first.select("first_clear_year").int16(),
            first.select("first_magnitude").float(),
            last.select("last_clear_year").int16(),
            last.select("last_magnitude").float(),
            n.int8(),
            n.gte(2).rename("replanted").int8(),
        ]
    )
    return out.updateMask(n.gte(1))


def palm_age(clearings: ee.Image, as_of_year: int | None = None, lag_years: int = 1) -> ee.Image:
    """Planting year, age and age class from the most recent clearing.

    Bands: planting_year, age, age_class (1-4, see ``ages.AGE_CLASSES``).
    """
    as_of_year = as_of_year or _dt.date.today().year
    planting = clearings.select("last_clear_year").add(lag_years).rename("planting_year")
    age = ee.Image.constant(as_of_year).subtract(planting).max(0).rename("age")
    age_class = ee.Image(0)
    for i, (_, lo, _hi) in enumerate(AGE_CLASSES, start=1):
        age_class = age_class.where(age.gte(lo), i)
    age_class = age_class.updateMask(age.mask()).rename("age_class")
    return ee.Image.cat([planting.int16(), age.int16(), age_class.int8()])


def hansen_agreement(
    clearings: ee.Image, hansen: ee.Image, region: ee.Geometry, tolerance: int = 1, scale: int = 30
) -> ee.Dictionary:
    """Self-check against Hansen loss year (2001+ only, forest-to-palm clearings).

    Returns pixel counts where both maps see a clearing, and the share that
    agree within ``tolerance`` years.
    """
    loss_year = hansen.select("lossyear").selfMask().add(2000)
    ours = clearings.select("last_clear_year")
    both = ours.updateMask(loss_year.gte(2001)).updateMask(ours.gte(2001))
    diff = both.subtract(loss_year).abs()
    stats = (
        diff.lte(tolerance)
        .rename("agree")
        .reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=region,
            scale=scale,
            maxPixels=1e10,
            tileScale=4,
        )
    )
    return stats
