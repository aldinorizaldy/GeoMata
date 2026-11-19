"""Where the oil palm is (notebook 01).

Baseline: the Descals et al. global oil palm map for 2019 (10 m), which
already separates industrial from smallholder plantations. We also add a
first deforestation-free check: palm pixels that were forest on 31 Dec 2020
(JRC GFC2020) or lost tree cover after 2020 (Hansen) are flagged for review.
"""

from __future__ import annotations

import ee

from . import assets

INDUSTRIAL = 1
SMALLHOLDER = 2

PALETTE = {"industrial": "#d95f02", "smallholder": "#1b9e77"}


def oil_palm_classes() -> ee.Image:
    """1 = industrial, 2 = smallholder; masked where there is no palm."""
    classes = ee.ImageCollection(assets.GLOBAL_OIL_PALM).select("classification").mosaic()
    return classes.updateMask(classes.lte(SMALLHOLDER)).rename("palm_class")


def palm_mask() -> ee.Image:
    """1 where there is oil palm of either type, masked elsewhere."""
    return oil_palm_classes().gt(0).selfMask().rename("palm")


def area_by_class_ha(region: ee.Geometry, scale: int = 10) -> ee.Dictionary:
    """Hectares of industrial and smallholder palm inside ``region``."""
    classes = oil_palm_classes()
    area = ee.Image.pixelArea().divide(1e4)
    stats = area.addBands(classes).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="palm_class"),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        tileScale=4,
    )
    groups = ee.List(stats.get("groups"))
    names = ee.Dictionary({str(INDUSTRIAL): "industrial_ha", str(SMALLHOLDER): "smallholder_ha"})

    def to_pair(group):
        group = ee.Dictionary(group)
        key = names.get(ee.Number(group.get("palm_class")).int().format())
        return [key, group.get("sum")]

    return ee.Dictionary(groups.map(to_pair).flatten())


def deforestation_flags(after_year: int = 2020) -> ee.Image:
    """EUDR first pass on palm pixels.

    Bands:
      forest2020   1 where palm now sits on land JRC called forest on 31 Dec 2020
      loss_after   1 where Hansen recorded tree-cover loss after ``after_year``
      flag         1 if either is true (needs a closer look)

    Note: the 2019 palm map predates 2020, so ``forest2020`` on palm pixels
    mostly flags mapping disagreements; ``loss_after`` is the stronger signal.
    """
    palm = palm_mask()
    forest2020 = assets.jrc_forest_2020().unmask(0)
    lossyear = assets.hansen().select("lossyear")
    loss_after = lossyear.gt(after_year - 2000).unmask(0)
    flags = ee.Image.cat(
        [forest2020.rename("forest2020"), loss_after.rename("loss_after")]
    )
    flags = flags.addBands(flags.reduce(ee.Reducer.max()).rename("flag"))
    return flags.updateMask(palm)
