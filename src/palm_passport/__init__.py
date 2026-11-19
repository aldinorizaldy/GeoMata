"""Palm Passport: per-plot intelligence for Indonesian oil palm smallholders.

Prototype modules, all built on Google Earth Engine:

- ``auth``          Earth Engine sign-in for Colab or a local machine.
- ``assets``        Catalog asset IDs, with fallbacks when a version moves.
- ``aoi``           Study areas (Siak district, a 20 km box inside it).
- ``palm_mask``     Where the oil palm is, split into industrial and smallholder.
- ``landsat``       Harmonised yearly Landsat composites, 1985 to now.
- ``planting_year`` Clearing year (planting year) and replanting from LandTrendr.
- ``ages``          Palm age classes (pure Python, no Earth Engine needed).
"""

__version__ = "0.1.0"
