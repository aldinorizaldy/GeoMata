# GeoMata: Palm Passport

Per-plot intelligence for Indonesian oil palm smallholders, from free satellite data on
Google Earth Engine. For each palm plot: where it is, when it was planted (from 40 years of
Landsat), how old it is, whether it is due for replanting, and a first deforestation-free
(EUDR, cut-off 31 Dec 2020) check.

The first prototype covers **Siak district, Riau**, starting with one 20 x 20 km box.

## Run it in Google Colab (no laptop needed)

| Notebook | What it does | |
|---|---|---|
| 01 Palm mask | Industrial vs smallholder palm, hectares, EUDR flags | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldinorizaldy/geomata/blob/main/notebooks/01_palm_mask.ipynb) |
| 02 Planting year | Clearing/planting year, palm age, age class, replanting, Hansen self-check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldinorizaldy/geomata/blob/main/notebooks/02_planting_year.ipynb) |

1. One time: register a Google Cloud project for Earth Engine at
   https://code.earthengine.google.com/register and copy its project ID.
2. Open a notebook, paste the ID into `PROJECT_ID` in the first cell, then **Runtime > Run all**.
3. Sign in to Google when Colab asks.

Earth Engine is free for noncommercial and research use. Once Palm Passport is sold to
customers, Earth Engine needs a commercial licence; Google offers trials for that.

## Method (short)

- **Palm mask:** Descals et al. global oil palm map 2019 (`BIOPAMA/GlobalOilPalm/v1`, 10 m).
- **Planting year:** yearly median Landsat 5/7/8/9 Collection 2 composites, TM/ETM+ harmonised
  to OLI (Roy et al. 2016), NBR series, LandTrendr. The biggest abrupt NBR drop is the land
  clearing; planting year = clearing year + 1. Two big drops = replanted.
- **Age classes:** 0-3, 4-8, 9-25, over 25 years (due for replanting).
- **EUDR first pass:** palm on JRC GFC2020 forest, or Hansen tree-cover loss after 2020.
- **Checks:** Hansen loss-year agreement now; Descals et al. 2024 planting-year map as the
  benchmark once uploaded.

Full spec: [docs/spec.md](docs/spec.md).

## Repository layout

```
notebooks/            Colab notebooks (01 palm mask, 02 planting year)
src/palm_passport/    Earth Engine code the notebooks import
tests/                pytest (pure-Python parts and notebook syntax)
legacy/streamlit-demo The earlier Sentinel-2 + WorldCover Streamlit app (also on the v0-demo branch)
```

## Develop locally

```bash
pip install -e ".[notebooks,dev]"
pytest
```
