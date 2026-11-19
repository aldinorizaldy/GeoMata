# Palm Passport prototype spec (summary)

The living spec is a Claude Doc in the GeoMata project: https://claude.ai/artifact/GL7DLFx2vNBAqC9dzNRYBD

## Goal
A per-plot "Palm Passport" for oil palm smallholders in Siak district, Riau: palm extent,
planting year from the Landsat archive, palm age and replanting priority, and a
deforestation-free-since-2020 (EUDR) check. Yield gap comes later.

## Steps
1. Spec and data sources (done).
2. Earth Engine notebooks: **01 palm mask**, **02 planting year**, for one 20 km box in Siak, then
   the whole district. (This repo, first PR.)
3. Validation against Descals et al. 2024 planting-year map (target to confirm: within 2 years
   for 80% of points).
4. Forest estate (kawasan hutan) overlay from KLHK or the partner.
5. Web demo.

## Data
Landsat C2 L2 (LT05, LE07, LC08, LC09), Sentinel-2 SR harmonised, Sentinel-1 GRD,
BIOPAMA/GlobalOilPalm/v1, Hansen GFC (latest), JRC/GFC2020, FAO GAUL level 2 for boundaries
(BIG / Ina-Geoportal later).

## Rules
Age classes: 0-3, 4-8, 9-25, over 25 years (due for replanting).
