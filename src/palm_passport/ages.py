"""Palm age classes from the spec (pure Python, no Earth Engine)."""

from __future__ import annotations

# (label, min_age_inclusive, max_age_inclusive); None = open-ended.
AGE_CLASSES = [
    ("immature (0-3 yr)", 0, 3),
    ("young (4-8 yr)", 4, 8),
    ("prime (9-25 yr)", 9, 25),
    ("due for replanting (>25 yr)", 26, None),
]

AGE_CLASS_PALETTE = ["#fee08b", "#a6d96a", "#1a9641", "#d7191c"]


def age_class(age: int) -> int:
    """Index into ``AGE_CLASSES`` for a palm age in whole years (1-based)."""
    if age < 0:
        raise ValueError("age cannot be negative")
    for i, (_, lo, hi) in enumerate(AGE_CLASSES, start=1):
        if age >= lo and (hi is None or age <= hi):
            return i
    raise AssertionError("unreachable")


def planting_year_from_clearing(clearing_year: int, lag_years: int = 1) -> int:
    """Planting usually follows land clearing by about a year."""
    return clearing_year + lag_years
