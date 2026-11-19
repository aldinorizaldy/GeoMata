import pytest

from palm_passport.ages import AGE_CLASSES, AGE_CLASS_PALETTE, age_class, planting_year_from_clearing


@pytest.mark.parametrize(
    "age, expected",
    [(0, 1), (3, 1), (4, 2), (8, 2), (9, 3), (25, 3), (26, 4), (40, 4)],
)
def test_age_class_boundaries(age, expected):
    assert age_class(age) == expected


def test_negative_age_rejected():
    with pytest.raises(ValueError):
        age_class(-1)


def test_palette_matches_classes():
    assert len(AGE_CLASS_PALETTE) == len(AGE_CLASSES)


def test_planting_lag():
    assert planting_year_from_clearing(2005) == 2006
    assert planting_year_from_clearing(2005, lag_years=0) == 2005
