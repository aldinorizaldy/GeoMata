"""Notebook code cells must at least parse (Earth Engine calls need sign-in, so they aren't run)."""

import ast
import json
import pathlib

import pytest

NOTEBOOKS = sorted(pathlib.Path(__file__).parents[1].joinpath("notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(path):
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in src.splitlines())
        ast.parse(src)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_saved_outputs(path):
    nb = json.loads(path.read_text())
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_landsat_harmonisation_coefficients():
    from palm_passport import landsat

    assert len(landsat.ROY_SLOPES) == len(landsat.ROY_INTERCEPTS) == len(landsat.BANDS)
