"""Browse products: three parameters stretched into an RGB image."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import numpy as np


def _load_definitions() -> dict[str, tuple[str, str, str]]:
    text = resources.files("hypyrameter").joinpath("data/browse_products.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return {row["BrowseProduct"]: (row["Param1"], row["Param2"], row["Param3"]) for row in rows}


BROWSE_PRODUCTS: dict[str, tuple[str, str, str]] = _load_definitions()


def valid_browse_products(available: Mapping[str, object] | list[str]) -> list[str]:
    """Browse products whose three parameters are all in ``available``."""
    names = set(available)
    return [name for name, params in BROWSE_PRODUCTS.items() if names.issuperset(params)]


def stretch(
    values: np.ndarray, kind: str = "mad", percent: float = 2.0, factor: float = 2.5
) -> np.ndarray:
    """Map an array onto 0-255 (float): ``linear`` clips at the ``percent``
    percentiles, ``std`` at mean ± factor·std, ``mad`` at median ± factor·MAD.
    NaN is ignored when finding the limits and stays NaN."""
    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.full(data.shape, np.nan)
    if kind == "linear":
        lo, hi = np.percentile(finite, percent), np.percentile(finite, 100 - percent)
    elif kind == "std":
        centre, spread = finite.mean(), finite.std()
        lo, hi = centre - factor * spread, centre + factor * spread
    elif kind == "mad":
        centre = np.median(finite)
        spread = np.median(np.abs(finite - centre))
        lo, hi = centre - factor * spread, centre + factor * spread
    else:
        raise ValueError(f"unknown stretch {kind!r}; use linear, std or mad")
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data - lo) * 255.0 / (hi - lo), 0.0, 255.0)


def browse_image(
    parameters: Mapping[str, np.ndarray],
    product: str,
    *,
    kind: str = "mad",
    percent: float = 2.0,
    factor: float = 2.5,
) -> np.ndarray:
    """An (rows, cols, 3) uint8 RGB image of a browse product from a mapping
    of parameter name to (rows, cols) array. NaN becomes black."""
    names = BROWSE_PRODUCTS[product]
    channels = [stretch(parameters[name], kind, percent, factor) for name in names]
    rgb = np.stack(channels, axis=-1)
    return np.nan_to_num(rgb, nan=0.0).astype(np.uint8)


def write_browse_products(
    parameters: Mapping[str, np.ndarray], folder: str | Path, products=None, **stretch_options
) -> list[Path]:
    """Write a PNG per browse product (all valid ones by default) into ``folder``."""
    import imageio.v3 as iio  # optional dependency ("browse" extra)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for product in products or valid_browse_products(parameters):
        path = folder / f"{product}.png"
        iio.imwrite(path, browse_image(parameters, product, **stretch_options))
        written.append(path)
    return written
