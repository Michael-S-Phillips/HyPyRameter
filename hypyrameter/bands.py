"""Wavelength lookups and band extraction.

Every function takes reflectance data with the wavelength axis *last* — an
image cube ``(rows, cols, bands)``, a table of spectra ``(n, bands)`` or a
single spectrum ``(bands,)`` — and wavelengths in nanometres. Missing data is
NaN and stays NaN: nothing here replaces a NaN with a number.
"""

from __future__ import annotations

import warnings

import numpy as np

# Below this, wavelengths are taken to be micrometres
_MICRON_LIMIT = 10.0


def to_nanometres(wavelengths) -> np.ndarray:
    """Wavelengths as a float array in nm (values below 10 are read as µm)."""
    values = np.asarray(wavelengths, dtype=np.float64).ravel()
    if values.size and np.nanmax(values) < _MICRON_LIMIT:
        return values * 1000.0
    return values


def closest_band(wavelengths: np.ndarray, target: float) -> int:
    """Index of the band closest to ``target`` (nm)."""
    return int(np.argmin(np.abs(np.asarray(wavelengths) - target)))


def closest_wavelength(wavelengths: np.ndarray, target: float) -> float:
    """The wavelength (nm) of the band closest to ``target``."""
    return float(np.asarray(wavelengths)[closest_band(wavelengths, target)])


def band(cube: np.ndarray, wavelengths: np.ndarray, target: float, width: int = 5) -> np.ndarray:
    """Reflectance at the band closest to ``target`` nm, as the NaN-ignoring
    median over ``width`` bands centred on it (clamped at the ends of the
    wavelength range). ``width`` 1 is the band itself."""
    cube = np.asarray(cube)
    centre = closest_band(wavelengths, target)
    if width <= 1:
        return cube[..., centre]
    half = (int(width) - 1) // 2
    lo = max(centre - half, 0)
    hi = min(centre + half, cube.shape[-1] - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        return np.nanmedian(cube[..., lo : hi + 1], axis=-1)


def bands_between(wavelengths: np.ndarray, low: float, high: float) -> np.ndarray:
    """Indices of the bands from the one closest to ``low`` to the one closest
    to ``high``, inclusive."""
    start, stop = closest_band(wavelengths, low), closest_band(wavelengths, high)
    if stop < start:
        start, stop = stop, start
    return np.arange(start, stop + 1)


def fill_nans_along_spectrum(cube: np.ndarray, min_valid_fraction: float = 0.5) -> np.ndarray:
    """A copy of ``cube`` with each spectrum's NaN bands filled by linear
    interpolation along the spectrum (nearest value beyond the ends). Spectra
    with fewer than ``min_valid_fraction`` of their bands finite stay NaN."""
    data = np.array(cube, dtype=np.float64, copy=True)
    flat = data.reshape(-1, data.shape[-1])
    bands = flat.shape[1]
    finite = np.isfinite(flat)
    needed = max(1 if bands == 1 else 2, int(np.ceil(min_valid_fraction * bands)))
    usable = finite.sum(axis=1) >= needed
    flat[~usable] = np.nan
    rows = np.flatnonzero(usable & ~finite.all(axis=1))
    idx = np.arange(bands)
    for start in range(0, rows.size, 16_384):
        sel = rows[start : start + 16_384]
        block, ok = flat[sel], finite[sel]
        prev_idx = np.maximum.accumulate(np.where(ok, idx, -1), axis=1)
        next_rev = np.maximum.accumulate(np.where(ok[:, ::-1], idx, -1), axis=1)
        next_idx = np.where(next_rev >= 0, bands - 1 - next_rev, -1)[:, ::-1]
        has_prev, has_next = prev_idx >= 0, next_idx >= 0
        prev_val = np.take_along_axis(block, np.clip(prev_idx, 0, bands - 1), axis=1)
        next_val = np.take_along_axis(block, np.clip(next_idx, 0, bands - 1), axis=1)
        both = has_prev & has_next
        span = np.where(both & (next_idx > prev_idx), next_idx - prev_idx, 1)
        weight = np.where(both, (idx - prev_idx) / span, 0.0)
        filled = np.where(
            both, prev_val + weight * (next_val - prev_val), np.where(has_prev, prev_val, next_val)
        )
        flat[sel] = np.where(ok, block, filled)
    return data


def apply_bad_bands(cube: np.ndarray, bad_band_list) -> np.ndarray:
    """A float copy of ``cube`` with the bands flagged 0 in an ENVI-style bad
    band list (1 = good, 0 = bad) set to NaN."""
    data = np.array(cube, dtype=np.float64, copy=True)
    flags = np.asarray(bad_band_list).ravel()
    if flags.size != data.shape[-1]:
        raise ValueError(f"bad band list has {flags.size} entries for {data.shape[-1]} bands")
    data[..., flags == 0] = np.nan
    return data
