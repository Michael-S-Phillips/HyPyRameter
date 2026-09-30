"""Vectorised curve fits used by the peak, area and integral parameters.

These replace per-pixel ``np.polyfit`` / spline calls (and the process pools
that fed them): a polynomial fit with a shared x axis is one least-squares
solve for every spectrum at once.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# Spectra evaluated per chunk when sampling fitted curves (memory bound)
_CHUNK = 50_000


def _scaled_vandermonde(x: np.ndarray, degree: int, x_ref: np.ndarray) -> np.ndarray:
    """Vandermonde matrix of ``x`` mapped onto [-1, 1] by ``x_ref``'s range, so
    wavelength-scale x values (hundreds of nm) stay well conditioned."""
    centre = 0.5 * (x_ref.min() + x_ref.max())
    half = 0.5 * (x_ref.max() - x_ref.min()) or 1.0
    return np.vander((x - centre) / half, degree + 1)


def fit_polynomials(x: np.ndarray, Y: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares polynomial coefficients (in the scaled basis of
    ``_scaled_vandermonde``) for every spectrum of ``Y`` (..., n): returns
    (..., degree + 1). Spectra with a non-finite value get NaN coefficients."""
    x = np.asarray(x, dtype=np.float64)
    flat = np.asarray(Y, dtype=np.float64).reshape(-1, x.size)
    good = np.isfinite(flat).all(axis=1)
    coefficients = np.full((flat.shape[0], degree + 1), np.nan)
    if good.any():
        V = _scaled_vandermonde(x, degree, x)
        coefficients[good] = np.linalg.lstsq(V, flat[good].T, rcond=None)[0].T
    return coefficients.reshape(Y.shape[:-1] + (degree + 1,))


def polynomial_peak(
    x: np.ndarray, Y: np.ndarray, degree: int = 5, samples: int = 521
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial to each spectrum and locate its maximum on a fine grid
    over ``x``'s range. Returns (peak x, peak value), each shaped like
    ``Y[..., 0]``; NaN where a spectrum could not be fitted."""
    x = np.asarray(x, dtype=np.float64)
    coefficients = fit_polynomials(x, Y, degree).reshape(-1, degree + 1)
    grid = np.linspace(x.min(), x.max(), samples)
    VgT = np.ascontiguousarray(_scaled_vandermonde(grid, degree, x).T)  # (degree + 1, samples)
    peak_x = np.full(coefficients.shape[0], np.nan)
    peak_y = np.full(coefficients.shape[0], np.nan)
    good = np.isfinite(coefficients).all(axis=1)
    for start in range(0, coefficients.shape[0], _CHUNK):
        sel = np.flatnonzero(good[start : start + _CHUNK]) + start
        if not sel.size:
            continue
        # np.dot, not @: numpy 2.2 on macOS raises spurious RuntimeWarnings
        # from single-row matmul
        values = np.dot(coefficients[sel], VgT)  # (m, samples)
        at = np.argmax(values, axis=1)
        peak_x[sel] = grid[at]
        peak_y[sel] = values[np.arange(sel.size), at]
    shape = np.asarray(Y).shape[:-1]
    return peak_x.reshape(shape), peak_y.reshape(shape)


def smoothed_peak(
    x: np.ndarray, Y: np.ndarray, window: int = 7, polyorder: int = 3, degree: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky–Golay smooth each spectrum, then ``polynomial_peak``."""
    Y = np.asarray(Y, dtype=np.float64)
    window = min(window, Y.shape[-1] if Y.shape[-1] % 2 else Y.shape[-1] - 1)
    if window <= polyorder:
        smoothed = Y
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            smoothed = savgol_filter(Y, window_length=window, polyorder=polyorder, axis=-1)
    return polynomial_peak(x, smoothed, degree=degree)


def polynomial_integral(x: np.ndarray, Y: np.ndarray, degree: int = 4) -> np.ndarray:
    """Integral over [x0, x_last] of a polynomial fitted to each spectrum."""
    x = np.asarray(x, dtype=np.float64)
    coefficients = fit_polynomials(x, Y, degree)
    # integrate the scaled polynomial analytically, then rescale to x units
    half = 0.5 * (x.max() - x.min()) or 1.0
    powers = np.arange(degree, -1, -1)  # np.vander orders highest power first
    lo, hi = -1.0, 1.0
    antiderivative = (hi ** (powers + 1) - lo ** (powers + 1)) / (powers + 1)
    return np.dot(coefficients, antiderivative) * half


def spline_area(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Integral over ``x`` of the cubic spline through each spectrum of ``Y``
    (NaN where a spectrum has a non-finite value)."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    flat = Y.reshape(-1, x.size)
    area = np.full(flat.shape[0], np.nan)
    good = np.isfinite(flat).all(axis=1)
    if good.any():
        area[good] = CubicSpline(x, flat[good], axis=1).integrate(x[0], x[-1])
    return area.reshape(Y.shape[:-1])
