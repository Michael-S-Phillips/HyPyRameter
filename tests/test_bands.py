import numpy as np
import pytest

from hypyrameter.bands import (
    apply_bad_bands,
    band,
    bands_between,
    closest_band,
    closest_wavelength,
    fill_nans_along_spectrum,
    to_nanometres,
)


def test_to_nanometres_converts_micrometres():
    np.testing.assert_allclose(to_nanometres([0.4, 1.0, 2.5]), [400.0, 1000.0, 2500.0])
    np.testing.assert_allclose(to_nanometres([400.0, 1000.0]), [400.0, 1000.0])


def test_closest_band_and_wavelength(wavelengths):
    assert closest_band(wavelengths, 1001.0) == 86  # 400 + 86*7 = 1002
    assert closest_wavelength(wavelengths, 1001.0) == 1002.0


def test_band_median_window_is_symmetric_and_the_requested_width(wavelengths):
    cube = np.zeros((1, 1, wavelengths.size))
    cube[0, 0, :] = np.arange(wavelengths.size)  # value == band index
    centre = closest_band(wavelengths, 1002.0)

    assert band(cube, wavelengths, 1002.0, width=5)[0, 0] == centre  # bands c-2..c+2
    assert band(cube, wavelengths, 1002.0, width=3)[0, 0] == centre
    assert band(cube, wavelengths, 1002.0, width=1)[0, 0] == centre


def test_band_window_is_clamped_at_the_ends_instead_of_wrapping(wavelengths):
    cube = np.zeros((1, 1, wavelengths.size))
    cube[0, 0, :] = np.arange(wavelengths.size)

    assert band(cube, wavelengths, 400.0, width=5)[0, 0] == 1.0  # median of bands 0..2
    last = wavelengths.size - 1
    assert band(cube, wavelengths, 2600.0, width=5)[0, 0] == last - 1


def test_band_ignores_nan_bands_in_the_window_but_stays_nan_when_all_are():
    wavelengths = np.arange(400.0, 470.0, 10.0)
    cube = np.array([[[1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]]])
    assert band(cube, wavelengths, 420.0, width=3)[0, 0] == 3.5  # median of 3, 4
    cube[0, 0, :] = np.nan
    assert np.isnan(band(cube, wavelengths, 420.0, width=3)[0, 0])


def test_band_works_for_a_table_of_spectra_and_a_single_spectrum(wavelengths):
    table = np.tile(np.arange(wavelengths.size, dtype=float), (3, 1))
    assert band(table, wavelengths, 1002.0).shape == (3,)
    assert band(table[0], wavelengths, 1002.0).shape == ()


def test_bands_between_is_inclusive(wavelengths):
    idx = bands_between(wavelengths, 1002.0, 1030.0)
    np.testing.assert_array_equal(wavelengths[idx], [1002.0, 1009.0, 1016.0, 1023.0, 1030.0])


def test_fill_nans_interpolates_along_the_spectrum():
    cube = np.tile(np.arange(6.0), (1, 2, 1))
    cube[0, 0, 2] = np.nan
    cube[0, 1, :] = np.nan
    filled = fill_nans_along_spectrum(cube)
    assert filled[0, 0, 2] == 2.0
    assert np.isnan(filled[0, 1]).all()


def test_apply_bad_bands_sets_flagged_bands_to_nan():
    cube = np.ones((2, 2, 4))
    out = apply_bad_bands(cube, [1, 0, 1, 0])
    assert np.isnan(out[..., 1]).all() and np.isnan(out[..., 3]).all()
    assert np.isfinite(out[..., 0]).all()
    with pytest.raises(ValueError):
        apply_bad_bands(cube, [1, 0])
