import numpy as np

from hypyrameter.fits import polynomial_integral, polynomial_peak, smoothed_peak, spline_area


def test_polynomial_peak_recovers_a_planted_peak():
    x = np.linspace(442.0, 989.0, 13)
    peaks = np.array([[600.0, 750.0], [800.0, 900.0]])
    Y = 1.0 - ((x - peaks[..., None]) / 300.0) ** 2  # parabolas peaking at `peaks`

    peak_x, peak_y = polynomial_peak(x, Y, degree=5, samples=2001)

    np.testing.assert_allclose(peak_x, peaks, atol=1.0)
    np.testing.assert_allclose(peak_y, 1.0, atol=1e-3)


def test_polynomial_peak_is_nan_for_spectra_with_missing_values():
    x = np.linspace(442.0, 989.0, 13)
    Y = np.ones((3, 13))
    Y[1, 4] = np.nan
    peak_x, peak_y = polynomial_peak(x, Y)
    assert np.isnan(peak_x[1]) and np.isnan(peak_y[1])
    assert np.isfinite(peak_x[[0, 2]]).all()


def test_smoothed_peak_survives_noise():
    rng = np.random.default_rng(0)
    x = np.arange(500.0, 1151.0, 7.0)
    Y = 1.0 - ((x - 800.0) / 400.0) ** 2 + 0.002 * rng.normal(size=(50, x.size))
    peak_x, _ = smoothed_peak(x, Y)
    np.testing.assert_allclose(peak_x, 800.0, atol=15.0)


def test_polynomial_integral_matches_the_analytic_integral():
    x = np.linspace(0.833, 0.989, 20)
    Y = np.stack([x**2, 2 * x + 1])
    expected = np.array([(x[-1] ** 3 - x[0] ** 3) / 3, (x[-1] ** 2 - x[0] ** 2) + (x[-1] - x[0])])
    np.testing.assert_allclose(polynomial_integral(x, Y, degree=4), expected, rtol=1e-6)


def test_spline_area_of_a_triangle():
    x = np.array([0.0, 1.0, 2.0])
    Y = np.array([[0.0, 1.0, 0.0], [0.0, np.nan, 0.0]])
    area = spline_area(x, Y)
    assert area[0] > 0.9 and np.isnan(area[1])
