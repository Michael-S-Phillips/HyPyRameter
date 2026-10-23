import numpy as np
import pytest
from conftest import gaussian_absorption

from hypyrameter import parameters
from hypyrameter.parameters import (
    PARAMETERS,
    Spectra,
    band_area,
    band_depth,
    band_depth_invert,
    clamp_reflectance,
    compute,
    parameter_names,
    valid_parameters,
)

# Every parameter the 0.2.x cube calculator offered must still exist.
LEGACY_NAMES = [
    "R463",
    "R550",
    "R637",
    "R1080",
    "R1506",
    "R2529",
    "HCPINDEX2",
    "LCPINDEX2",
    "OLINDEX3",
    "ESINDEX",
    "SINDEX2",
    "GINDEX",
    "CPLINDEX",
    "CHLORINDEX",
    "BD530_2",
    "BD670",
    "BD875",
    "BD905",
    "BD920_2",
    "BD1200",
    "BD1300",
    "BD1400",
    "BD1450",
    "BD1750",
    "BD1900_2",
    "BD1900r2",
    "BD2100_2",
    "BD2100_3",
    "BD2165",
    "BD2190",
    "BD2210_2",
    "BD2250",
    "BD2265",
    "BD2290",
    "BD2443",
    "BD2355",
    "BD2600",
    "BDCARB",
    "BA1200",
    "BA1450",
    "BA1900",
    "SH460",
    "D700",
    "D2200",
    "D2300",
    "MIN2295_2480",
    "MIN2250",
    "MIN2345_2537",
    "BH1500",
    "RPEAK1",
    "RPEAK1_2",
    "BDI1000VIS",
    "SLOPE420_500",
    "SLOPE1815_2530",
    "BR800",
    "BR2530",
    "BR3500",
    "NDVI",
    "NDWI",
    "NDMI",
]


def test_every_legacy_parameter_is_registered_with_bounds_and_a_description():
    names = parameter_names()
    missing = [n for n in LEGACY_NAMES if n not in names]
    assert missing == []
    for name in names:
        p = PARAMETERS[name]
        assert p.bounds[0] <= p.bounds[1]
        assert p.description and p.group


def test_valid_parameters_follow_the_wavelength_range(wavelengths):
    assert valid_parameters(wavelengths) == [
        n for n in parameter_names() if PARAMETERS[n].bounds[1] < 2606
    ]
    vnir = valid_parameters(np.arange(400.0, 1001.0, 5.0))
    assert "BD530_2" in vnir and "BD2290" not in vnir
    assert "BR3500" not in valid_parameters(wavelengths)
    assert valid_parameters([0.4, 0.5, 0.6, 0.7]) == valid_parameters([400, 500, 600, 700])


def test_band_depth_of_a_gaussian_absorption(wavelengths):
    cube = np.empty((2, 3, wavelengths.size))
    cube[..., :] = 0.4 * gaussian_absorption(wavelengths, 1930.0, 0.25, 30.0)
    s = Spectra(cube, wavelengths)

    depth = band_depth(s, 1850, 1930, 2067, mw=1)

    np.testing.assert_allclose(depth, 0.25, atol=0.01)
    assert band_depth_invert(s, 1850, 1930, 2067, mw=1)[0, 0] < 0  # a trough, not a peak
    assert band_area(s, 1850, 2067)[0, 0] > 0


def test_flat_spectrum_has_zero_depth_slope_and_unit_ratio(flat_cube, wavelengths):
    out = compute(flat_cube, wavelengths, ["BD2290", "SLOPE1815_2530", "BR2530", "NDVI", "D2300"])
    np.testing.assert_allclose(out, 0.0 * out + np.array([0, 0, 1, 0, 0]), atol=1e-6)


def test_nan_stays_nan_instead_of_becoming_the_minimum(flat_cube, wavelengths):
    cube = flat_cube.copy()
    cube[0, 0, :] = np.nan  # a nodata pixel
    cube[1, 1, 100:110] = 0.05  # a genuine deep feature elsewhere

    out = compute(cube, wavelengths, ["BD1200", "SLOPE420_500", "BR800", "HCPINDEX2", "MIN2250"])

    assert np.isnan(out[0, 0]).all()
    assert np.isfinite(out[1:, 1:]).all()


def test_compute_defaults_to_every_valid_parameter_and_reports_progress(flat_cube, wavelengths):
    seen = []
    out = compute(flat_cube, wavelengths, progress=lambda f, name: seen.append((f, name)))
    assert out.shape == (4, 5, len(valid_parameters(wavelengths)))
    assert out.dtype == np.float32
    assert seen[0][1] == valid_parameters(wavelengths)[0] and seen[-1] == (1.0, "done")


def test_compute_works_on_a_table_and_a_single_spectrum(wavelengths):
    spectrum = 0.4 * gaussian_absorption(wavelengths, 2290.0, 0.1, 20.0)
    table = np.tile(spectrum, (3, 1))
    assert compute(table, wavelengths, ["BD2290"]).shape == (3, 1)
    single = compute(spectrum, wavelengths, ["BD2290"])
    assert single.shape == (1,) and single[0] > 0.05


def test_rpeak1_finds_the_visible_peak_and_bdi1000vis_uses_it(wavelengths):
    cube = np.empty((2, 2, wavelengths.size))
    cube[..., :] = 0.5 * np.exp(-0.5 * ((wavelengths - 750.0) / 250.0) ** 2) + 0.1
    out = compute(cube, wavelengths, ["BDI1000VIS", "RPEAK1"])

    np.testing.assert_allclose(out[..., 1], 0.75, atol=0.02)  # µm
    assert np.isfinite(out[..., 0]).all()


def test_unknown_parameter_is_an_error(flat_cube, wavelengths):
    with pytest.raises(KeyError, match="NOPE"):
        compute(flat_cube, wavelengths, ["NOPE"])


def test_clamp_reflectance_drops_fill_values():
    out = clamp_reflectance(np.array([0.5, 1.5, -9999.0, -0.2]))
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])


def test_bad_wavelength_count_is_an_error():
    with pytest.raises(ValueError):
        parameters.Spectra(np.zeros((2, 2, 5)), [400, 500, 600])


def test_bdi1000vis_is_the_integrated_band_depth_below_the_peak(wavelengths):
    """Viviano-Beck et al. (2014): R833-R1023 divided by RPEAK1, then the
    integral of 1 - R/RPEAK1 (µm): 0 without a 1 µm band, positive and
    growing with one. (0.3.0 integrated R/RPEAK1 itself, over 833-989 nm.)"""
    flat = np.full(wavelengths.size, 0.3)
    assert compute(flat, wavelengths, ["BDI1000VIS"])[0] == pytest.approx(0, abs=1e-6)

    shallow = 0.3 * gaussian_absorption(wavelengths, 1000.0, 0.1, 100.0)
    deep = 0.3 * gaussian_absorption(wavelengths, 1000.0, 0.3, 100.0)
    out = compute(np.stack([shallow, deep]), wavelengths, ["BDI1000VIS"])[:, 0]
    assert 0 < out[0] < out[1]
    # the peak is the unabsorbed 0.3, so the value is the depth's integral
    fine = np.linspace(833.0, 1023.0, 2001)
    depth = 0.1 * np.exp(-0.5 * ((fine - 1000.0) / 100.0) ** 2)
    assert out[0] == pytest.approx(np.trapezoid(depth, fine / 1000.0), rel=0.05)


@pytest.fixture
def long_wavelengths():
    """CRISM's MTRDR range, 436-3920 nm every 6.55 nm."""
    return np.arange(436.0, 3925.0, 6.55)


# The CRISM parameters added in 0.4.0 (CRISM Data Product SIS, Table 3-12;
# Viviano-Beck et al. 2014): a featureless spectrum gives 0
NEW = ["SH600_2", "BD1435", "BD1500_2", "ISLOPE1", "BD2500_2", "BD3000", "CINDEX2"]


def test_the_new_crism_parameters_are_zero_without_their_feature(long_wavelengths):
    flat = np.full(long_wavelengths.size, 0.3)
    out = compute(flat, long_wavelengths, NEW)
    np.testing.assert_allclose(out, 0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "centre", "absorbing"),
    [
        ("BD1435", 1435.0, True),
        ("BD1500_2", 1525.0, True),
        ("BD2500_2", 2480.0, True),
        ("BD3000", 3000.0, True),
        ("SH600_2", 600.0, False),  # a shoulder: higher than its neighbors
    ],
)
def test_the_new_crism_parameters_see_their_feature(long_wavelengths, name, centre, absorbing):
    depth = 0.2 if absorbing else -0.2
    spectrum = 0.3 * gaussian_absorption(long_wavelengths, centre, depth, 25.0)
    value = compute(spectrum, long_wavelengths, [name])[0]
    assert value > 0.05, (name, value)


def test_cindex2_sees_the_convexity_between_two_carbonate_bands(long_wavelengths):
    spectrum = (
        0.3
        * gaussian_absorption(long_wavelengths, 3450.0, 0.2, 60.0)
        * gaussian_absorption(long_wavelengths, 3875.0, 0.2, 60.0)
    )
    assert compute(spectrum, long_wavelengths, ["CINDEX2"])[0] > 0.05


def test_islope1_is_the_crism_name_of_islope(long_wavelengths):
    sloped = 0.2 + 0.0001 * (long_wavelengths - 1000.0)
    out = compute(sloped, long_wavelengths, ["ISLOPE1", "ISLOPE"])
    assert out[0] == pytest.approx(out[1])
    # (R1815 - R2530) / (W2530 - W1815), wavelengths in µm
    assert out[0] == pytest.approx(-0.1, rel=0.01)


def test_the_browse_reflectances_use_the_sis_kernels(long_wavelengths):
    for name in ("R440", "R530", "R600", "R770", "R1330", "R3920"):
        assert name in PARAMETERS
    spike = np.full(long_wavelengths.size, 0.3)
    nearest = int(np.argmin(np.abs(long_wavelengths - 1330.0)))
    spike[nearest - 4 : nearest + 5] = 0.6  # 9 of IRA's 11 bands
    # an 11-band median sees mostly the spike; a 5-band one would see only it
    assert compute(spike, long_wavelengths, ["R1330"])[0] == pytest.approx(0.6)
    spike[nearest - 4 : nearest - 1] = 0.3
    spike[nearest + 2 : nearest + 5] = 0.3  # now only 3 of 11
    assert compute(spike, long_wavelengths, ["R1330"])[0] == pytest.approx(0.3)
