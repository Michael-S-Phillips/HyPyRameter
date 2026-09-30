"""The spectral parameter library.

Each parameter is registered with its name, the wavelength range it needs,
a description and the function that computes it from a :class:`Spectra`.
``compute`` evaluates any set of them on an image cube, a table of spectra
or a single spectrum; ``valid_parameters`` says which ones a wavelength
range supports.

Definitions follow Viviano-Beck et al. (2014) for the CRISM parameters plus
the additions described in the HyPyRameter paper; see each parameter's
``description``. Missing data is NaN throughout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hypyrameter.bands import (
    band,
    bands_between,
    closest_wavelength,
    fill_nans_along_spectrum,
    to_nanometres,
)
from hypyrameter.fits import polynomial_integral, polynomial_peak, smoothed_peak, spline_area

# Wavelength slack (nm) allowed when deciding whether a parameter fits a range
DEFAULT_TOLERANCE_NM = 5.0

ProgressCallback = Callable[[float, str], None]


class Spectra:
    """Reflectance with the wavelength axis last, wavelengths in nm, and a
    cache of band lookups so parameters that share anchors do not re-extract
    them. ``results`` holds by-products other parameters depend on."""

    def __init__(self, cube, wavelengths) -> None:
        self.cube = np.asarray(cube)
        if self.cube.dtype.kind != "f":
            self.cube = self.cube.astype(np.float64)
        self.wavelengths = to_nanometres(wavelengths)
        if self.wavelengths.size != self.cube.shape[-1]:
            raise ValueError(f"{self.wavelengths.size} wavelengths for {self.cube.shape[-1]} bands")
        self.results: dict[str, np.ndarray] = {}
        self._bands: dict[tuple[float, int], np.ndarray] = {}

    def R(self, wavelength: float, width: int = 5) -> np.ndarray:
        """Reflectance at the band nearest ``wavelength`` (median over ``width`` bands)."""
        key = (float(wavelength), int(width))
        if key not in self._bands:
            self._bands[key] = band(self.cube, self.wavelengths, wavelength, width)
        return self._bands[key]

    def W(self, wavelength: float) -> float:
        """The actual wavelength of the band nearest ``wavelength``."""
        return closest_wavelength(self.wavelengths, wavelength)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.cube.shape[:-1]


@dataclass(frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float]  # wavelength range (nm) the parameter needs
    description: str
    group: str
    func: Callable[[Spectra], np.ndarray]
    depends: tuple[str, ...] = field(default=())


PARAMETERS: dict[str, Parameter] = {}


def parameter(
    name: str,
    bounds: tuple[float, float],
    description: str,
    group: str,
    depends: Sequence[str] = (),
):
    """Register a parameter function (in definition order)."""

    def register(func: Callable[[Spectra], np.ndarray]):
        PARAMETERS[name] = Parameter(name, bounds, description, group, func, tuple(depends))
        return func

    return register


# --- building blocks ----------------------------------------------------------


def band_depth(s: Spectra, low, mid, hi, lw=5, mw=5, hw=5) -> np.ndarray:
    """1 - R(mid) / continuum, the continuum being the line between R(low) and R(hi)."""
    WL, WM, WH = s.W(low), s.W(mid), s.W(hi)
    a = (WM - WL) / (WH - WL)  # weight of the longer-wavelength anchor
    b = 1.0 - a
    return 1.0 - s.R(mid, mw) / (b * s.R(low, lw) + a * s.R(hi, hw))


def band_depth_invert(s: Spectra, low, mid, hi, lw=5, mw=5, hw=5) -> np.ndarray:
    """1 - continuum / R(mid): the height of a shoulder or peak above the continuum."""
    WL, WM, WH = s.W(low), s.W(mid), s.W(hi)
    a = (WM - WL) / (WH - WL)
    b = 1.0 - a
    return 1.0 - (b * s.R(low, lw) + a * s.R(hi, hw)) / s.R(mid, mw)


def band_area(s: Spectra, low, high, lw=5, hw=5) -> np.ndarray:
    """Area between the continuum line (R(low) to R(high)) and the spectrum,
    integrated by cubic spline; positive for an absorption."""
    y1, y2 = s.R(low, lw), s.R(high, hw)
    x1, x2 = s.W(low), s.W(high)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2
    idx = bands_between(s.wavelengths, low, high)
    wol = s.wavelengths[idx]
    continuum = slope[..., None] * wol + intercept[..., None]
    height = s.cube[..., idx] - continuum
    return -spline_area(wol, height)


def slope(s: Spectra, low, high, width=5) -> np.ndarray:
    """Reflectance change per nm between the bands nearest ``low`` and ``high``."""
    return (s.R(high, width) - s.R(low, width)) / (s.W(high) - s.W(low))


def band_ratio(s: Spectra, numerator, denominator, nw=5, dw=5) -> np.ndarray:
    return s.R(numerator, nw) / s.R(denominator, dw)


def normalized_difference(s: Spectra, a, b) -> np.ndarray:
    Ra, Rb = s.R(a), s.R(b)
    return (Ra - Rb) / (Ra + Rb)


def continuum_at(s: Spectra, low, high, wavelength, lw=5, hw=5) -> np.ndarray:
    """The line between R(low) and R(high), evaluated at ``wavelength``."""
    Rlow, Rhigh = s.R(low, lw), s.R(high, hw)
    WL, WH = s.W(low), s.W(high)
    return Rlow + (Rhigh - Rlow) / (WH - WL) * (s.W(wavelength) - WL)


def _fmin(*arrays: np.ndarray) -> np.ndarray:
    """Element-wise minimum ignoring NaN (NaN only where every input is NaN)."""
    out = arrays[0]
    for other in arrays[1:]:
        out = np.fmin(out, other)
    return out


# --- reflectance --------------------------------------------------------------

for _wl in (463, 550, 637, 1080, 1506, 2529):

    def _reflectance(s: Spectra, wl=_wl) -> np.ndarray:
        return s.R(wl)

    parameter(f"R{_wl}", (_wl, _wl), f"Reflectance at {_wl} nm (5-band median)", "Reflectance")(
        _reflectance
    )


# --- indices ------------------------------------------------------------------


@parameter(
    "HCPINDEX2",
    (1690, 2530),
    (
        "High-calcium pyroxene index: weighted continuum-relative depths between 2120 and "
        "2460 nm (continuum 1690-2530 nm)"
    ),
    "Index",
)
def HCPINDEX2(s: Spectra) -> np.ndarray:
    total = 0.0
    for wl, weight in (
        (2120, 0.1),
        (2140, 0.1),
        (2230, 0.15),
        (2250, 0.3),
        (2430, 0.2),
        (2460, 0.15),
    ):
        total = total + (1 - s.R(wl) / continuum_at(s, 1690, 2530, wl)) * weight
    return total


@parameter(
    "LCPINDEX2",
    (1560, 2450),
    (
        "Low-calcium pyroxene index: weighted continuum-relative depths between 1690 and "
        "1870 nm (continuum 1560-2450 nm)"
    ),
    "Index",
)
def LCPINDEX2(s: Spectra) -> np.ndarray:
    total = 0.0
    for wl, weight in ((1690, 0.2), (1750, 0.2), (1810, 0.3), (1870, 0.3)):
        total = total + (1 - s.R(wl) / continuum_at(s, 1560, 2450, wl)) * weight
    return total


@parameter(
    "OLINDEX3",
    (1210, 1862),
    (
        "Olivine index: weighted continuum-relative depths between 1210 and 1330 nm "
        "(continuum 1750-1862 nm)"
    ),
    "Index",
)
def OLINDEX3(s: Spectra) -> np.ndarray:
    total = 0.0
    for wl, weight in ((1210, 0.1), (1250, 0.1), (1263, 0.2), (1276, 0.2), (1330, 0.4)):
        Rc = continuum_at(s, 1750, 1862, wl)
        total = total + ((Rc - s.R(wl)) / np.abs(Rc)) * weight
    return total


@parameter(
    "ESINDEX",
    (420, 950),
    "Elemental sulfur index: steep 420-500 nm slope minus |550-950 nm slope| (per µm)",
    "Index",
)
def ESINDEX(s: Spectra) -> np.ndarray:
    return slope(s, 420, 500) * 1000.0 - np.abs(slope(s, 550, 950) * 1000.0)


@parameter(
    "SINDEX2", (2120, 2400), "Sulfate index: inverted band depth at 2290 nm (2120-2400 nm)", "Index"
)
def SINDEX2(s: Spectra) -> np.ndarray:
    return band_depth_invert(s, 2120, 2290, 2400, mw=7, hw=3)


@parameter(
    "GINDEX",
    (1440, 1570),
    "Gypsum index: minimum of the 1447, 1491 and 1540 nm band depths",
    "Index",
)
def GINDEX(s: Spectra) -> np.ndarray:
    return _fmin(
        band_depth(s, 1440, 1447, 1476, 1, 1, 1),
        band_depth(s, 1476, 1491, 1515, 1, 1, 1),
        band_depth(s, 1515, 1540, 1570, 1, 1, 1),
    )


@parameter(
    "CPLINDEX", (600, 710), "Chlorophyll index: minimum of the 626 and 678 nm band depths", "Index"
)
def CPLINDEX(s: Spectra) -> np.ndarray:
    return _fmin(band_depth(s, 600, 626, 646), band_depth(s, 646, 678, 710))


@parameter(
    "CHLORINDEX",
    (1800, 2450),
    (
        "Chlorite index: weighted 2347, 2254, 2000 and 1921 nm depths and 2105 nm "
        "shoulder, minus BD2210_2"
    ),
    "Index",
)
def CHLORINDEX(s: Spectra) -> np.ndarray:
    weights = np.array((0.25, 0.25, 0.25, 0.25, 0.05))
    weights = weights / weights.sum()
    terms = (
        band_depth(s, 2286, 2347, 2408),
        band_depth(s, 2184, 2254, 2279),
        band_depth(s, 1986, 2000, 2030),
        band_depth(s, 1888, 1921, 1962),
        band_depth_invert(s, 2000, 2105, 2257),
    )
    total = sum(term * weight for term, weight in zip(terms, weights, strict=True))
    return total - band_depth(s, 2165, 2210, 2290)


# --- band depths ----------------------------------------------------------------

_BAND_DEPTHS = {
    # name: (bounds, low, mid, hi, widths, description)
    "BD530_2": ((440, 614), 440, 530, 614, {}, "530 nm band depth (ferric oxides)"),
    "BD670": ((620, 745), 620, 670, 745, {}, "670 nm band depth (custom; hematite/chlorophyll)"),
    "BD875": ((747, 980), 747, 875, 980, {}, "875 nm band depth (custom; ferric minerals)"),
    "BD905": ((750, 1300), 750, 905, 1300, {}, "905 nm band depth"),
    "BD920_2": ((807, 984), 807, 920, 984, {}, "920 nm band depth (ferric minerals)"),
    "BD1200": ((1115, 1260), 1115, 1200, 1260, {}, "1200 nm band depth"),
    "BD1300": ((1080, 1750), 1260, 1320, 1750, {"mw": 15}, "1300 nm band depth (plagioclase)"),
    "BD1400": ((1330, 1467), 1330, 1395, 1467, {"mw": 3}, "1400 nm band depth (hydration)"),
    "BD1450": ((1340, 1535), 1340, 1450, 1535, {"mw": 3}, "1450 nm band depth (hydration)"),
    "BD1750": ((1688, 1820), 1688, 1750, 1820, {}, "1750 nm band depth (gypsum)"),
    "BD1900_2": ((1850, 2067), 1850, 1930, 2067, {}, "1930 nm band depth (H2O)"),
    "BD2100_2": (
        (1930, 2250),
        1930,
        2132,
        2250,
        {"lw": 3, "hw": 3},
        "2130 nm band depth (monohydrated sulfates)",
    ),
    "BD2100_3": ((2016, 2220), 2016, 2100, 2220, {}, "2100 nm band depth"),
    "BD2165": (
        (2120, 2230),
        2120,
        2165,
        2230,
        {"mw": 3, "hw": 3},
        "2165 nm band depth (kaolinite group)",
    ),
    "BD2190": (
        (2120, 2250),
        2120,
        2185,
        2250,
        {"mw": 3, "hw": 3},
        "2190 nm band depth (beidellite, allophane)",
    ),
    "BD2210_2": ((2165, 2290), 2165, 2210, 2290, {}, "2210 nm band depth (Al-OH phyllosilicates)"),
    "BD2250": ((2120, 2340), 2120, 2245, 2340, {"mw": 7, "hw": 3}, "2250 nm band depth"),
    "BD2265": ((2120, 2340), 2120, 2265, 2340, {"mw": 3, "hw": 5}, "2265 nm band depth (jarosite)"),
    "BD2290": ((2250, 2350), 2250, 2290, 2350, {}, "2290 nm band depth (Fe/Mg phyllosilicates)"),
    "BD2443": ((2320, 2480), 2320, 2443, 2480, {}, "2443 nm band depth (nitrate)"),
    "BD2355": ((2300, 2450), 2300, 2355, 2450, {}, "2355 nm band depth (Fe/Mg phyllosilicates)"),
    "BD2600": ((2530, 2630), 2530, 2600, 2630, {}, "2600 nm band depth (H2O)"),
}

for _name, (_bounds, _low, _mid, _hi, _widths, _description) in _BAND_DEPTHS.items():

    def _depth(s: Spectra, low=_low, mid=_mid, hi=_hi, widths=_widths) -> np.ndarray:
        return band_depth(s, low, mid, hi, **widths)

    parameter(_name, _bounds, _description, "Band depth")(_depth)


@parameter(
    "BD1900r2",
    (1815, 2132),
    (
        "1900 nm band depth, ratio form: mean continuum-relative reflectance at 1908-1941 "
        "nm over that at 1862-1875 and 2112-2126 nm (continuum 1815-2132 nm)"
    ),
    "Band depth",
)
def BD1900r2(s: Spectra) -> np.ndarray:
    def relative(wl):
        return s.R(wl, 1) / continuum_at(s, 1815, 2132, wl)

    inside = sum(relative(wl) for wl in (1908, 1914, 1921, 1928, 1934, 1941))
    outside = sum(relative(wl) for wl in (1862, 1869, 1875, 2112, 2120, 2126))
    return 1.0 - inside / outside


@parameter(
    "BDCARB",
    (2230, 2600),
    "Carbonate band depth: geometric mean of the 2330 and 2530 nm band depths",
    "Band depth",
)
def BDCARB(s: Spectra) -> np.ndarray:
    WL1, WH1 = s.W(2230), s.W(2390)
    WC1 = 0.5 * (s.W(2330) + s.W(2320))
    a = (WC1 - WL1) / (WH1 - WL1)
    b = 1.0 - a
    WL2, WH2 = s.W(2390), s.W(2600)
    WC2 = 0.5 * (s.W(2530) + s.W(2520))
    c = (WC2 - WL2) / (WH2 - WL2)
    d = 1.0 - c
    first = (0.5 * (s.R(2320) + s.R(2330))) / (b * s.R(2230) + a * s.R(2390))
    second = (0.5 * (s.R(2520) + s.R(2530))) / (d * s.R(2390) + c * s.R(2600))
    return 1.0 - np.sqrt(first * second)


# --- band areas ---------------------------------------------------------------

for _name, _low, _high, _description in (
    ("BA1200", 1115, 1260, "1200 nm band area"),
    ("BA1450", 1340, 1535, "1450 nm band area"),
    ("BA1900", 1850, 2067, "1900 nm band area"),
):

    def _area(s: Spectra, low=_low, high=_high) -> np.ndarray:
        return band_area(s, low, high)

    parameter(_name, (_low, _high), _description, "Band area")(_area)


# --- shoulders and drop-offs ---------------------------------------------------


@parameter("SH460", (420, 520), "460 nm shoulder height (inverted band depth)", "Shoulder")
def SH460(s: Spectra) -> np.ndarray:
    return band_depth_invert(s, 420, 460, 520)


@parameter("D460", (420, 520), "Alias of SH460 (point-spectra name)", "Shoulder")
def D460(s: Spectra) -> np.ndarray:
    return SH460(s)


@parameter(
    "D700",
    (630, 830),
    (
        "700 nm drop-off (chlorophyll red edge): continuum-relative 690-720 nm over "
        "740-770 nm (continuum 630-830 nm)"
    ),
    "Drop-off",
)
def D700(s: Spectra) -> np.ndarray:
    def relative(wl, width=5):
        return s.R(wl, width) / continuum_at(s, 630, 830, wl)

    numerator = relative(690, 3) + relative(710, 3) + relative(720, 3)
    denominator = relative(740) + relative(760) + relative(770)
    return 1.0 - numerator / denominator


@parameter(
    "D2200",
    (1815, 2430),
    "2200 nm drop-off: continuum-relative 2210 and 2230 nm over 2165 nm (continuum 1815-2430 nm)",
    "Drop-off",
)
def D2200(s: Spectra) -> np.ndarray:
    def relative(wl):
        return s.R(wl) / continuum_at(s, 1815, 2430, wl)

    return 1.0 - (relative(2210) + relative(2230)) / (2 * relative(2165))


@parameter(
    "D2300",
    (1815, 2530),
    "2300 nm drop-off: continuum-relative 2290-2330 nm over 2120-2210 nm (continuum 1815-2530 nm)",
    "Drop-off",
)
def D2300(s: Spectra) -> np.ndarray:
    def relative(wl, width=5):
        return s.R(wl, width) / continuum_at(s, 1815, 2530, wl)

    numerator = relative(2290, 3) + relative(2320, 3) + relative(2330, 3)
    denominator = relative(2120) + relative(2170) + relative(2210)
    return 1.0 - numerator / denominator


# --- minima -------------------------------------------------------------------


@parameter(
    "MIN2295_2480",
    (2165, 2570),
    "Minimum of the 2295 and 2480 nm band depths (Mg carbonate)",
    "Minimum",
)
def MIN2295_2480(s: Spectra) -> np.ndarray:
    return _fmin(band_depth(s, 2165, 2295, 2364), band_depth(s, 2364, 2480, 2570))


@parameter("MIN2250", (2165, 2350), "Minimum of the 2210 and 2265 nm band depths (opal)", "Minimum")
def MIN2250(s: Spectra) -> np.ndarray:
    return _fmin(band_depth(s, 2165, 2210, 2350), band_depth(s, 2165, 2265, 2350))


@parameter(
    "MIN2345_2537",
    (2250, 2602),
    "Minimum of the 2345 and 2537 nm band depths (Ca/Fe carbonate)",
    "Minimum",
)
def MIN2345_2537(s: Spectra) -> np.ndarray:
    return _fmin(band_depth(s, 2250, 2345, 2430), band_depth(s, 2430, 2537, 2602))


# --- peaks, slopes, ratios -------------------------------------------------------


@parameter("BH1500", (1250, 1750), "1500 nm band height (inverted band depth)", "Shoulder")
def BH1500(s: Spectra) -> np.ndarray:
    return band_depth_invert(s, 1250, 1510, 1750)


_RPEAK1_ANCHORS = (442, 533, 600, 710, 740, 775, 800, 833, 860, 892, 925, 963, 989)


@parameter(
    "RPEAK1",
    (442, 989),
    (
        "Wavelength (µm) of the reflectance peak of a 5th-order polynomial through 13 "
        "anchors between 442 and 989 nm"
    ),
    "Peak",
)
def RPEAK1(s: Spectra) -> np.ndarray:
    idx = np.array([np.argmin(np.abs(s.wavelengths - wl)) for wl in _RPEAK1_ANCHORS])
    idx = np.unique(idx)
    x = s.wavelengths[idx]
    Y = s.cube[..., idx]
    peak_x, peak_y = polynomial_peak(x, Y, degree=5, samples=521)
    all_zero = np.all(Y == 0, axis=-1)
    peak_x = np.where(all_zero, np.nan, peak_x)
    s.results["RPEAK1_reflectance"] = np.where(all_zero, np.nan, peak_y)
    return peak_x / 1000.0


@parameter(
    "RPEAK1_2",
    (500, 1150),
    (
        "Wavelength (µm) of the reflectance peak between 500 and 1150 nm after "
        "Savitzky-Golay smoothing"
    ),
    "Peak",
)
def RPEAK1_2(s: Spectra) -> np.ndarray:
    idx = bands_between(s.wavelengths, 500, 1150)
    peak_x, peak_y = smoothed_peak(s.wavelengths[idx], s.cube[..., idx])
    s.results["RPEAK1_2_reflectance"] = peak_y
    return peak_x / 1000.0


@parameter(
    "BDI1000VIS",
    (833, 989),
    (
        "Integrated 1000 nm band depth (VIS): integral of a 4th-order polynomial through "
        "the 833-989 nm reflectance normalised by the RPEAK1 peak"
    ),
    "Peak",
    depends=("RPEAK1",),
)
def BDI1000VIS(s: Spectra) -> np.ndarray:
    peak = s.results.get("RPEAK1_reflectance")
    if peak is None:
        RPEAK1(s)
        peak = s.results["RPEAK1_reflectance"]
    idx = bands_between(s.wavelengths, 833, 989)
    x_um = s.wavelengths[idx] / 1000.0
    normalised = s.cube[..., idx] / peak[..., None]
    normalised = fill_nans_along_spectrum(normalised)  # a bad band must not sink the pixel
    return polynomial_integral(x_um, normalised, degree=4)


@parameter(
    "SLOPE420_500",
    (420, 500),
    "Spectral slope 420-500 nm (reflectance per µm); >5 suggests elemental sulfur",
    "Slope",
)
def SLOPE420_500(s: Spectra) -> np.ndarray:
    return slope(s, 420, 500) * 1000.0


@parameter(
    "SLOPE1815_2530", (1815, 2530), "Spectral slope 1815-2530 nm (reflectance per µm)", "Slope"
)
def SLOPE1815_2530(s: Spectra) -> np.ndarray:
    return slope(s, 1815, 2530) * 1000.0


@parameter(
    "ISLOPE",
    (1815, 2530),
    "Inverse spectral slope 1815-2530 nm: -SLOPE1815_2530 (point-spectra name)",
    "Slope",
)
def ISLOPE(s: Spectra) -> np.ndarray:
    return -slope(s, 1815, 2530) * 1000.0


@parameter("BR800", (800, 997), "Band ratio R800 / R997", "Ratio")
def BR800(s: Spectra) -> np.ndarray:
    return band_ratio(s, 800, 997)


@parameter("BR2530", (2210, 2530), "Band ratio R2530 / R2210", "Ratio")
def BR2530(s: Spectra) -> np.ndarray:
    return band_ratio(s, 2530, 2210)


@parameter("IRR2", (2210, 2530), "Alias of BR2530 (point-spectra name)", "Ratio")
def IRR2(s: Spectra) -> np.ndarray:
    return band_ratio(s, 2530, 2210)


@parameter("BR3500", (3390, 3500), "Band ratio R3500 / R3390", "Ratio")
def BR3500(s: Spectra) -> np.ndarray:
    return band_ratio(s, 3500, 3390)


@parameter(
    "NDVI",
    (665, 833),
    "Normalised difference vegetation index (833, 665 nm)",
    "Normalised difference",
)
def NDVI(s: Spectra) -> np.ndarray:
    return normalized_difference(s, 833, 665)


@parameter(
    "NDWI", (560, 833), "Normalised difference water index (560, 833 nm)", "Normalised difference"
)
def NDWI(s: Spectra) -> np.ndarray:
    return normalized_difference(s, 560, 833)


@parameter(
    "NDMI",
    (833, 1670),
    "Normalised difference moisture index (833, 1670 nm)",
    "Normalised difference",
)
def NDMI(s: Spectra) -> np.ndarray:
    return normalized_difference(s, 833, 1670)


# --- public API -----------------------------------------------------------------


def parameter_names() -> list[str]:
    """Every parameter, in definition order."""
    return list(PARAMETERS)


def valid_parameters(wavelengths, tolerance: float = DEFAULT_TOLERANCE_NM) -> list[str]:
    """The parameters whose wavelength range lies within ``wavelengths``
    (nm or µm), allowing ``tolerance`` nm of slack at either end."""
    values = to_nanometres(wavelengths)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    return [
        name
        for name, p in PARAMETERS.items()
        if p.bounds[0] > low - tolerance and p.bounds[1] < high + tolerance
    ]


def clamp_reflectance(cube, limit: float = 1.0) -> np.ndarray:
    """A float copy of ``cube`` with |R| > ``limit`` set to NaN (fill values,
    saturation), as the original calculator did on loading."""
    data = np.array(cube, dtype=np.float64, copy=True)
    data[np.abs(data) > limit] = np.nan
    return data


def compute(
    cube,
    wavelengths,
    names: Iterable[str] | None = None,
    *,
    progress: ProgressCallback | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Evaluate parameters on reflectance data with the wavelength axis last.

    ``names`` defaults to every parameter the wavelength range supports.
    Returns an array shaped like the input with the wavelength axis replaced
    by one band per requested parameter, in the requested order. Parameters
    another one depends on are computed as needed. ``progress(fraction,
    name)`` is called before each parameter.
    """
    s = Spectra(cube, wavelengths)
    requested = list(names) if names is not None else valid_parameters(s.wavelengths)
    unknown = [name for name in requested if name not in PARAMETERS]
    if unknown:
        raise KeyError(f"Unknown parameters: {unknown}")
    out = np.empty(s.shape + (len(requested),), dtype=dtype)
    computed: dict[str, np.ndarray] = {}

    def evaluate(name: str) -> np.ndarray:
        if name not in computed:
            for dependency in PARAMETERS[name].depends:
                evaluate(dependency)
            computed[name] = PARAMETERS[name].func(s)
        return computed[name]

    for i, name in enumerate(requested):
        if progress is not None:
            progress(i / len(requested), name)
        out[..., i] = evaluate(name)
    if progress is not None:
        progress(1.0, "done")
    return out
