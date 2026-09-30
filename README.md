# HyPyRameter

Spectral parameters (band depths, band areas, slopes, ratios, peaks and
indices) for hyperspectral reflectance data: the CRISM parameter set of
Viviano-Beck et al. (2014) plus additions for terrestrial and laboratory
spectra. Works on image cubes, tables of spectra or a single spectrum.

[![DOI](https://zenodo.org/badge/567958968.svg)](https://zenodo.org/doi/10.5281/zenodo.10801541)

## Installation

```bash
pip install hypyrameter            # or:  uv add hypyrameter
pip install "hypyrameter[browse]"  # + PNG browse products (imageio)
pip install "hypyrameter[denoise]" # + the IOVF denoiser
```

Requires Python 3.10 or newer. The only hard dependencies are numpy, scipy,
pandas and [spectral](https://www.spectralpython.net/) (for ENVI files).

## Usage

### Core API

```python
import numpy as np
from hypyrameter import compute, valid_parameters, PARAMETERS
from hypyrameter.io import read_envi, write_envi

cube, wavelengths, metadata = read_envi("scene.hdr")   # (rows, cols, bands), nm

names = valid_parameters(wavelengths)      # every parameter the range supports
params = compute(cube, wavelengths, names, progress=lambda f, n: print(f"{f:4.0%} {n}"))
write_envi("scene_params.hdr", params, names, base_metadata=metadata)

# or just a few, on any array with the wavelength axis last:
bd = compute(cube, wavelengths, ["BD2290", "BD1900_2", "RPEAK1"])
one = compute(cube[10, 20], wavelengths, ["BD2290"])   # a single spectrum
```

* `wavelengths` may be in nm or µm (values below 10 are taken as µm).
* Missing data is NaN in and NaN out. Fill bad bands first with
  `hypyrameter.fill_nans_along_spectrum` if you want a parameter whose
  window straddles them.
* `PARAMETERS` maps each name to its wavelength range, description and group;
  `valid_parameters(wavelengths, tolerance=5)` filters by range.
* Parameters that depend on another (`BDI1000VIS` needs `RPEAK1`) compute
  what they need automatically.

### Browse products

```python
from hypyrameter.browse import BROWSE_PRODUCTS, browse_image, write_browse_products

layers = {name: params[..., i] for i, name in enumerate(names)}
rgb = browse_image(layers, "PHY")                   # (rows, cols, 3) uint8
write_browse_products(layers, "browse/")            # a PNG per valid product
```

### The 0.2.x classes

`cubeParamCalculator` and `pointParamCalculator` still exist as thin wrappers
over the core and behave as before, minus the interactive prompts when you
give them their inputs:

```python
from hypyrameter.paramCalculator import cubeParamCalculator, pointParamCalculator

pc = cubeParamCalculator(in_file="scene.hdr", outdir="out/")   # dialogs if omitted
pc.run()                       # writes out/scene_params.hdr and browse PNGs

ppc = pointParamCalculator(data_path="spectra.csv")   # or df=..., or a .sed glob
table = ppc.run()              # DataFrame: parameters (rows) x spectra (columns)
```

Both accept `parameters=[...]` to compute a subset. See the notebooks in
`hypyrameter/` for worked examples.

## Parameters

| Parameter | Range (nm) | Description |
|---|---|---|
| `R463` | 463–463 | Reflectance at 463 nm (5-band median) |
| `R550` | 550–550 | Reflectance at 550 nm (5-band median) |
| `R637` | 637–637 | Reflectance at 637 nm (5-band median) |
| `R1080` | 1080–1080 | Reflectance at 1080 nm (5-band median) |
| `R1506` | 1506–1506 | Reflectance at 1506 nm (5-band median) |
| `R2529` | 2529–2529 | Reflectance at 2529 nm (5-band median) |
| `HCPINDEX2` | 1690–2530 | High-calcium pyroxene index: weighted continuum-relative depths between 2120 and 2460 nm (continuum 1690-2530 nm) |
| `LCPINDEX2` | 1560–2450 | Low-calcium pyroxene index: weighted continuum-relative depths between 1690 and 1870 nm (continuum 1560-2450 nm) |
| `OLINDEX3` | 1210–1862 | Olivine index: weighted continuum-relative depths between 1210 and 1330 nm (continuum 1750-1862 nm) |
| `ESINDEX` | 420–950 | Elemental sulfur index: steep 420-500 nm slope minus |550-950 nm slope| (per µm) |
| `SINDEX2` | 2120–2400 | Sulfate index: inverted band depth at 2290 nm (2120-2400 nm) |
| `GINDEX` | 1440–1570 | Gypsum index: minimum of the 1447, 1491 and 1540 nm band depths |
| `CPLINDEX` | 600–710 | Chlorophyll index: minimum of the 626 and 678 nm band depths |
| `CHLORINDEX` | 1800–2450 | Chlorite index: weighted 2347, 2254, 2000 and 1921 nm depths and 2105 nm shoulder, minus BD2210_2 |
| `BD530_2` | 440–614 | 530 nm band depth (ferric oxides) |
| `BD670` | 620–745 | 670 nm band depth (custom; hematite/chlorophyll) |
| `BD875` | 747–980 | 875 nm band depth (custom; ferric minerals) |
| `BD905` | 750–1300 | 905 nm band depth |
| `BD920_2` | 807–984 | 920 nm band depth (ferric minerals) |
| `BD1200` | 1115–1260 | 1200 nm band depth |
| `BD1300` | 1080–1750 | 1300 nm band depth (plagioclase) |
| `BD1400` | 1330–1467 | 1400 nm band depth (hydration) |
| `BD1450` | 1340–1535 | 1450 nm band depth (hydration) |
| `BD1750` | 1688–1820 | 1750 nm band depth (gypsum) |
| `BD1900_2` | 1850–2067 | 1930 nm band depth (H2O) |
| `BD2100_2` | 1930–2250 | 2130 nm band depth (monohydrated sulfates) |
| `BD2100_3` | 2016–2220 | 2100 nm band depth |
| `BD2165` | 2120–2230 | 2165 nm band depth (kaolinite group) |
| `BD2190` | 2120–2250 | 2190 nm band depth (beidellite, allophane) |
| `BD2210_2` | 2165–2290 | 2210 nm band depth (Al-OH phyllosilicates) |
| `BD2250` | 2120–2340 | 2250 nm band depth |
| `BD2265` | 2120–2340 | 2265 nm band depth (jarosite) |
| `BD2290` | 2250–2350 | 2290 nm band depth (Fe/Mg phyllosilicates) |
| `BD2443` | 2320–2480 | 2443 nm band depth (nitrate) |
| `BD2355` | 2300–2450 | 2355 nm band depth (Fe/Mg phyllosilicates) |
| `BD2600` | 2530–2630 | 2600 nm band depth (H2O) |
| `BD1900r2` | 1815–2132 | 1900 nm band depth, ratio form: mean continuum-relative reflectance at 1908-1941 nm over that at 1862-1875 and 2112-2126 nm (continuum 1815-2132 nm) |
| `BDCARB` | 2230–2600 | Carbonate band depth: geometric mean of the 2330 and 2530 nm band depths |
| `BA1200` | 1115–1260 | 1200 nm band area |
| `BA1450` | 1340–1535 | 1450 nm band area |
| `BA1900` | 1850–2067 | 1900 nm band area |
| `SH460` | 420–520 | 460 nm shoulder height (inverted band depth) |
| `D460` | 420–520 | Alias of SH460 (point-spectra name) |
| `D700` | 630–830 | 700 nm drop-off (chlorophyll red edge): continuum-relative 690-720 nm over 740-770 nm (continuum 630-830 nm) |
| `D2200` | 1815–2430 | 2200 nm drop-off: continuum-relative 2210 and 2230 nm over 2165 nm (continuum 1815-2430 nm) |
| `D2300` | 1815–2530 | 2300 nm drop-off: continuum-relative 2290-2330 nm over 2120-2210 nm (continuum 1815-2530 nm) |
| `MIN2295_2480` | 2165–2570 | Minimum of the 2295 and 2480 nm band depths (Mg carbonate) |
| `MIN2250` | 2165–2350 | Minimum of the 2210 and 2265 nm band depths (opal) |
| `MIN2345_2537` | 2250–2602 | Minimum of the 2345 and 2537 nm band depths (Ca/Fe carbonate) |
| `BH1500` | 1250–1750 | 1500 nm band height (inverted band depth) |
| `RPEAK1` | 442–989 | Wavelength (µm) of the reflectance peak of a 5th-order polynomial through 13 anchors between 442 and 989 nm |
| `RPEAK1_2` | 500–1150 | Wavelength (µm) of the reflectance peak between 500 and 1150 nm after Savitzky-Golay smoothing |
| `BDI1000VIS` | 833–989 | Integrated 1000 nm band depth (VIS): integral of a 4th-order polynomial through the 833-989 nm reflectance normalised by the RPEAK1 peak |
| `SLOPE420_500` | 420–500 | Spectral slope 420-500 nm (reflectance per µm); >5 suggests elemental sulfur |
| `SLOPE1815_2530` | 1815–2530 | Spectral slope 1815-2530 nm (reflectance per µm) |
| `ISLOPE` | 1815–2530 | Inverse spectral slope 1815-2530 nm: -SLOPE1815_2530 (point-spectra name) |
| `BR800` | 800–997 | Band ratio R800 / R997 |
| `BR2530` | 2210–2530 | Band ratio R2530 / R2210 |
| `IRR2` | 2210–2530 | Alias of BR2530 (point-spectra name) |
| `BR3500` | 3390–3500 | Band ratio R3500 / R3390 |
| `NDVI` | 665–833 | Normalised difference vegetation index (833, 665 nm) |
| `NDWI` | 560–833 | Normalised difference water index (560, 833 nm) |
| `NDMI` | 833–1670 | Normalised difference moisture index (833, 1670 nm) |

## Denoising

`hypyrameter.iovf_generic.iovf` is the iterative outlier voting filter of
[Phillips et al. (2023)](https://www.sciencedirect.com/science/article/pii/S0019103523002890);
it is slow and optional (`pip install "hypyrameter[denoise]"`).
`ngMeet_denoiser.m` is a MATLAB script for the NGMeet algorithm.

## Development

```bash
git clone git@github.com:Michael-S-Phillips/HyPyRameter.git
cd HyPyRameter
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format .
```

Releases: bump `version` in `pyproject.toml`, add a CHANGELOG entry, tag
`vX.Y.Z` and push the tag; CI tests, builds and publishes to PyPI.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the
associated Planetary Science Journal publication (Phillips et al.).
