# Changelog

## 0.4.0 — 2026-10-23

### Added
- The CRISM summary parameters the browse products need, as the CRISM Data
  Product SIS (Table 3-12) and Viviano-Beck et al. (2014) define them:
  `SH600_2` (600 nm shoulder height), `BD1435` (CO2 ice), `BD1500_2` (H2O
  ice), `BD2500_2` (Mg-carbonate overtone), `BD3000` (3 µm H2O band,
  allowing for the slope), `CINDEX2` (3.6 µm carbonate convexity) and
  `ISLOPE1` (the CRISM name of `ISLOPE`), with the SIS's kernel widths.
  Against the PDS browse products of the CRISM MTRDR FRT000047A3 they are
  0.94–1.01 × CRISM's values (r = 0.90–0.99).
- Reflectances `R440`, `R530`, `R600`, `R770`, `R1330` (11-band median, as
  the SIS's IRA) and `R3920`.
- The browse products VNA, FEM, IRA, ICE, IC2, CHL, CAR and TAN, so all 18
  CRISM MTRDR browse products are defined.

### Changed
- **The CRISM browse products are defined as the PDS defines them** (the
  channels the MTRDR browse products' labels name): `TRU` is R600, R530,
  R440 (was R637, R550, R463); `PHY` is D2300, D2200, BD1900r2 (red and
  green were swapped); `CR2`'s blue is CINDEX2 (was BDCARB). HyPyRameter's
  own products (BIO, FM3, HEM, HYD2, HYD3, PLG, SED, SUL) are unchanged.

## 0.3.1 — 2026-10-23

### Fixed
- **`BDI1000VIS` was not the integrated band depth.** It integrated the
  reflectance normalised by the RPEAK1 peak, R/RPEAK1, instead of the band
  depth 1 − R/RPEAK1 (Viviano-Beck et al. 2014), so it ran opposite to
  CRISM's: high where there is no 1 µm absorption. It also stopped at 989 nm
  instead of 1023 nm. It is now the integral (µm) of 1 − R/RPEAK1 over
  833–1023 nm: 0 for a spectrum without a 1 µm band, positive with one.
  Against the CRISM MTRDR browse product of FRT000047A3 (its FM2 product's
  blue channel), the new values are 0.990 × CRISM's − 0.0011 (r = 0.987,
  median difference 0.0013); the old ones were anti-correlated (r = −0.97).
  **Values of `BDI1000VIS` computed with 0.3.0 or earlier are wrong**,
  including the FM2 browse product built from it; compute them again. The
  error predates 0.3.0 (the original code integrated R/RPEAK1 too).

## 0.3.0 — 2026-09-30

A rewrite around a pure, array-based core. The old classes still work.

### Added
- `hypyrameter.compute(cube, wavelengths, names, progress=None)`: evaluate any
  parameters on an image cube, a table of spectra or a single spectrum
  (wavelength axis last, nm or µm). `valid_parameters(wavelengths)` and the
  `PARAMETERS` registry (name, wavelength range, description, group,
  dependencies) replace the old introspection of class methods.
- `hypyrameter.bands`: `band`, `closest_band`, `closest_wavelength`,
  `bands_between`, `to_nanometres`, `apply_bad_bands`,
  `fill_nans_along_spectrum` (vectorised; replaces `interpNans`).
- `hypyrameter.fits`: vectorised polynomial peak / integral and spline area.
- `hypyrameter.browse`: browse-product definitions from a bundled CSV,
  `browse_image`, `write_browse_products` (PNG via imageio, `[browse]` extra).
- `hypyrameter.io`: `read_envi`, `write_envi`.
- `parameters=[...]` on both calculator classes to compute a subset.
- Point-spectra names `D460`, `ISLOPE`, `IRR2` are now parameters for cubes too.
- Packaging with `pyproject.toml`; published on PyPI; CI on Python 3.10–3.13.

### Fixed
- The band median window: `kwidth=5` used to take the four bands
  `[centre-2, centre+1]` and wrapped around to the end of the spectrum near
  band 0. It is now the symmetric `centre ± 2`, clamped at the ends.
- NaN handling: parameters no longer replace NaN and ±inf with the image
  minimum; missing data stays NaN.
- `BDI1000VIS` declares its dependency on `RPEAK1` instead of relying on call
  order.
- `RPEAK1_2` and the point-spectrum `BDI1000VIS` now use the same definitions
  as the cube versions (`BDI1000VIS`: 833–989 nm, 4th-order polynomial).

### Changed
- `RPEAK1_2` smooths with a Savitzky–Golay filter and fits a polynomial
  instead of a per-pixel smoothing spline (`UnivariateSpline(k=5, s=0.1)`).
  Peak positions agree to within a few nm on smooth spectra but are not
  bit-identical.
- Cube parameters are computed in one vectorised pass per parameter; no
  `multiprocessing` pools or `tqdm` bars. Progress goes through `logging`
  and the `progress` callback.
- `cubeParamCalculator` only opens tkinter dialogs when `in_file` / `outdir`
  are omitted; `denoise=True` now raises and points at `iovf_generic`.
- Optional dependencies moved to extras: `[browse]` (imageio),
  `[denoise]` (outlier-utils, tqdm, matplotlib), `[notebooks]`. OpenCV and
  openpyxl are no longer needed.
- `hypyrameter.utils` cube helpers (`getBand`, `getBandDepth`, …) delegate to
  the new core; the `.sed`/USGS readers and `getRvalue*` helpers are unchanged.

### Removed
- `interpNans.py`, `setup.py`, `meta.yaml`, `environment.yml`, the conda
  recipe files and `bin/browseDefinitions.xlsx` (now `data/browse_products.csv`).
