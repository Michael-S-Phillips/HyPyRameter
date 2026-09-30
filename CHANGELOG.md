# Changelog

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
