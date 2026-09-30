"""ENVI reading and writing (via the ``spectral`` package)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spectral.io.envi as envi

from hypyrameter.bands import to_nanometres


def read_envi(header: str | Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """(cube (rows, cols, bands) float32, wavelengths in nm, header metadata).
    Wavelengths are None when the header has none or they are not numeric
    (a parameter cube names its bands instead)."""
    header = Path(header)
    if header.suffix.lower() != ".hdr":
        header = header.with_suffix(".hdr")
    image = envi.open(str(header))
    cube = np.asarray(image.load(), dtype=np.float32)
    metadata = dict(image.metadata)
    return cube, _wavelengths(metadata), metadata


def _wavelengths(metadata: dict) -> np.ndarray | None:
    try:
        return to_nanometres([float(w) for w in metadata["wavelength"]])
    except (KeyError, TypeError, ValueError):
        return None


def write_envi(
    header: str | Path,
    data: np.ndarray,
    band_names: list[str],
    *,
    base_metadata: dict | None = None,
    wavelength_units: str = "parameters",
    default_bands: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Write ``data`` (rows, cols, bands) as an ENVI image whose bands are
    named ``band_names`` (used as the 'wavelength' too, so viewers that label
    bands by wavelength show the parameter names). Georeferencing and other
    items are copied from ``base_metadata``."""
    header = Path(header).with_suffix(".hdr")
    metadata = dict(base_metadata or {})
    for key in ("wavelength", "band names", "fwhm", "bbl", "default bands", "data ignore value"):
        metadata.pop(key, None)
    metadata["wavelength"] = list(band_names)
    metadata["band names"] = list(band_names)
    metadata["wavelength units"] = wavelength_units
    if default_bands:
        metadata["default bands"] = [
            str(band_names.index(b) + 1) for b in default_bands if b in band_names
        ]
    envi.save_image(
        str(header),
        np.asarray(data, dtype=np.float32),
        metadata=metadata,
        dtype=np.float32,
        force=force,
    )
    return header
