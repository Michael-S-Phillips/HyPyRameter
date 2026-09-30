"""HyPyRameter: spectral parameters for hyperspectral reflectance data."""

from hypyrameter.bands import (
    apply_bad_bands,
    band,
    closest_band,
    closest_wavelength,
    fill_nans_along_spectrum,
    to_nanometres,
)
from hypyrameter.parameters import (
    PARAMETERS,
    Parameter,
    Spectra,
    clamp_reflectance,
    compute,
    parameter_names,
    valid_parameters,
)

__version__ = "0.3.0"

__all__ = [
    "PARAMETERS",
    "Parameter",
    "Spectra",
    "__version__",
    "apply_bad_bands",
    "band",
    "clamp_reflectance",
    "closest_band",
    "closest_wavelength",
    "compute",
    "fill_nans_along_spectrum",
    "parameter_names",
    "to_nanometres",
    "valid_parameters",
]
