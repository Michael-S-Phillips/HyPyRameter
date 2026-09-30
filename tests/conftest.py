import numpy as np
import pytest


def gaussian_absorption(wavelengths, centre, depth, width):
    return 1.0 - depth * np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)


@pytest.fixture
def wavelengths():
    """A CRISM/EMIT-like wavelength axis, 400-2600 nm every 7 nm."""
    return np.arange(400.0, 2601.0, 7.0)


@pytest.fixture
def flat_cube(wavelengths):
    """A 4x5 image of a flat 0.3 reflectance spectrum."""
    return np.full((4, 5, wavelengths.size), 0.3)
