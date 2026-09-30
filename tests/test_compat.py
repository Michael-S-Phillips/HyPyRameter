"""The 0.2.x entry points still work, now as thin wrappers over the core."""

import numpy as np
import pandas as pd
import pytest
import spectral.io.envi as envi
from conftest import gaussian_absorption

from hypyrameter import compute, valid_parameters
from hypyrameter.paramCalculator import cubeParamCalculator, pointParamCalculator


@pytest.fixture
def cube_file(tmp_path, wavelengths):
    data = np.empty((3, 4, wavelengths.size), dtype=np.float32)
    data[..., :] = 0.4 * gaussian_absorption(wavelengths, 2290.0, 0.1, 20.0)
    header = tmp_path / "cube.hdr"
    envi.save_image(
        str(header),
        data,
        metadata={"wavelength": [str(w) for w in wavelengths], "map info": ["UTM"]},
    )
    return header


def test_cube_calculator_reads_a_file_without_dialogs_and_writes_params(cube_file, wavelengths):
    outdir = cube_file.parent / "out"
    calculator = cubeParamCalculator(in_file=str(cube_file), outdir=str(outdir))

    assert calculator.validParams == valid_parameters(wavelengths)
    calculator.run()

    params = envi.open(str(outdir / "cube_params.hdr"))
    assert params.metadata["band names"] == calculator.validParams
    assert (outdir / "TRU.png").exists()
    np.testing.assert_allclose(
        np.asarray(params.load()),
        compute(calculator.cube, wavelengths, calculator.validParams),
        rtol=1e-5,
    )


def test_cube_calculator_accepts_bad_bands_crop_and_a_parameter_subset(cube_file, wavelengths):
    bbl = [1] * wavelengths.size
    bbl[10] = 0
    calculator = cubeParamCalculator(
        in_file=str(cube_file),
        outdir=str(cube_file.parent),
        bbl=bbl,
        crop=[0, 2, 1, 3],
        parameters=["BD2290", "R550"],
    )
    params = calculator.calculateParams()
    assert params.shape == (2, 2, 2)
    assert np.isnan(calculator.cube[..., 10]).all()
    assert calculator.validParams == ["BD2290", "R550"]


def test_cube_calculator_without_a_file_needs_tkinter_only_then(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tkinter", None)
    with pytest.raises(ImportError):
        cubeParamCalculator()


def test_point_calculator_returns_parameters_by_spectrum(wavelengths):
    df = pd.DataFrame(
        {
            "Wavelength": wavelengths,
            "deep": 0.4 * gaussian_absorption(wavelengths, 2290.0, 0.2, 20.0),
            "flat": np.full(wavelengths.size, 0.4),
        }
    )
    calculator = pointParamCalculator(df=df)

    result = calculator.run()

    assert list(result.columns) == ["deep", "flat"]
    assert list(result.index) == calculator.validParams
    assert "ISLOPE" in result.index and "IRR2" in result.index and "D460" in result.index
    assert result.loc["BD2290", "deep"] > 0.1 and abs(result.loc["BD2290", "flat"]) < 1e-6
    assert result.loc["BR2530", "flat"] == pytest.approx(1.0)


def test_point_calculator_reads_a_csv(tmp_path, wavelengths):
    path = tmp_path / "spectra.csv"
    pd.DataFrame({"Wavelength": wavelengths, "a": np.full(wavelengths.size, 0.3)}).to_csv(
        path, index=False
    )
    result = pointParamCalculator(data_path=str(path)).run()
    assert result.shape[1] == 1 and "BD2290" in result.index


def test_point_calculator_accepts_micrometre_wavelengths(wavelengths):
    df = pd.DataFrame({"Wavelength": wavelengths / 1000.0, "a": np.full(wavelengths.size, 0.3)})
    assert "BD2290" in pointParamCalculator(df=df).validParams
