import numpy as np

from hypyrameter.io import read_envi, write_envi


def test_envi_round_trip_keeps_band_names_as_wavelengths(tmp_path):
    data = np.random.default_rng(0).random((3, 4, 2)).astype(np.float32)
    header = write_envi(
        tmp_path / "params.hdr",
        data,
        ["BD2290", "RPEAK1"],
        base_metadata={"description": "test", "map info": ["UTM"]},
        default_bands=["RPEAK1", "BD2290", "BD2290"],
    )

    cube, wavelengths, metadata = read_envi(header)

    np.testing.assert_allclose(cube, data)
    assert wavelengths is None  # parameter names are not wavelengths
    assert metadata["band names"] == ["BD2290", "RPEAK1"]
    assert metadata["wavelength units"] == "parameters"
    assert metadata["default bands"] == ["2", "1", "1"]


def test_read_envi_converts_micrometre_wavelengths(tmp_path):
    import spectral.io.envi as envi

    data = np.zeros((2, 2, 3), dtype=np.float32)
    envi.save_image(
        str(tmp_path / "cube.hdr"), data, metadata={"wavelength": ["0.4", "0.5", "0.6"]}
    )
    _, wavelengths, _ = read_envi(tmp_path / "cube.img")
    np.testing.assert_allclose(wavelengths, [400.0, 500.0, 600.0])
