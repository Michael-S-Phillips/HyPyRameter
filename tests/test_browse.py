import numpy as np
import pytest

from hypyrameter import PARAMETERS
from hypyrameter.browse import (
    BROWSE_PRODUCTS,
    browse_image,
    stretch,
    valid_browse_products,
    write_browse_products,
)

# The CRISM MTRDR browse products, as the PDS products' labels name their
# channels (CRISM Data Product SIS, §3.12.3)
CRISM = {
    "TRU": ("R600", "R530", "R440"),
    "VNA": ("R770", "R770", "R770"),
    "FEM": ("BD530_2", "SH600_2", "BDI1000VIS"),
    "FM2": ("BD530_2", "BD920_2", "BDI1000VIS"),
    "FAL": ("R2529", "R1506", "R1080"),
    "IRA": ("R1330", "R1330", "R1330"),
    "MAF": ("OLINDEX3", "LCPINDEX2", "HCPINDEX2"),
    "HYD": ("SINDEX2", "BD2100_2", "BD1900_2"),
    "PHY": ("D2300", "D2200", "BD1900r2"),
    "PFM": ("BD2355", "D2300", "BD2290"),
    "PAL": ("BD2210_2", "BD2190", "BD2165"),
    "HYS": ("MIN2250", "BD2250", "BD1900r2"),
    "ICE": ("BD1900_2", "BD1500_2", "BD1435"),
    "IC2": ("R3920", "BD1500_2", "BD1435"),
    "CHL": ("ISLOPE1", "BD3000", "IRR2"),
    "CAR": ("D2300", "BD2500_2", "BD1900_2"),
    "CR2": ("MIN2295_2480", "MIN2345_2537", "CINDEX2"),
    "TAN": ("R2529", "R1330", "R770"),
}


def test_the_crism_products_are_defined_as_the_pds_defines_them():
    for name, channels in CRISM.items():
        assert BROWSE_PRODUCTS[name] == channels, name
    # and HyPyRameter's own products are kept
    assert {"BIO", "FM3", "HEM", "HYD2", "HYD3", "PLG", "SED", "SUL"} <= set(BROWSE_PRODUCTS)


def test_every_browse_channel_is_a_parameter():
    missing = {c for chans in BROWSE_PRODUCTS.values() for c in chans} - set(PARAMETERS)
    assert not missing


def test_valid_browse_products_need_all_three_parameters():
    assert valid_browse_products(["R600", "R530", "R440", "BD530_2"]) == ["TRU"]


def test_stretch_ignores_nan_and_maps_to_0_255():
    values = np.array([0.0, 1.0, 2.0, np.nan, 3.0, 4.0])
    out = stretch(values, "linear", percent=0)
    assert out[0] == 0 and out[-1] == 255 and np.isnan(out[3])
    assert stretch(np.full(3, np.nan)).shape == (3,)


def test_browse_image_is_uint8_rgb_with_nan_black():
    params = {n: np.random.default_rng(0).random((3, 4)) for n in ("R600", "R530", "R440")}
    params["R600"][0, 0] = np.nan
    image = browse_image(params, "TRU")
    assert image.shape == (3, 4, 3) and image.dtype == np.uint8
    assert image[0, 0, 0] == 0


def test_write_browse_products_writes_pngs(tmp_path):
    pytest.importorskip("imageio")
    params = {n: np.random.default_rng(0).random((3, 4)) for n in ("R600", "R530", "R440")}
    written = write_browse_products(params, tmp_path)
    assert [p.name for p in written] == ["TRU.png"]
    assert written[0].stat().st_size > 0
