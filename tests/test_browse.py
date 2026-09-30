import numpy as np
import pytest

from hypyrameter.browse import (
    BROWSE_PRODUCTS,
    browse_image,
    stretch,
    valid_browse_products,
    write_browse_products,
)


def test_browse_definitions_load_from_the_package():
    assert BROWSE_PRODUCTS["TRU"] == ("R637", "R550", "R463")
    assert len(BROWSE_PRODUCTS) == 18


def test_valid_browse_products_need_all_three_parameters():
    assert valid_browse_products(["R637", "R550", "R463", "BD530_2"]) == ["TRU"]


def test_stretch_ignores_nan_and_maps_to_0_255():
    values = np.array([0.0, 1.0, 2.0, np.nan, 3.0, 4.0])
    out = stretch(values, "linear", percent=0)
    assert out[0] == 0 and out[-1] == 255 and np.isnan(out[3])
    assert stretch(np.full(3, np.nan)).shape == (3,)


def test_browse_image_is_uint8_rgb_with_nan_black():
    params = {n: np.random.default_rng(0).random((3, 4)) for n in ("R637", "R550", "R463")}
    params["R637"][0, 0] = np.nan
    image = browse_image(params, "TRU")
    assert image.shape == (3, 4, 3) and image.dtype == np.uint8
    assert image[0, 0, 0] == 0


def test_write_browse_products_writes_pngs(tmp_path):
    pytest.importorskip("imageio")
    params = {n: np.random.default_rng(0).random((3, 4)) for n in ("R637", "R550", "R463")}
    written = write_browse_products(params, tmp_path)
    assert [p.name for p in written] == ["TRU.png"]
    assert written[0].stat().st_size > 0
