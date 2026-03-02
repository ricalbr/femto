from __future__ import annotations

import numpy as np
import numpy.testing
import pytest
from femto.rasterimage import RasterImage
from PIL import Image


@pytest.fixture
def param() -> dict:
    p = {
        'scan': 6,
        'speed': 20.0,
        'y_init': 1.5,
        'z_init': 0.035,
        'lsafe': 4.3,
        'speed_closed': 75,
        'speed_pos': 0.1,
        'samplesize': (100, 15),
        'px_to_mm': 0.02,
    }
    return p


@pytest.fixture
def rimg(param) -> RasterImage:
    return RasterImage(**param)


def test_id(param) -> None:
    ri = RasterImage(**param)
    assert ri.id == 'RI'


def test_default_values() -> None:
    rimg = RasterImage()
    assert rimg.scan == int(1)
    assert rimg.speed == float(1.0)
    assert rimg.x_init == float(-2.0)
    assert rimg.y_init == float(0.0)
    assert rimg.z_init == float(0.0)
    assert rimg.lsafe == float(2.0)
    assert rimg.speed_closed == float(5.0)
    assert rimg.speed_pos == float(0.5)
    assert rimg.cmd_rate_max == int(1200)
    assert rimg.acc_max == int(500)
    assert rimg.samplesize == (100, 50)
    assert rimg.px_to_mm == float(0.01)
    assert rimg.img_size == (0, 0)


def test_rimg_values(param) -> None:
    rimg = RasterImage(**param)
    assert rimg.scan == int(6)
    assert rimg.speed == float(20.0)
    assert rimg.x_init == float(-2.0)
    assert rimg.y_init == float(1.5)
    assert rimg.z_init == float(0.035)
    assert rimg.lsafe == float(4.3)
    assert rimg.speed_closed == float(75)
    assert rimg.speed_pos == float(0.1)
    assert rimg.cmd_rate_max == int(1200)
    assert rimg.acc_max == int(500)
    assert rimg.samplesize == (100, 15)
    assert rimg.px_to_mm == float(0.02)
    assert rimg.img_size == (0, 0)


def test_path_size(param) -> None:
    rimg = RasterImage(**param)
    with pytest.raises(ValueError):
        print(rimg.path_size)
    del rimg

    im = Image.new(mode='RGB', size=(200, 200))
    rimg = RasterImage(**param)
    rimg.image_to_path(im)
    w, h = rimg.path_size
    assert w == 200 * rimg.px_to_mm
    assert h == 200 * rimg.px_to_mm


def test_image_to_path_color(rimg) -> None:
    from PIL import Image, ImageDraw

    im = Image.new('RGB', (16, 16))
    d = ImageDraw.Draw(im)
    d.rectangle([(0, 0), (16, 16)], fill='#ffffff')
    d.rectangle([(0, 5), (16, 10)], fill='#000033')

    x_len = 16 * rimg.px_to_mm
    y_scan = np.linspace(0, 16 * rimg.px_to_mm, 16, endpoint=True)
    y_scan = y_scan[5:11]

    x_arr = np.array([])
    y_arr = np.array([])
    z_arr = np.array([])
    f_arr = np.array([])
    s_arr = np.array([])

    for y_val in y_scan:
        xp = np.array([0, 0, x_len, x_len, 0])
        yp = y_val * np.ones_like(xp)
        zp = rimg.z_init * np.ones_like(xp)
        fp = np.array([rimg.speed_closed, rimg.speed, rimg.speed, rimg.speed_closed, rimg.speed_closed])
        sp = np.array([0, 1, 1, 0, 0])

        x_arr = np.concatenate((x_arr, xp))
        y_arr = np.concatenate((y_arr, yp))
        z_arr = np.concatenate((z_arr, zp))
        f_arr = np.concatenate((f_arr, fp))
        s_arr = np.concatenate((s_arr, sp))

    rimg.image_to_path(im)
    x, y, z, f, s = rimg.points
    numpy.testing.assert_almost_equal(x, x_arr)
    numpy.testing.assert_almost_equal(y, y_arr)
    numpy.testing.assert_almost_equal(z, z_arr)
    numpy.testing.assert_almost_equal(f, f_arr)
    numpy.testing.assert_almost_equal(s, s_arr)


def test_image_to_path_bw(rimg) -> None:
    from PIL import Image, ImageDraw

    im = Image.new('RGB', (16, 16))
    d = ImageDraw.Draw(im)
    d.rectangle([(0, 0), (16, 16)], fill='#ffffff')
    d.rectangle([(0, 5), (16, 10)], fill='#000000')

    x_len = 16 * rimg.px_to_mm
    y_scan = np.linspace(0, 16 * rimg.px_to_mm, 16, endpoint=True)
    y_scan = y_scan[5:11]

    x_arr = np.array([])
    y_arr = np.array([])
    z_arr = np.array([])
    f_arr = np.array([])
    s_arr = np.array([])

    for y_val in y_scan:
        xp = np.array([0, 0, x_len, x_len, 0])
        yp = y_val * np.ones_like(xp)
        zp = rimg.z_init * np.ones_like(xp)
        fp = np.array([rimg.speed_closed, rimg.speed, rimg.speed, rimg.speed_closed, rimg.speed_closed])
        sp = np.array([0, 1, 1, 0, 0])

        x_arr = np.concatenate((x_arr, xp))
        y_arr = np.concatenate((y_arr, yp))
        z_arr = np.concatenate((z_arr, zp))
        f_arr = np.concatenate((f_arr, fp))
        s_arr = np.concatenate((s_arr, sp))

    rimg.image_to_path(im)
    x, y, z, f, s = rimg.points
    numpy.testing.assert_almost_equal(x, x_arr)
    numpy.testing.assert_almost_equal(y, y_arr)
    numpy.testing.assert_almost_equal(z, z_arr)
    numpy.testing.assert_almost_equal(f, f_arr)
    numpy.testing.assert_almost_equal(s, s_arr)
