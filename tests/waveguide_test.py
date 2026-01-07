from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from femto.curves import abv
from femto.curves import arc
from femto.curves import arctan
from femto.curves import circ
from femto.curves import erf
from femto.curves import euler_S2
from femto.curves import euler_S4
from femto.curves import rad
from femto.curves import sin
from femto.curves import spline
from femto.curves import spline_bridge
from femto.waveguide import coupler
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide


@pytest.fixture
def param() -> dict:
    p = {
        'scan': 6,
        'speed': 20.0,
        'y_init': 1.5,
        'z_init': 0.050,
        'speed_closed': 75,
        'samplesize': (100, 15),
        'depth': 0.035,
        'radius': 25,
        'pitch': 0.127,
        'int_dist': 0.005,
        'int_length': 0.0,
        'arm_length': 1.0,
        'ltrench': 1.5,
        'dz_bridge': 0.006,
    }
    return p


@pytest.fixture
def empty_wg(param) -> Waveguide:
    return Waveguide(**param)


def test_default_values() -> None:
    wg = Waveguide()
    assert wg.scan == int(1)
    assert wg.speed == float(1.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(0.0)
    assert wg.z_init == float(0.035)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(5.0)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (100, 50)
    assert wg.depth == float(0.035)
    assert wg.radius == float(15)
    assert wg.pitch == float(0.080)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist is None
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(0.0)
    assert wg.ltrench == float(0.0)
    assert wg.dz_bridge == float(0.007)


def test_wg_values(param) -> None:
    wg = Waveguide(**param)
    assert wg.scan == int(6)
    assert wg.speed == float(20.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(1.5)
    assert wg.z_init == float(0.050)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(75)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (100, 15)
    assert wg.depth == float(0.035)
    assert wg.radius == float(25)
    assert wg.pitch == float(0.127)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist == float(0.005)
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(1.0)
    assert wg.ltrench == float(1.5)
    assert wg.dz_bridge == float(0.006)


def test_wg_from_dict(param) -> None:
    wg = Waveguide.from_dict(param)
    assert wg.scan == int(6)
    assert wg.speed == float(20.0)
    assert wg.x_init == float(-2.0)
    assert wg.y_init == float(1.5)
    assert wg.z_init == float(0.050)
    assert wg.lsafe == float(2.0)
    assert wg.speed_closed == float(75)
    assert wg.speed_pos == float(0.5)
    assert wg.cmd_rate_max == int(1200)
    assert wg.acc_max == int(500)
    assert wg.samplesize == (100, 15)
    assert wg.depth == float(0.035)
    assert wg.radius == float(25)
    assert wg.pitch == float(0.127)
    assert wg.pitch_fa == float(0.127)
    assert wg.int_dist == float(0.005)
    assert wg.int_length == float(0.0)
    assert wg.arm_length == float(1.0)
    assert wg.ltrench == float(1.5)
    assert wg.dz_bridge == float(0.006)


def test_slots(param) -> None:
    wg = Waveguide(**param)
    with pytest.raises(AttributeError):
        # non-existing attribrute
        wg.scns = 0.00


def test_id(param) -> None:
    w = Waveguide(**param)
    assert w.id == 'WG'


def test_z_init(param) -> None:
    del param['z_init']
    param['depth'] = 0.05
    wg = Waveguide(**param)
    assert wg.z_init == float(0.05)


def test_scan(param) -> None:
    param['scan'] = 1.2
    with pytest.raises(TypeError):
        Waveguide(**param)


def test_dy_bend_int_dist_error(param) -> None:
    wg = Waveguide(**param)
    wg.int_dist = None
    with pytest.raises(ValueError):
        print(wg.dy_bend)

    param['int_dist'] = None
    wg = Waveguide(**param)
    with pytest.raises(ValueError):
        print(wg.dy_bend)


def test_dy_bend(param) -> None:
    wg = Waveguide(**param)
    assert wg.dy_bend == 0.061


@pytest.mark.parametrize(
    'r, exp',
    [
        (0, pytest.raises(ValueError)),
        (10, does_not_raise()),
        (None, pytest.raises(ValueError)),
        (16, does_not_raise()),
        (None, pytest.raises(ValueError)),
    ],
)
def test_dx_bend_radius_error(param, r, exp) -> None:
    wg = Waveguide(**param)
    wg.radius = r
    with exp:
        print(wg.dx_bend)

    param['radius'] = r
    wg = Waveguide(**param)
    with exp:
        print(wg.dx_bend)


# dy = 0.5 * (self.pitch - self.int_dist)
# dx = np.sqrt(4 * np.abs(dy1) * radius - dy1 ** 2)
@pytest.mark.parametrize(
    'r, intd, p, exp',
    [
        (1, 0.001, 0.080, np.sqrt(2 * abs(0.080 - 0.001) * 1 - (0.5 * (0.08 - 0.001)) ** 2)),
        (10, 0.005, 0.069, np.sqrt(2 * abs(0.069 - 0.005) * 10 - (0.5 * (0.069 - 0.005)) ** 2)),
        (5.555, 0.01, 0.080, np.sqrt(2 * abs(0.080 - 0.01) * 5.555 - (0.5 * (0.08 - 0.01)) ** 2)),
        (16, 0.006, 0.300, np.sqrt(2 * abs(0.300 - 0.006) * 16 - (0.5 * (0.300 - 0.006)) ** 2)),
        (0.5, 0.1, 0.80, np.sqrt(2 * abs(0.80 - 0.1) * 0.5 - (0.5 * (0.8 - 0.1)) ** 2)),
    ],
)
def test_dx_bend(param, r, intd, p, exp) -> None:
    param['radius'] = r
    param['int_dist'] = intd
    param['pitch'] = p
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_bend) == exp


def test_dx_acc_none(param) -> None:
    param['int_length'] = None
    wg = Waveguide(**param)
    with pytest.raises(TypeError):
        print(wg.dx_coupler)


@pytest.mark.parametrize(
    'r, intd, intl, p, exp',
    [
        (1, 0.001, 1, 0.080, 2 * np.sqrt(2 * abs(0.080 - 0.001) * 1 - (0.5 * (0.08 - 0.001)) ** 2) + 1),
        (10, 0.005, 2, 0.069, 2 * np.sqrt(2 * abs(0.069 - 0.005) * 10 - (0.5 * (0.069 - 0.005)) ** 2) + 2),
        (5.555, 0.01, 4, 0.080, 2 * np.sqrt(2 * abs(0.080 - 0.01) * 5.555 - (0.5 * (0.08 - 0.01)) ** 2) + 4),
        (16, 0.006, 6.6666, 0.300, 2 * np.sqrt(2 * abs(0.300 - 0.006) * 16 - (0.5 * (0.300 - 0.006)) ** 2) + 6.6666),
        (0.5, 0.1, 0.0001, 0.80, 2 * np.sqrt(2 * abs(0.80 - 0.1) * 0.5 - (0.5 * (0.8 - 0.1)) ** 2) + 0.0001),
    ],
)
def test_dx_coupler(param, r, intd, intl, p, exp) -> None:
    param['radius'] = r
    param['int_dist'] = intd
    param['int_length'] = intl
    param['pitch'] = p
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_coupler) == exp


def test_dx_mzi_none_intl(param) -> None:
    param['int_length'] = None
    wg = Waveguide(**param)
    with pytest.raises(TypeError):
        print(wg.dx_mzi)


def test_dx_mzi_none_arml(param) -> None:
    param['arm_length'] = None
    wg = Waveguide(**param)
    with pytest.raises(TypeError):
        print(wg.dx_mzi)


@pytest.mark.parametrize(
    'r, intd, intl, arml, p, exp',
    [
        (1, 0.001, 1, 55, 0.080, 4 * np.sqrt(2 * abs(0.080 - 0.001) * 1 - (0.5 * (0.08 - 0.001)) ** 2) + 1 * 2 + 55),
        (
            10,
            0.005,
            2,
            0.9,
            0.069,
            4 * np.sqrt(2 * abs(0.069 - 0.005) * 10 - (0.5 * (0.069 - 0.005)) ** 2) + 2 * 2 + 0.9,
        ),
        (
            5.555,
            0.01,
            4,
            33.3333,
            0.080,
            4 * np.sqrt(2 * abs(0.080 - 0.01) * 5.555 - (0.5 * (0.08 - 0.01)) ** 2) + 4 * 2 + 33.3333,
        ),
        (
            16,
            0.006,
            6.6666,
            0.123456,
            0.300,
            4 * np.sqrt(2 * abs(0.300 - 0.006) * 16 - (0.5 * (0.300 - 0.006)) ** 2) + 6.6666 * 2 + 0.123456,
        ),
        (
            0.5,
            0.1,
            0.0001,
            9.876543,
            0.80,
            4 * np.sqrt(2 * abs(0.80 - 0.1) * 0.5 - (0.5 * (0.8 - 0.1)) ** 2) + 0.0001 * 2 + 9.876543,
        ),
    ],
)
def test_dx_mzi(param, r, intd, intl, arml, p, exp) -> None:
    param['radius'] = r
    param['int_dist'] = intd
    param['int_length'] = intl
    param['arm_length'] = arml
    param['pitch'] = p
    wg = Waveguide(**param)
    assert pytest.approx(wg.dx_mzi) == exp


@pytest.mark.parametrize(
    'dy, r, exp',
    [
        (0.08, 0, pytest.raises(ValueError)),
        (0.08, 10, does_not_raise()),
        (0.08, None, pytest.raises(ValueError)),
        (None, 16, pytest.raises(ValueError)),
        (None, None, pytest.raises(ValueError)),
    ],
)
def test_get_sbend_param_error(dy, r, exp, param) -> None:
    wg = Waveguide(**param)
    with exp:
        wg.get_sbend_parameter(dy, r)


def test_get_sbend_param(param) -> None:
    dy = 0.08
    r = 30
    wg = Waveguide(**param)
    assert type(wg.get_sbend_parameter(dy, r)) == tuple
    assert pytest.approx(wg.get_sbend_parameter(dy, r)[0]) == 0.0516455
    assert pytest.approx(wg.get_sbend_parameter(dy, r)[1]) == 3.097354


@pytest.mark.parametrize('dy, r, exp', [(0.0, 15, 0.0), (0.127, 15, 2.757512)])
def test_get_sbend_length_nil_dy(dy, r, exp, param) -> None:
    wg = Waveguide(**param)
    assert pytest.approx(wg.get_sbend_parameter(dy, r)[1]) == exp


def test_repr(param) -> None:
    r = Waveguide(**param).__repr__()
    print()
    print(r)
    cname, _ = r.split('@')
    assert cname == 'Waveguide'


def test_arc_acc_len(param):
    dy = 0.055
    r = 32
    i_len = 1

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=wg.dy_bend, dz=0, fx=circ, radius=wg.radius)
    x, _ = wg.x, wg.y
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0, fx=circ, radius=wg.radius)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, wg.radius)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0, radius=r, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, r)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0, radius=r, int_length=i_len, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, r)[1] + i_len
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=wg.dy_bend, dz=0, int_length=-i_len, fx=circ, radius=wg.radius)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + i_len
    wg.end()


def test_arc_mzi_len(param):
    dy = 0.065
    r = 24.6
    i_len = 5
    a_len = 8

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=wg.dy_bend, dz=0, fx=circ)
    x = wg.x
    assert (
        pytest.approx(x[-1] - x[0])
        == 4 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + 2 * wg.int_length + wg.arm_length
    )
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0, fx=circ)
    x = wg.x
    assert (
        pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, wg.radius)[1] + +2 * wg.int_length + wg.arm_length
    )
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0, radius=r, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * wg.int_length + wg.arm_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0, radius=r, int_length=i_len, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * i_len + wg.arm_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0, radius=r, int_length=i_len, arm_length=a_len, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * i_len + a_len
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=wg.dy_bend, dz=0, int_length=-i_len, arm_length=-a_len, fx=circ)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + 2 * i_len + a_len
    wg.end()


def test_sin_bend_default(param):
    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(dy=wg.dy_bend, dz=0.0, fx=sin)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == wg.get_sbend_parameter(wg.dy_bend, np.abs(wg.radius))[1]
    assert pytest.approx(np.max(y) - np.min(y)) == wg.dy_bend
    assert pytest.approx(y[-1] - y[0]) == wg.dy_bend
    assert pytest.approx(np.max(z) - np.min(z)) == 0.0
    wg.end()


def test_sin_bend_values(param):
    r = 23
    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(dy=wg.dy_bend, dz=0.0, fx=sin, radius=r)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == wg.get_sbend_parameter(wg.dy_bend, np.abs(r))[1]
    assert pytest.approx(np.max(y) - np.min(y)) == wg.dy_bend
    assert pytest.approx(y[-1] - y[0]) == wg.dy_bend
    assert pytest.approx(np.max(z) - np.min(z)) == 0.0
    wg.end()

    r = 17
    dy = 0.9876
    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(dy=dy, dz=0.0, fx=sin, radius=r)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == wg.get_sbend_parameter(dy, np.abs(r))[1]
    assert pytest.approx(np.max(y) - np.min(y)) == dy
    assert pytest.approx(y[-1] - y[0]) == dy
    assert pytest.approx(np.max(z) - np.min(z)) == 0.0
    wg.end()


def test_sin_acc_len(param):
    dy = 0.415
    r = 12.04
    i_len = 1

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=wg.dy_bend, dz=0.0, fx=sin)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0.0, fx=sin)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, wg.radius)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0.0, fx=sin, radius=r)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, r)[1] + wg.int_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=dy, dz=0.0, fx=sin, radius=r, int_length=i_len)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(dy, r)[1] + i_len
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).coupler(dy=wg.dy_bend, dz=0.0, fx=sin, int_length=-i_len)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 2 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + i_len
    wg.end()


def test_sin_mzi_len(param):
    dy = 0.3335
    r = 28.12
    i_len = 5
    a_len = 8

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=wg.dy_bend, dz=0.0, fx=sin)
    x = wg.x
    assert (
        pytest.approx(x[-1] - x[0])
        == 4 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + 2 * wg.int_length + wg.arm_length
    )
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0.0, fx=sin)
    x = wg.x
    assert (
        pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, wg.radius)[1] + 2 * wg.int_length + wg.arm_length
    )
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0.0, fx=sin, radius=r)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * wg.int_length + wg.arm_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0.0, fx=sin, radius=r, int_length=i_len)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * i_len + wg.arm_length
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=dy, dz=0.0, fx=sin, radius=r, int_length=i_len, arm_length=a_len)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(dy, r)[1] + 2 * i_len + a_len
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).mzi(dy=wg.dy_bend, dz=0.0, fx=sin, int_length=-i_len, arm_length=-a_len)
    x = wg.x
    assert pytest.approx(x[-1] - x[0]) == 4 * wg.get_sbend_parameter(wg.dy_bend, wg.radius)[1] + 2 * i_len + a_len
    wg.end()


def test_spline_dy_default(param):
    dx = 5
    dz = 0.01

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=wg.dy_bend, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == wg.dy_bend
    assert pytest.approx(np.max(z) - np.min(z)) == dz
    wg.end()


def test_spline_dy_custom(param):
    dx = 3.12
    dy = 0.08
    dz = 0.15

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == dy
    assert pytest.approx(np.max(z) - np.min(z)) == dz
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=-dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == -dy
    assert pytest.approx(np.max(z) - np.min(z)) == dz
    wg.end()


def test_spline_dz_default(param):
    dx = 5
    dy = 0.01

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=wg.dz_bridge, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == dy
    assert pytest.approx(z[-1] - z[0]) == wg.dz_bridge
    wg.end()


def test_spline_dz_custom(param):
    dx = 3.12
    dy = 1.85
    dz = 1.2

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == dy
    assert pytest.approx(z[-1] - z[0]) == dz
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=-dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1] - x[0]) == dx
    assert pytest.approx(y[-1] - y[0]) == dy
    assert pytest.approx(z[-1] - z[0]) == -dz
    wg.end()


def test_spline_init_pos_default(param):
    dx = 3.12
    dy = 1.85
    dz = 1.2
    init_p = [-1, 5, -0.21]

    wg = Waveguide(**param)
    wg.start(init_p).bend(disp_x=dx, dy=dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1]) == dx + init_p[0]
    assert pytest.approx(y[-1]) == dy + init_p[1]
    assert pytest.approx(z[-1]) == dz + init_p[2]
    wg.end()


def test_spline_init_pos_custom(param):
    dx = 3.12
    dy = 1.85
    dz = 1.2
    i_pos = np.array([5.8, 9.2, 3.57])

    wg = Waveguide(**param)
    wg.x_init = 1.0
    wg.y_init = 2.0
    wg.z_init = 3.0
    wg.start(i_pos).bend(disp_x=dx, dy=dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1]) == dx + i_pos[0]
    assert pytest.approx(y[-1]) == dy + i_pos[1]
    assert pytest.approx(z[-1]) == dz + i_pos[2]
    wg.end()


def test_spline_init_none(param):
    # if wg.x_init, wg.y_init, wg.z_init are None and init_point is None attach the spline to last point

    x0, y0, z0 = (5, 6, 7)
    dx, dy, dz = (1.2, 2.3, 3.4)

    wg = Waveguide(**param)
    wg.x_init = None
    wg.y_init = None
    wg.z_init = None
    wg.start([x0, y0, z0]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline)
    x, y, z, *_ = wg.points
    assert pytest.approx(x[-1]) == dx + x0
    assert pytest.approx(y[-1]) == dy + y0
    assert pytest.approx(z[-1]) == dz + z0
    wg.end()


def test_spline_radius_default(param):
    dy, dz = (0.3, 0.69)

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=None, dy=dy, dz=dz, fx=spline)
    x = wg.x
    assert pytest.approx(x[-1]) == wg.get_sbend_parameter(np.sqrt(dy**2 + dz**2), wg.radius)[1]
    wg.end()


def test_spline_radius_custom(param):
    dy, dz = (0.3, 0.69)
    r = 90

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=None, dy=dy, dz=dz, fx=spline, radius=r)
    x = wg.x
    assert pytest.approx(x[-1]) == wg.get_sbend_parameter(np.sqrt(dy**2 + dz**2), r)[1]
    wg.end()


def test_spline_radius_none(param):
    dy, dz = (0.3, 0.69)
    r = None

    wg = Waveguide(**param)
    wg.radius = None
    with pytest.raises(ValueError):
        wg.start([0, 0, 0]).bend(disp_x=None, dy=dy, dz=dz, fx=spline, radius=r).end()


@pytest.mark.parametrize(
    'ddy', [((0.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)), ((-2.0, 0.5), (0.5, 0.6)), ((0.1, 0.1), (0.0, 0.23))]
)
def test_spline_y_derivative(param, ddy):
    _, dy, dz = (0.1, 0.23, 0.456)
    dz_der = ((0.0, 0.0), (0.0, 0.0))

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(dy=dy, dz=dz, fx=spline, y_derivatives=ddy, z_derivatives=dz_der)

    # Extract the x and y coordinates as separate arrays
    x, y, z = wg.path3d
    yp = np.gradient(y, x)
    ypp = 2 * np.gradient(yp, x)

    assert pytest.approx(yp[0], abs=1e-1) == ddy[0][0]
    assert pytest.approx(yp[-2], abs=1e-1) == ddy[-1][0]
    assert pytest.approx(ypp[0], abs=1e-1) == ddy[0][1]
    assert pytest.approx(ypp[-1], abs=1e-1) == ddy[-1][1]


@pytest.mark.parametrize(
    'ddz', [((0.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)), ((-2.0, 0.5), (0.5, 0.6)), ((0.1, 0.1), (0.0, 0.23))]
)
def test_spline_z_derivative(param, ddz):
    _, dy, dz = (0.1, 0.23, 0.456)
    dy_der = ((0.0, 0.0), (0.0, 0.0))

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(dy=dy, dz=dz, fx=spline, y_derivatives=dy_der, z_derivatives=ddz)

    # Extract the x and y coordinates as separate arrays
    x, y, z = wg.path3d
    zp = np.gradient(z, x)
    zpp = 2 * np.gradient(zp, x)

    assert pytest.approx(zp[0], abs=1e-1) == ddz[0][0]
    assert pytest.approx(zp[-2], abs=1e-1) == ddz[-1][0]
    assert pytest.approx(zpp[0], abs=1e-1) == ddz[0][1]
    assert pytest.approx(zpp[-1], abs=1e-1) == ddz[-1][1]


@pytest.mark.parametrize('r_input', [5, 10, 15, 20, 25, 30, 35, 40, 50, 60])
def test_curvature_radius(param, r_input) -> None:
    # mean curvature radius is within 1.5% of the original radius
    r = r_input
    x = 0.015
    wg = Waveguide(**param)
    wg.start().bend(2 * r, dz=0, radius=r, fx=arc)

    assert np.mean(wg.curvature_radius) <= (1 + x) * r
    assert np.mean(wg.curvature_radius) >= (1 - x) * r


def test_curvature_radius_default(param) -> None:
    # mean curvature radius is within 5% of the default radius
    x = 0.05
    wg = Waveguide(**param)
    wg.start().bend(2 * wg.radius, dz=0, fx=arc)

    assert np.mean(wg.curvature_radius) <= (1 + x) * wg.radius
    assert np.mean(wg.curvature_radius) >= (1 - x) * wg.radius


def test_cmd_rate(param) -> None:
    wg = Waveguide(**param)
    wg.start().linear([1, 2, 3], mode='abs').bend(dy=wg.dy_bend, dz=0.0, fx=sin).linear([4, 5, 6]).end()

    assert np.mean(wg.cmd_rate) <= wg.cmd_rate_max


def test_spline_bridge_speed(param) -> None:
    dy, dz, f_custom = (0.06, 0.006, 99)

    wg = Waveguide(**param)
    wg.start().bend(disp_x=None, dy=dy, dz=dz, fx=spline_bridge)
    assert wg._f[-1] == wg.speed
    wg.end()

    wg = Waveguide(**param)
    wg.start().bend(disp_x=None, dy=dy, dz=dz, fx=spline_bridge, speed=f_custom)
    assert wg._f[-1] == f_custom
    wg.end()


def test_spline_bridge_dx(param) -> None:
    dx, dy, dz = (10, 0.06, 0.006)

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=None, dy=dy, dz=dz, fx=spline_bridge)
    assert pytest.approx(wg.x[-1]) == wg.get_sbend_parameter(dy=np.sqrt(dy**2 + dz**2), radius=wg.radius)[-1]
    wg.end()

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline_bridge)
    assert pytest.approx(wg.x[-1]) == dx
    wg.end()


def test_spline_bridge_dy_dz(param) -> None:
    dx, dy, dz = (15, 0.789, 0.123)

    wg = Waveguide(**param)
    wg.start([0, 0, 0]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline_bridge)
    assert pytest.approx(wg.y[-1]) == dy
    assert pytest.approx(wg.z[-1] - wg.z[0], abs=1e-8) == 0.0
    assert pytest.approx(np.max(wg.z) - np.min(wg.z)) == dz
    wg.end()


def test_spline_bridge_derivatives(param) -> None:
    dx, dy, dz = (19.45, 12.38, 3.56)
    wg = Waveguide(**param)

    wg.start([0, 0, 0.5]).bend(disp_x=dx, dy=dy, dz=dz, fx=spline_bridge)
    x, y, z = wg.path3d
    assert pytest.approx((y[1] - y[0]) / (x[1] - x[0]), abs=1e-2) == 0.0
    assert pytest.approx((y[-2] - y[-1]) / (x[-2] - x[-1]), abs=1e-2) == 0.0
    assert pytest.approx((z[1] - z[0]) / (x[1] - x[0]), abs=1e-2) == 0.0
    assert pytest.approx((z[-2] - z[-1]) / (x[-2] - x[-1]), abs=1e-2) == 0.0


def test_nasu_default_values() -> None:
    ng = NasuWaveguide()
    assert ng.scan == int(1)
    assert ng.speed == float(1.0)
    assert ng.x_init == float(-2.0)
    assert ng.y_init == float(0.0)
    assert ng.z_init == float(0.035)
    assert ng.lsafe == float(2.0)
    assert ng.speed_closed == float(5.0)
    assert ng.speed_pos == float(0.5)
    assert ng.cmd_rate_max == int(1200)
    assert ng.acc_max == int(500)
    assert ng.samplesize == (100, 50)
    assert ng.depth == float(0.035)
    assert ng.radius == float(15)
    assert ng.pitch == float(0.080)
    assert ng.pitch_fa == float(0.127)
    assert ng.int_dist is None
    assert ng.int_length == float(0.0)
    assert ng.arm_length == float(0.0)
    # assert ng.ltrench == float(1.0)
    assert ng.dz_bridge == float(0.007)
    assert ng.adj_scan_shift == (0, 0.0004, 0)
    assert ng.adj_scan == int(5)


def test_nasu_values(param) -> None:
    ng = NasuWaveguide(adj_scan_shift=(0.1, 0.2, 0.003), adj_scan=3, **param)
    assert ng.scan == int(6)
    assert ng.speed == float(20.0)
    assert ng.x_init == float(-2.0)
    assert ng.y_init == float(1.5)
    assert ng.z_init == float(0.050)
    assert ng.lsafe == float(2.0)
    assert ng.speed_closed == float(75)
    assert ng.speed_pos == float(0.5)
    assert ng.cmd_rate_max == int(1200)
    assert ng.acc_max == int(500)
    assert ng.samplesize == (100, 15)
    assert ng.depth == float(0.035)
    assert ng.radius == float(25)
    assert ng.pitch == float(0.127)
    assert ng.pitch_fa == float(0.127)
    assert ng.int_dist == float(0.005)
    assert ng.int_length == float(0.0)
    assert ng.arm_length == float(1.0)
    assert ng.dz_bridge == float(0.006)
    assert ng.adj_scan_shift == (0.1, 0.2, 0.003)
    assert ng.adj_scan == int(3)


def test_nasu_id(param) -> None:
    nw = NasuWaveguide(**param)
    assert nw.id == 'NWG'


def test_nasu_depth(param) -> None:
    del param['z_init']
    nw = NasuWaveguide(**param)
    assert nw.z_init == nw.depth


@pytest.mark.parametrize(
    'a_scan, exp',
    [
        (3, does_not_raise()),
        (3.33, pytest.raises(TypeError)),
        (0.01, pytest.raises(TypeError)),
        (5, does_not_raise()),
        (2, does_not_raise()),
    ],
)
def test_nasu_raise(a_scan, exp, param) -> None:
    with exp:
        NasuWaveguide(adj_scan=a_scan, **param)


@pytest.mark.parametrize(
    'a_scan, exp',
    [
        (3, [0, 1, -1]),
        (5, [0, 1, -1, 2, -2]),
        (17, [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8]),
        (1, [0]),
        (4, [0.5, -0.5, 1.5, -1.5]),
        (8, [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5]),
    ],
)
def test_nasu_adjscans(param, a_scan, exp) -> None:
    ng = NasuWaveguide(adj_scan=a_scan)
    assert ng.adj_scan_order == exp


def test_coupler_pitch(param) -> None:
    mode1, mode2 = coupler(param, f_profile=sin)
    assert mode2.y[-1] == pytest.approx(mode1.y[-1] + param['pitch'])


def test_coupler_wg_type(param) -> None:
    mode1, mode2 = coupler(param, f_profile=sin, nasu=False)
    assert isinstance(mode1, type(mode2))
    assert isinstance(mode1, Waveguide)
    del mode1, mode2

    mode1, mode2 = coupler(param, f_profile=sin, nasu=True)
    assert isinstance(mode1, type(mode2))
    assert isinstance(mode1, NasuWaveguide)


@pytest.mark.parametrize('d_input', [0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.0011, 0.0015, 0.0025])
def test_coupler_d_int(d_input) -> None:
    p = {
        'scan': 6,
        'speed': 20.0,
        'samplesize': (100, 15),
        'depth': 0.035,
        'radius': 25,
        'pitch': 0.127,
        'int_dist': d_input,
        'int_length': 0.0,
        'arm_length': 1.0,
    }

    mode1, mode2 = coupler(p, f_profile=sin)

    # test difference between min/max of mode2 and mode1 (respectively) is int_dist
    assert pytest.approx(np.min(mode2.y) - np.max(mode1.y), abs=1e-6) == d_input
    # test the point of min distance is at the same x value
    assert pytest.approx(mode2.x[np.where(mode2.y == np.min(mode2.y))]) == mode1.x[np.where(mode1.y == np.max(mode1.y))]

    mode1, mode2 = coupler(p, f_profile=circ)

    # test difference between min/max of mode2 and mode1 (respectively) is int_dist
    assert pytest.approx(np.min(mode2.y) - np.max(mode1.y), abs=1e-6) == d_input
    # test the point of min distance is at the same x value
    assert pytest.approx(mode2.x[np.where(mode2.y == np.min(mode2.y))]) == mode1.x[np.where(mode1.y == np.max(mode1.y))]

    mode1, mode2 = coupler(p, f_profile=spline)

    # test difference between min/max of mode2 and mode1 (respectively) is int_dist
    assert pytest.approx(np.min(mode2.y) - np.max(mode1.y), abs=1e-6) == d_input
    # test the point of min distance is at the same x value
    assert pytest.approx(mode2.x[np.where(mode2.y == np.min(mode2.y))]) == mode1.x[np.where(mode1.y == np.max(mode1.y))]

    mode1, mode2 = coupler(p, f_profile=euler_S4)

    # test difference between min/max of mode2 and mode1 (respectively) is int_dist
    assert pytest.approx(np.min(mode2.y) - np.max(mode1.y), abs=1e-6) == d_input
    # test the point of min distance is at the same x value
    assert pytest.approx(mode2.x[np.where(mode2.y == np.min(mode2.y))]) == mode1.x[np.where(mode1.y == np.max(mode1.y))]


def test_reverse_bend(param) -> None:
    dx = 9
    dz = 0.1
    x0, y0, z0 = 0, 0, 0
    TOL = 1e-10

    # Forward curve
    wg_fwd = Waveguide(**param)
    wg_fwd.start([x0, y0, z0])
    wg_fwd.bend(dy=wg_fwd.dy_bend, dz=dz, disp_x=dx, fx=sin, reverse=False)

    # Reversed curve
    wg_rev = Waveguide(**param)
    wg_rev.start([x0, y0, z0])
    wg_rev.bend(dy=wg_rev.dy_bend, dz=dz, disp_x=dx, fx=sin, reverse=True)

    # 1) Endpoint check (global geometry)
    print(f'{x0=}')
    print(f'{wg_rev.dx_bend=}')
    print(f'{wg_rev.lastx=}')
    assert wg_rev.lastx == pytest.approx(x0 - dx, abs=TOL)
    assert wg_rev.lasty == pytest.approx(y0 + wg_rev.dy_bend, abs=TOL)
    assert wg_rev.lastz == pytest.approx(z0 - dz, abs=TOL)

    # 2) Curve shape check (local geometry) Work in coordinates relative to the start point
    xf = np.asarray(wg_fwd.x, dtype=np.float64)
    yf = np.asarray(wg_fwd.y, dtype=np.float64)
    zf = np.asarray(wg_fwd.z, dtype=np.float64)

    xr = np.asarray(wg_rev.x, dtype=np.float64)
    yr = np.asarray(wg_rev.y, dtype=np.float64)
    zr = np.asarray(wg_rev.z, dtype=np.float64)

    # local frames
    xf -= xf[0]
    yf -= yf[0]
    zf -= zf[0]

    xr -= xr[0]
    yr -= yr[0]
    zr -= zr[0]

    assert xr == pytest.approx(-xf, abs=TOL)  # X: reversed path, mirrored
    assert yr == pytest.approx(yf, abs=TOL)  # Y: reversed path, same orientation
    assert zr == pytest.approx(-zf, abs=TOL)  # Z: reversed path, mirrored


@pytest.mark.parametrize(
    'f',
    [euler_S4, sin, spline, arctan, rad, abv, euler_S2, erf],
)
def test_reverse_bend_curves(f, param) -> None:
    dx = 9.0
    dz = 0.1
    x0, y0, z0 = 3.0, 4.0, 5.0
    TOL = 1e-10

    # Forward curve
    wg_fwd = Waveguide(**param)
    wg_fwd.start([x0, y0, z0])
    wg_fwd.bend(dy=wg_fwd.dy_bend, dz=dz, disp_x=dx, fx=f, reverse=False)

    # Reversed curve
    wg_rev = Waveguide(**param)
    wg_rev.start([x0, y0, z0])
    wg_rev.bend(dy=wg_rev.dy_bend, dz=dz, disp_x=dx, fx=f, reverse=True)

    # 1) Endpoint check (global geometry)
    assert wg_rev.lastx == pytest.approx(x0 - dx, abs=TOL)
    assert wg_rev.lasty == pytest.approx(y0 + wg_rev.dy_bend, abs=TOL)
    assert wg_rev.lastz == pytest.approx(z0 - dz, abs=TOL)

    # 2) Curve shape check (local geometry) Work in coordinates relative to the start point
    xf = np.asarray(wg_fwd.x, dtype=np.float64)
    yf = np.asarray(wg_fwd.y, dtype=np.float64)
    zf = np.asarray(wg_fwd.z, dtype=np.float64)

    xr = np.asarray(wg_rev.x, dtype=np.float64)
    yr = np.asarray(wg_rev.y, dtype=np.float64)
    zr = np.asarray(wg_rev.z, dtype=np.float64)

    # local frames
    xf -= xf[0]
    yf -= yf[0]
    zf -= zf[0]

    xr -= xr[0]
    yr -= yr[0]
    zr -= zr[0]

    assert xr == pytest.approx(-xf, abs=TOL)  # X: reversed path, mirrored
    assert yr == pytest.approx(yf, abs=TOL)  # Y: reversed path, same orientation
    assert zr == pytest.approx(-zf, abs=TOL)  # Z: reversed path, mirrored


def test_reverse_bend_circ(param) -> None:
    dz = 0.1
    x0, y0, z0 = 3, 4, 5
    TOL = 1e-10

    # Forward curve
    wg_fwd = Waveguide(**param)
    wg_fwd.start([x0, y0, z0])
    wg_fwd.bend(dy=wg_fwd.dy_bend, dz=dz, fx=circ, reverse=False)

    # Reversed curve
    wg_rev = Waveguide(**param)
    wg_rev.start([x0, y0, z0])
    wg_rev.bend(dy=wg_rev.dy_bend, dz=dz, fx=circ, reverse=True)

    # 1) Endpoint check (global geometry)
    print(f'{x0=}')
    print(f'{wg_rev.dx_bend=}')
    print(f'{wg_rev.lastx=}')
    assert wg_rev.lastx == pytest.approx(x0 - wg_rev.dx_bend, abs=TOL)
    assert wg_rev.lasty == pytest.approx(y0 + wg_rev.dy_bend, abs=TOL)
    assert wg_rev.lastz == pytest.approx(z0 - dz, abs=TOL)

    # 2) Curve shape check (local geometry) Work in coordinates relative to the start point
    xf = np.asarray(wg_fwd.x, dtype=np.float64)
    yf = np.asarray(wg_fwd.y, dtype=np.float64)
    zf = np.asarray(wg_fwd.z, dtype=np.float64)

    xr = np.asarray(wg_rev.x, dtype=np.float64)
    yr = np.asarray(wg_rev.y, dtype=np.float64)
    zr = np.asarray(wg_rev.z, dtype=np.float64)

    # local frames
    xf -= xf[0]
    yf -= yf[0]
    zf -= zf[0]

    xr -= xr[0]
    yr -= yr[0]
    zr -= zr[0]

    assert xr == pytest.approx(-xf, abs=TOL)  # X: reversed path, mirrored
    assert yr == pytest.approx(yf, abs=TOL)  # Y: reversed path, same orientation
    assert zr == pytest.approx(-zf, abs=TOL)  # Z: reversed path, mirrored


def test_reverse_coupler(param) -> None:
    dz = 0.1
    x0, y0, z0 = 3, 4, 5
    lin = 5
    TOL = 1e-11

    # Forward curve
    wg_fwd = Waveguide(**param)
    wg_fwd.start([x0, y0, z0])
    wg_fwd.coupler(dy=wg_fwd.dy_bend, dz=dz, fx=circ, int_length=lin, reverse=False)

    # Reversed curve
    wg_rev = Waveguide(**param)
    wg_rev.start([x0, y0, z0])
    wg_rev.coupler(dy=wg_rev.dy_bend, dz=dz, fx=circ, int_length=lin, reverse=True)

    # 1) Endpoint check (global geometry)
    assert wg_rev.lastx == pytest.approx(x0 - 2 * wg_rev.dx_bend - lin, abs=TOL)
    assert wg_rev.lasty == pytest.approx(y0, abs=TOL)
    assert wg_rev.lastz == pytest.approx(z0, abs=TOL)

    # 2) Curve shape check (local geometry) Work in coordinates relative to the start point
    xf = np.asarray(wg_fwd._x, dtype=np.float64)
    yf = np.asarray(wg_fwd._y, dtype=np.float64)
    zf = np.asarray(wg_fwd._z, dtype=np.float64)

    xr = np.asarray(wg_rev._x, dtype=np.float64)
    yr = np.asarray(wg_rev._y, dtype=np.float64)
    zr = np.asarray(wg_rev._z, dtype=np.float64)

    # local frames
    xf -= xf[0]
    yf -= yf[0]
    zf -= zf[0]

    xr -= xr[0]
    yr -= yr[0]
    zr -= zr[0]

    assert xr == pytest.approx(-xf, abs=TOL)  # X: reversed path, mirrored
    assert yr == pytest.approx(yf, abs=TOL)  # Y: reversed path, same orientation
    assert zr == pytest.approx(-zf, abs=TOL)  # Z: reversed path, mirrored


def test_reverse_mzi(param) -> None:
    dz = 0.1
    x0, y0, z0 = 3, 4, 5
    lin = 5
    lar = 3
    TOL = 1e-12

    # Forward curve
    wg_fwd = Waveguide(**param)
    wg_fwd.start([x0, y0, z0])
    wg_fwd.mzi(dy=wg_fwd.dy_bend, dz=dz, fx=circ, int_length=lin, arm_length=lar, reverse=False)

    # Reversed curve
    wg_rev = Waveguide(**param)
    wg_rev.start([x0, y0, z0])
    wg_rev.mzi(dy=wg_rev.dy_bend, dz=dz, fx=circ, int_length=lin, arm_length=lar, reverse=True)

    # 1) Endpoint check (global geometry)
    assert wg_rev.lastx == pytest.approx(x0 - 4 * wg_rev.dx_bend - 2 * lin - lar, abs=TOL)
    assert wg_rev.lasty == pytest.approx(y0, abs=TOL)
    assert wg_rev.lastz == pytest.approx(z0, abs=TOL)

    # 2) Curve shape check (local geometry) Work in coordinates relative to the start point
    xf = np.asarray(wg_fwd._x, dtype=np.float64)
    yf = np.asarray(wg_fwd._y, dtype=np.float64)
    zf = np.asarray(wg_fwd._z, dtype=np.float64)

    xr = np.asarray(wg_rev._x, dtype=np.float64)
    yr = np.asarray(wg_rev._y, dtype=np.float64)
    zr = np.asarray(wg_rev._z, dtype=np.float64)

    # local frames
    xf -= xf[0]
    yf -= yf[0]
    zf -= zf[0]

    xr -= xr[0]
    yr -= yr[0]
    zr -= zr[0]

    assert xr == pytest.approx(-xf, abs=TOL)  # X: reversed path, mirrored
    assert yr == pytest.approx(yf, abs=TOL)  # Y: reversed path, same orientation
    assert zr == pytest.approx(-zf, abs=TOL)  # Z: reversed path, mirrored


def test_add_points(param):
    pts = np.random.rand(3, 123)
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.add_curve_points(pts, speed=None, shutter=1)
    x, y, z = wg.path3d
    np.testing.assert_array_equal(x[1:], np.array(pts[0, :], dtype=np.float64))
    np.testing.assert_array_equal(y[1:], np.array(pts[1, :], dtype=np.float64))
    np.testing.assert_array_equal(z[1:], np.array(pts[2, :], dtype=np.float64))
    del wg

    pts = np.random.rand(123, 3)
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.add_curve_points(pts, speed=None, shutter=1)
    x, y, z = wg.path3d
    np.testing.assert_array_equal(x[1:], np.array(pts[:, 0], dtype=np.float64))
    np.testing.assert_array_equal(y[1:], np.array(pts[:, 1], dtype=np.float64))
    np.testing.assert_array_equal(z[1:], np.array(pts[:, 2], dtype=np.float64))


@pytest.mark.parametrize(
    'pts, exp',
    [
        (np.random.rand(4, 123), pytest.raises(ValueError)),
        (np.random.rand(1, 123), pytest.raises(ValueError)),
        (np.random.rand(2, 123), pytest.raises(ValueError)),
        (np.random.rand(7, 123), pytest.raises(ValueError)),
        (np.random.rand(3, 123), does_not_raise()),
        (np.random.rand(3, 3), does_not_raise()),
        (np.random.rand(123, 3), does_not_raise()),
        (np.random.rand(123, 4), pytest.raises(ValueError)),
        (np.random.rand(123, 1), pytest.raises(ValueError)),
        (np.random.rand(123, 2), pytest.raises(ValueError)),
        (np.random.rand(123, 7), pytest.raises(ValueError)),
    ],
)
def test_add_points_raise(param, pts, exp) -> None:
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    with exp:
        wg.add_curve_points(pts)


@pytest.mark.parametrize(
    'dy1, dy2, dz1, dz2',
    [
        (0.2, 0.55, 0.2, -0.34),
        (3.2, -1.50, 0.2, -0.34),
        (0.2, 0.55, 0.5, 4.34),
        (0.2, 1.55, -3.2, -0.34),
        (0.2, 0.5, 0.6, 0.34),
        (0.2, 0.2, 0.2, 0.2),
        (-0.2, -0.2, -0.2, -0.2),
        (-0.2, 0.2, -0.2, 0.2),
        (0.2, -0.2, -0.2, 0.2),
    ],
)
def test_double_bend_non_sin_disp_x_none(param, dy1, dy2, dz1, dz2) -> None:
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.double_bend(dy1=dy1, dy2=dy2, dz1=dz1, dz2=dz2, fx=spline)

    _, dx1 = wg.get_sbend_parameter(np.sqrt(dy1**2 + dz1**2), wg.radius)
    _, dx2 = wg.get_sbend_parameter(np.sqrt(dy2**2 + dz2**2), wg.radius)
    assert pytest.approx(np.around(wg.lastx, decimals=8)) == dx1 + dx2
    assert pytest.approx(np.around(wg.lasty, decimals=8)) == dy1 + dy2
    assert pytest.approx(np.around(wg.lastz, decimals=8)) == dz1 + dz2


@pytest.mark.parametrize(
    'dy1, dy2, dz1, dz2, disp_x1, disp_x2',
    [
        (0.2, 0.55, 0.2, -0.34, 1, 2),
        (3.2, -1.50, 0.2, -0.34, 3, 4),
        (0.2, 0.55, 0.5, 4.34, 1, 1),
        (0.2, 1.55, -3.2, -0.34, 10, 10),
        (0.2, 0.5, 0.6, 0.34, 20, 20),
        (0.2, 0.2, 0.2, 0.2, 50, 50),
        (-0.2, -0.2, -0.2, -0.2, 4, 4),
        (-0.2, 0.2, -0.2, 0.2, 5, 5),
        (0.2, -0.2, -0.2, 0.2, 9, 3),
    ],
)
def test_double_bend_non_sin_disp_x(param, dy1, dy2, dz1, dz2, disp_x1, disp_x2) -> None:
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.double_bend(dy1=dy1, dy2=dy2, dz1=dz1, dz2=dz2, fx=spline, disp_x1=disp_x1, disp_x2=disp_x2)

    assert pytest.approx(np.around(wg.lastx, decimals=8)) == disp_x1 + disp_x2
    assert pytest.approx(np.around(wg.lasty, decimals=8)) == dy1 + dy2
    assert pytest.approx(np.around(wg.lastz, decimals=8)) == dz1 + dz2


@pytest.mark.parametrize(
    'dy1, dy2, dz1, dz2, disp_x1, disp_x2',
    [
        (0.2, 0.55, 0.2, -0.34, 1, 2),
        (3.2, -1.50, 0.2, -0.34, 3, 4),
        (0.2, 0.55, 0.5, 4.34, 1, 1),
        (0.2, 1.55, -3.2, -0.34, 10, 10),
        (0.2, 0.5, 0.6, 0.34, 20, 20),
        (0.2, 0.2, 0.2, 0.2, 50, 50),
        (-0.2, -0.2, -0.2, -0.2, 4, 4),
        (-0.2, 0.2, -0.2, 0.2, 5, 5),
        (0.2, -0.2, -0.2, 0.2, 9, 3),
    ],
)
def test_double_bend_sin_disp_x_none(param, dy1, dy2, dz1, dz2, disp_x1, disp_x2) -> None:
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.double_bend(dy1=dy1, dy2=dy2, dz1=dz1, dz2=dz2, fx=sin, disp_x1=disp_x1, disp_x2=disp_x2)

    assert pytest.approx(np.around(wg.lastx, decimals=8)) == disp_x1 + disp_x2
    assert pytest.approx(np.around(wg.lasty, decimals=8)) == dy1 + dy2
    assert pytest.approx(np.around(wg.lastz, decimals=8)) == dz1 + dz2


@pytest.mark.parametrize(
    'dy1, dy2, dz1, dz2, disp_x1',
    [
        (0.2, 0.55, 0.2, -0.34, 1),
        (3.2, -1.50, 0.2, -0.34, 3),
        (0.2, 0.55, 0.5, 4.34, 1),
        (0.2, 1.55, -3.2, -0.34, 10),
        (0.2, 0.5, 0.6, 0.34, 20),
        (0.2, 0.2, 0.2, 0.2, 50),
        (-0.2, -0.2, -0.2, -0.2, 4),
        (-0.2, 0.2, -0.2, 0.2, 5),
        (0.2, -0.2, -0.2, 0.2, 9),
    ],
)
def test_double_bend_sin_disp_x(param, dy1, dy2, dz1, dz2, disp_x1) -> None:
    wg = Waveguide(**param)
    wg.start([0, 0, 0])
    wg.double_bend(dy1=dy1, dy2=dy2, dz1=dz1, dz2=dz2, fx=sin, disp_x1=disp_x1)

    disp_x2 = np.sqrt(4 * np.abs(dy2) * wg.radius - np.abs(dy1 * dy2))
    assert pytest.approx(np.around(wg.lastx, decimals=8)) == disp_x1 + disp_x2
    assert pytest.approx(np.around(wg.lasty, decimals=8)) == dy1 + dy2
    assert pytest.approx(np.around(wg.lastz, decimals=8)) == dz1 + dz2
