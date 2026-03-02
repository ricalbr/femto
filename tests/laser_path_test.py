from __future__ import annotations

import math
from pathlib import Path

import attrs
import dill
import numpy as np
import pytest
from femto.laserpath import LaserPath


@pytest.fixture
def param() -> dict:
    p = {
        'scan': 6,
        'speed': 20.0,
        'y_init': 1.5,
        'z_init': 0.035,
        'lsafe': 4.3,
        'radius': 50,
        'speed_closed': 75,
        'speed_pos': 0.1,
        'samplesize': (100, 15),
        'metadata': dict(name='LPth', power='300'),
    }
    return p


@pytest.fixture
def empty_path(param) -> LaserPath:
    return LaserPath(**param)


@pytest.fixture
def laser_path(param) -> LaserPath:
    # create LaserPath instance
    lp = LaserPath(**param)

    # add path
    x = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
    y = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3])
    z = np.array([0, 0, 1, 1, 1, 0, 3, 3, 3])
    f = np.array([1, 2, 1, 1, 1, 3, 4, 4, 4])
    s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    lp.add_path(x, y, z, f, s)
    return lp


def test_default_values() -> None:
    lp = LaserPath()

    assert lp.radius == float(15)
    assert lp.scan == int(1)
    assert lp.speed == float(1.0)
    assert lp.x_init == float(-2.0)
    assert lp.y_init == float(0.0)
    assert math.isnan(lp.z_init)
    assert lp.lsafe == float(2.0)
    assert lp.speed_closed == float(5.0)
    assert lp.speed_pos == float(0.5)
    assert lp.cmd_rate_max == int(1200)
    assert lp.acc_max == int(500)
    assert lp.samplesize == (100, 50)
    assert lp.end_off_sample is True
    assert lp.warp_flag is False
    assert lp.metadata == {'name': 'LaserPath'}


def test_laserpath_values(laser_path) -> None:
    assert laser_path.radius == float(50.0)
    assert laser_path.scan == int(6)
    assert laser_path.speed == float(20.0)
    assert laser_path.x_init == float(-2.0)
    assert laser_path.y_init == float(1.5)
    assert laser_path.z_init == float(0.035)
    assert laser_path.lsafe == float(4.3)
    assert laser_path.speed_closed == float(75)
    assert laser_path.speed_pos == float(0.1)
    assert laser_path.cmd_rate_max == int(1200)
    assert laser_path.acc_max == int(500)
    assert laser_path.samplesize == (100, 15)
    assert laser_path.end_off_sample is True
    assert laser_path.warp_flag is False
    assert laser_path.metadata == dict(name='LPth', power='300')


def test_slots(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(AttributeError):
        # non-existing attribrute
        lp.sped = 10.00


def test_scan_float_err(param) -> None:
    param['scan'] = 1.2
    with pytest.raises(TypeError):
        LaserPath(**param)


def test_from_dict(param) -> None:
    lp = LaserPath.from_dict(param)

    assert lp.radius == float(50.0)
    assert lp.scan == int(6)
    assert lp.speed == float(20.0)
    assert lp.x_init == float(-2.0)
    assert lp.y_init == float(1.5)
    assert lp.z_init == float(0.035)
    assert lp.lsafe == float(4.3)
    assert lp.speed_closed == float(75)
    assert lp.speed_pos == float(0.1)
    assert lp.cmd_rate_max == int(1200)
    assert lp.acc_max == int(500)
    assert lp.samplesize == (100, 15)
    assert lp.end_off_sample is True
    assert lp.metadata == dict(name='LPth', power='300')


def test_from_dict_update(param) -> None:
    p = dict(radius=100, scan=10, speed_closed=3)
    lp = LaserPath.from_dict(param, **p)

    assert lp.radius == float(100.0)
    assert lp.scan == int(10)
    assert lp.speed == float(20.0)
    assert lp.x_init == float(-2.0)
    assert lp.y_init == float(1.5)
    assert lp.z_init == float(0.035)
    assert lp.lsafe == float(4.3)
    assert lp.speed_closed == float(3)
    assert lp.speed_pos == float(0.1)
    assert lp.cmd_rate_max == int(1200)
    assert lp.acc_max == int(500)
    assert lp.samplesize == (100, 15)
    assert lp.end_off_sample is True
    assert lp.metadata == dict(name='LPth', power='300')


def test_load(param) -> None:
    lp1 = LaserPath(**param)
    fn = Path('obj.pickle')
    with open(fn, 'wb') as f:
        dill.dump(attrs.asdict(lp1), f)

    lp2 = LaserPath.load(fn)
    assert isinstance(lp1, type(lp2))
    assert sorted(attrs.asdict(lp1)) == sorted(attrs.asdict(lp2))
    fn.unlink()


def test_id(param) -> None:
    lp = LaserPath(**param)
    assert lp.id == 'LP'


def test_repr(param) -> None:
    r = LaserPath(**param).__repr__()
    cname, _ = r.split('@')
    assert cname == 'LaserPath'


def test_init_point(laser_path) -> None:
    assert laser_path.init_point == (-2.0, 1.5, 0.035)


def test_lvelo(laser_path) -> None:
    assert pytest.approx(laser_path.lvelo) == 1.2


@pytest.mark.parametrize('s, cmd, exp', [(5, 500, 0.01), (0, 1000, 0.0), (100, 1000, 0.1), (100, 100, 1)])
def test_dl(s, cmd, exp, param) -> None:
    param['speed'] = s
    param['cmd_rate_max'] = cmd
    lp = LaserPath(**param)
    assert pytest.approx(lp.dl) == exp


def test_x_end(laser_path) -> None:
    assert pytest.approx(laser_path.x_end) == 104.3


def test_x_end_inside(param) -> None:
    lp = LaserPath(end_off_sample=False, **param)
    assert pytest.approx(lp.x_end) == 95.7


def test_x_end_none(laser_path) -> None:
    laser_path.samplesize = (None, None)
    assert laser_path.x_end is None


def test_start(param) -> None:
    lp = LaserPath(**param)
    lp.start()
    np.testing.assert_almost_equal(lp._x, np.array([-2.0, -2.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.5, 1.5]))
    np.testing.assert_almost_equal(lp._z, np.array([0.035, 0.035]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0]))


def test_start_values(param) -> None:
    init_p = [0.0, 1.0, -0.1]
    speed_p = 1.25
    lp = LaserPath(**param)

    lp.start(init_pos=init_p, speed_pos=speed_p)
    np.testing.assert_almost_equal(lp._x, np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([-0.1, -0.1]))
    np.testing.assert_almost_equal(lp._f, np.array([1.25, 1.25]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0]))


def test_start_value_error(param) -> None:
    init_p = [0.0, 1.0, -0.1, 0.3]
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)

    init_p = [0.0]
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)

    init_p = []
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start(init_p)


def test_end(param) -> None:
    lp = LaserPath(**param)
    lp.start([0.0, 0.0, 0.0])
    lp.end()

    np.testing.assert_almost_equal(lp._x, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._y, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._z, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 0.1, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 0.0, 0.0]))


def test_empty_end(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(IndexError):
        lp.end()


def test_linear_value_error(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start().linear([1, 1, 1], mode='rand').end()

    lp = LaserPath(**param)
    lp.speed = None
    with pytest.raises(ValueError):
        lp.start().linear([1, 1, 1], mode='ABS', speed=None).end()


def test_linear_invalid_increment(param) -> None:
    lp = LaserPath(**param)
    with pytest.raises(ValueError):
        lp.start().linear([1, 2]).end()

    with pytest.raises(ValueError):
        lp.start().linear([1, 2, 3, 4]).end()


def test_linear_default_speed(param) -> None:
    lp = LaserPath(**param)
    lp.start().linear([1, 2, 3])
    assert lp._f[-1] == param['speed']


def test_linear_abs(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    lp.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 3.0, 3.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_inc(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [3, 4, 5]
    lp.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 5.0, 5.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_linear_none(param) -> None:
    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [4, None, None]
    lp.start(init_p).linear(increm, mode='abs').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 4.0, 4.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))

    lp = LaserPath(**param)
    init_p = [1, 1, 1]
    increm = [5, None, None]
    lp.start(init_p).linear(increm, mode='inc').end()

    np.testing.assert_almost_equal(lp._x, np.array([1.0, 1.0, 6.0, 6.0, 1.0]))
    np.testing.assert_almost_equal(lp._y, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._z, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(lp._f, np.array([0.1, 0.1, 20.0, 20.0, 75.0]))
    np.testing.assert_almost_equal(lp._s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


# def test_circ_input_validation(param) -> None:
#     a_i, a_f = 0, np.pi / 2
#
#     param['radius'] = None
#     lp = LaserPath(**param)
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f)
#
#     lp = LaserPath(**param)
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f, radius=None)
#
#     param['radius'] = 20
#     lp = LaserPath(**param)
#     lp.radius = None
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f)
#
#     param['speed'] = None
#     lp = LaserPath(**param)
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f)
#
#     lp = LaserPath(**param)
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f, speed=None)
#
#     param['speed'] = 15
#     lp = LaserPath(**param)
#     lp.speed = None
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f)


# def test_circ_length(param) -> None:
#     a_i, a_f = 0, np.pi / 2
#
#     # DEFAULT VALUES FROM PARAM
#     lp = LaserPath(**param)
#     lp.start([0, 0, 0]).circ(a_i, a_f)
#     lc = np.abs(a_f - a_i) * lp.radius
#     # add two points from the start method
#     assert lp._x.size == int(np.ceil(lc * lp.cmd_rate_max / lp.speed)) + 2
#
#     # CUSTOM RADIUS
#     lp = LaserPath(**param)
#     custom_r = 5
#     lp.start([0, 0, 0]).circ(a_i, a_f, radius=custom_r)
#     lc = np.abs(a_f - a_i) * custom_r
#     assert lp._x.size == int(np.ceil(lc * lp.cmd_rate_max / lp.speed)) + 2
#
#     # CUSTOM SPEED
#     lp = LaserPath(**param)
#     custom_f = 5
#     lp.start([0, 0, 0]).circ(a_i, a_f, speed=custom_f)
#     lc = np.abs(a_f - a_i) * lp.radius
#     assert lp._x.size == int(np.ceil(lc * lp.cmd_rate_max / custom_f)) + 2
#
#
# def test_circ_coordinates(param) -> None:
#     a_i, a_f = 1.5 * np.pi, 0
#
#     lp = LaserPath(**param)
#     lp.start([0, 0, 0]).circ(a_i, a_f)
#     assert pytest.approx(lp._x[-1]) == lp.radius
#     assert pytest.approx(lp._y[-1]) == lp.radius
#     assert lp._z[-1] == lp._z[0]
#     lp.end()
#
#     a_i, a_f = 1.5 * np.pi, 1.75 * np.pi
#
#     lp = LaserPath(**param)
#     lp.start([0, 0, 0]).circ(a_i, a_f)
#     assert pytest.approx(lp._x[-1]) == lp.radius / np.sqrt(2)
#     assert pytest.approx(lp._y[-1]) == lp.radius * (1 - 1 / np.sqrt(2))
#     assert lp._z[-1] == lp._z[0]
#     lp.end()
#
#
# def test_circ_negative_radius(param) -> None:
#     a_i, a_f = 1.5 * np.pi, 0
#
#     lp = LaserPath(**param)
#     lp.radius = -60
#     with pytest.raises(ValueError):
#         lp.start([0, 0, 0]).circ(a_i, a_f).end()
@pytest.mark.parametrize(
    'increment, len',
    [
        ([1, 1, 1], np.sqrt(3)),
        ([1, None, 4], np.sqrt(17)),
        ([1, None, -2], np.sqrt(5)),
        ([None, None, -2], 2),
        ([-5, -5, -5], np.sqrt(75)),
    ],
)
def test_linear_warp(param, increment, len) -> None:
    lp = LaserPath(warp_flag=True, **param)
    init_p = [0, 0, 0]
    lp.start(init_p).linear(increment, mode='abs')
    assert lp.x.size == lp.num_subdivisions(len, lp.speed) + 2
    assert lp.y.size == lp.num_subdivisions(len, lp.speed) + 2
    assert lp.z.size == lp.num_subdivisions(len, lp.speed) + 2


def test_linear_warp_none(param) -> None:
    lp = LaserPath(warp_flag=True, **param)
    init_p = [0, 0, 0]
    lp.start(init_p).linear([None, None, None], mode='abs')
    assert lp.x.size == 3
    assert lp.y.size == 3
    assert lp.z.size == 3


def test_add_path(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path._x, np.array([0, 1, 1, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_almost_equal(laser_path._y, np.array([0, 0, 1, 1, 1, 2, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._z, np.array([0, 0, 1, 1, 1, 0, 3, 3, 3]))
    np.testing.assert_almost_equal(laser_path._f, np.array([1, 2, 1, 1, 1, 3, 4, 4, 4]))
    np.testing.assert_almost_equal(laser_path._s, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_x(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.x, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))


def test_x_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.x, np.array([]))


def test_lastx(laser_path) -> None:
    assert laser_path.lastx == 2.0


def test_lastx_empty(empty_path) -> None:
    with pytest.raises(IndexError):
        empty_path.lastx


def test_y(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.y, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))


def test_y_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.y, np.array([]))


def test_lasty(laser_path) -> None:
    assert laser_path.lasty == 3.0


def test_lasty_empty(empty_path) -> None:
    with pytest.raises(IndexError):
        empty_path.lasty


def test_z(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.z, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))


def test_z_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.x, np.array([]))


def test_lastz(laser_path) -> None:
    assert laser_path.lastz == 3.0


def test_lastz_empty(empty_path) -> None:
    with pytest.raises(IndexError):
        empty_path.lastz


def test_last_point(laser_path) -> None:
    np.testing.assert_almost_equal(laser_path.lastpt, np.array([2.0, 3.0, 3.0]))


def test_last_point_empty(empty_path) -> None:
    np.testing.assert_array_equal(empty_path.lastpt, np.array([]))


def test_laserpath_points(laser_path) -> None:
    x, y, z, f, s = laser_path.points
    np.testing.assert_almost_equal(x, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))
    np.testing.assert_almost_equal(y, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_almost_equal(z, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))
    np.testing.assert_almost_equal(f, np.array([1.0, 2.0, 1.0, 3.0, 4.0]))
    np.testing.assert_almost_equal(s, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))


def test_path3d(laser_path) -> None:
    xpath, ypath, zpath = laser_path.path3d

    np.testing.assert_almost_equal(xpath, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))
    np.testing.assert_almost_equal(ypath, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_almost_equal(zpath, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))


def test_path3d_empty(empty_path) -> None:
    xpath, ypath, zpath = empty_path.path3d

    np.testing.assert_array_equal(xpath, np.array([]))
    np.testing.assert_array_equal(ypath, np.array([]))
    np.testing.assert_array_equal(zpath, np.array([]))


def test_path(laser_path) -> None:
    xpath, ypath = laser_path.path

    np.testing.assert_almost_equal(xpath, np.array([0.0, 1.0, 1.0, 1.0, 2.0]))
    np.testing.assert_almost_equal(ypath, np.array([0.0, 0.0, 1.0, 2.0, 3.0]))


def test_path_empty(empty_path) -> None:
    xpath, ypath = empty_path.path

    np.testing.assert_array_equal(xpath, np.array([]))
    np.testing.assert_array_equal(ypath, np.array([]))


def test_length(laser_path) -> None:
    assert pytest.approx(laser_path.length) == 7.145052


def test_length_empty(empty_path) -> None:
    assert empty_path.length == 0.0


def test_fabrication_time(laser_path) -> None:
    assert pytest.approx(laser_path.fabrication_time) == 42.740724


def test_fabrication_time_empty(empty_path) -> None:
    assert pytest.approx(empty_path.fabrication_time) == 0.0


def test_subs_num_exception(laser_path) -> None:
    with pytest.raises(ValueError):
        laser_path.num_subdivisions(10, 0)


def test_subs_num_custom_inputs(laser_path) -> None:
    assert laser_path.num_subdivisions(103.45, 34), 3652


def test_subs_num_empty_custom_inputs(empty_path) -> None:
    assert empty_path.num_subdivisions(103.45, 34), 3652


def test_subs_num(laser_path) -> None:
    assert laser_path.num_subdivisions(10), 600


def test_subs_num_empty(empty_path) -> None:
    assert empty_path.num_subdivisions(10), 600


def test_subs_num_base_case(laser_path) -> None:
    assert laser_path.num_subdivisions(0.010), 3


def test_subs_num_empty_base_case(empty_path) -> None:
    assert empty_path.num_subdivisions(0.010), 3


def test_pickle(laser_path) -> None:
    filename = Path('test.pickle')
    laser_path.export(filename.name)
    assert filename.is_file()

    with open(filename, 'rb') as f:
        lp = dill.load(f)
    assert isinstance(lp, type(laser_path))
    filename.unlink()


def test_pickle_no_extension(laser_path) -> None:
    filename = Path('test.pickle')
    laser_path.export(filename.stem)
    assert filename.is_file()

    with open(filename, 'rb') as f:
        lp = dill.load(f)
    assert isinstance(lp, type(laser_path))
    filename.unlink()


def test_pickle_as_dict(laser_path) -> None:
    filename = Path('test.pickle')
    laser_path.export(filename.name, as_dict=True)
    assert filename.is_file()

    with open(filename, 'rb') as f:
        lp = dill.load(f)
    assert isinstance(lp, dict)
    filename.unlink()
