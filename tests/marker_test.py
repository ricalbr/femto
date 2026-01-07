from __future__ import annotations

import numpy as np
import pytest
from femto.marker import Marker


@pytest.fixture
def param() -> dict:
    p = {
        'scan': 1,
        'speed': 2.0,
        'y_init': 1.5,
        'z_init': 0.050,
        'speed_closed': 24,
        'depth': 0.001,
        'lx': 2.0,
        'ly': 0.050,
    }
    return p


@pytest.fixture
def empty_mk(param) -> Marker:
    return Marker(**param)


def test_default_values() -> None:
    mk = Marker()
    assert mk.scan == int(1)
    assert mk.speed == float(1.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(0.0)
    assert mk.z_init == float(0.0)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(5.0)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (100, 50)
    assert mk.depth == float(0.0)
    assert mk.lx == float(1.0)
    assert mk.ly == float(0.060)


def test_mk_values(param) -> None:
    mk = Marker(**param)
    assert mk.scan == int(1)
    assert mk.speed == float(2.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(1.5)
    assert mk.z_init == float(0.050)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(24)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (100, 50)
    assert mk.depth == float(0.001)
    assert mk.lx == float(2.0)
    assert mk.ly == float(0.050)


def test_mk_from_dict(param) -> None:
    mk = Marker.from_dict(param)
    assert mk.scan == int(1)
    assert mk.speed == float(2.0)
    assert mk.x_init == float(-2.0)
    assert mk.y_init == float(1.5)
    assert mk.z_init == float(0.050)
    assert mk.lsafe == float(2.0)
    assert mk.speed_closed == float(24)
    assert mk.speed_pos == float(0.5)
    assert mk.cmd_rate_max == int(1200)
    assert mk.acc_max == int(500)
    assert mk.samplesize == (100, 50)
    assert mk.depth == float(0.001)
    assert mk.lx == float(2.0)
    assert mk.ly == float(0.050)


def test_z_init(param) -> None:
    del param['z_init']
    param['depth'] = -0.001
    mk = Marker(**param)
    assert mk.z_init == float(-0.001)


def test_scan(param) -> None:
    param['scan'] = 1.2
    with pytest.raises(TypeError):
        Marker(**param)


def test_slots(param) -> None:
    m = Marker(**param)
    with pytest.raises(AttributeError):
        # non-existing attribrute
        m.zinit = 0.00


def test_id(param) -> None:
    mk = Marker(**param)
    assert mk.id == 'MK'


def test_repr(param) -> None:
    r = Marker(**param).__repr__()
    cname, _ = r.split('@')
    assert cname == 'Marker'


def test_cross_init_pos(param) -> None:
    i_pos = [1, 2]
    mk = Marker(**param)
    mk.cross(i_pos)
    x, y, z, *_ = mk.points

    assert pytest.approx(x[-1]) == i_pos[0]
    assert pytest.approx(y[-1]) == i_pos[1]
    assert pytest.approx(z[-1]) == mk.depth

    i_pos = [0.0, 0.2, 5.9]
    mk = Marker(**param)
    mk.cross(i_pos)
    x, y, z, *_ = mk.points

    assert pytest.approx(x[-1]) == i_pos[0]
    assert pytest.approx(y[-1]) == i_pos[1]
    assert pytest.approx(z[-1]) == i_pos[2]

    i_pos = []
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)

    i_pos = [1]
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)

    i_pos = [8, 4, 3.45, 22.12, 89]
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.cross(i_pos)


def test_cross_l_default(param) -> None:
    i_pos = [0, 0, 0]
    mk = Marker(**param)
    mk.cross(i_pos)

    assert pytest.approx(np.max(mk.x)) == float(1.0)
    assert pytest.approx(np.min(mk.x)) == float(-1.0)
    assert pytest.approx(np.max(mk.y)) == float(0.025)
    assert pytest.approx(np.min(mk.y)) == float(-0.025)


def test_cross_l_custom(param) -> None:
    i_pos = [0, 0, 0]
    lx = 3
    ly = 1
    mk = Marker(**param)
    mk.cross(i_pos, lx, ly)

    assert pytest.approx(np.max(mk.x)) == float(1.5)
    assert pytest.approx(np.min(mk.x)) == float(-1.5)
    assert pytest.approx(np.max(mk.y)) == float(0.5)
    assert pytest.approx(np.min(mk.y)) == float(-0.5)


def test_cross_points(param) -> None:
    i_pos = [0, 0, 0.1]
    mk = Marker(**param)
    mk.cross(i_pos)

    x, y, z, f, s = mk.points
    np.testing.assert_almost_equal(x, np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(y, np.array([0.0, 0.0, 0.0, 0.0, -0.025, -0.025, 0.025, 0.025, 0.0]))
    np.testing.assert_almost_equal(z, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    np.testing.assert_almost_equal(f, np.array([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    np.testing.assert_almost_equal(s, np.array([0, 1, 1, 0, 0, 1, 1, 0, 0]))


def test_ruler_yticks(param) -> None:
    yt = []
    mk = Marker(**param)

    # empty list return None
    assert mk.ruler(yt) is None
    assert len(mk.y) == 0

    # check the given values are present in the matrix of points
    yt = np.array([0, 0.3, 0.6, 0.78, 0.99, 1, 45])
    mk = Marker(**param)
    mk.ruler(yt)
    print(mk.y[2:-2:4])
    assert pytest.approx(mk.y[2:-2:4]) == yt

    # check the list is ordered and unique
    yt = np.array([3, 2, 6, 4, 8, 9, 1, 3])
    mk = Marker(**param)
    mk.ruler(yt)
    assert pytest.approx(mk.y[2:-2:4]) == np.unique(yt)


def test_ruler_lx(param) -> None:
    # test default lx
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(np.max(mk.x)) == mk.lx

    # test custom lx
    lxx = 5
    mk = Marker(**param)
    mk.ruler([1, 2, 3], lx=lxx)
    assert pytest.approx(np.max(mk.x)) == lxx


def test_ruler_lx_short(param) -> None:
    # test default lx2
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(mk.x[8]) == 0.75 * mk.lx

    # test custom lx2
    lxx = 5
    mk = Marker(**param)
    mk.ruler([1, 2, 3], lx2=lxx)
    assert pytest.approx(mk.x[8]) == lxx


def test_ruler_x_init(param) -> None:
    # test default x_init
    mk = Marker(**param)
    mk.ruler([1, 2, 3])
    assert pytest.approx(mk.x[0]) == mk.x_init

    # test custom x_init
    x0 = 0.0
    mk = Marker(**param)
    mk.ruler([1, 2, 3], x_init=x0)
    assert pytest.approx(mk.x[0]) == x0

    x1 = 2.0
    mk = Marker(**param)
    mk.ruler([1, 2, 3], x_init=x1)
    assert pytest.approx(mk.x[0]) == x1


def test_ruler_points(param) -> None:
    mk = Marker(**param)
    mk.ruler([1, 2, 3, 4], 5, 2, x_init=-2)

    x, y, z, f, s = mk.points
    print(x)
    print(y)
    print(z)
    print(f)
    print(s)
    np.testing.assert_almost_equal(
        x,
        np.array(
            [-2.0, -2.0, -2.0, -2.0, 5.0, 5.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0]
        ),
    )
    np.testing.assert_almost_equal(
        y, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0])
    )
    np.testing.assert_almost_equal(z, np.repeat(0.001, len(y)))
    np.testing.assert_almost_equal(
        f, np.array([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 24.0])
    )
    np.testing.assert_almost_equal(
        s, np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )


@pytest.mark.parametrize('ori', ['d', '', 'xy', 'z'])
def test_meander_error(param, ori) -> None:
    mk = Marker(**param)
    with pytest.raises(ValueError):
        mk.meander([1, 2, 3], [4, 5, 6], width=0.01, orientation=ori)


@pytest.mark.parametrize('i_pos', [[], [1], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
def test_meander_init_pos_error(param, i_pos) -> None:
    mk = Marker(**param)
    f_pos = [1, 2, 3]
    with pytest.raises(ValueError):
        mk.meander(i_pos, f_pos, width=0.01)


@pytest.mark.parametrize('f_pos', [[], [1], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
def test_meander_final_pos_error(param, f_pos) -> None:
    mk = Marker(**param)
    i_pos = [1, 2, 3]
    with pytest.raises(ValueError):
        mk.meander(i_pos, f_pos, width=0.01)


def test_meander_width(param) -> None:
    i_pos = [0, 0, 0]
    f_pos = [1, 2]
    mk = Marker(**param)
    mk.meander(i_pos, f_pos, orientation='x')
    x = mk.x
    assert pytest.approx(np.max(x)) == i_pos[0] + 1  # width = 1 by default
    assert pytest.approx(np.min(x)) == i_pos[0]

    i_pos = [0, 0, 0]
    f_pos = [1, 2]
    w = 5
    mk = Marker(**param)
    mk.meander(i_pos, f_pos, width=5, orientation='y')
    y = mk.y
    assert pytest.approx(np.max(y)) == i_pos[1] + w
    assert pytest.approx(np.min(y)) == i_pos[1]


def test_meader_points_x_or(param) -> None:
    mk = Marker(**param)
    mk.meander([0, 0, 0], [1, 2, 3], width=2, delta=0.5, orientation='x')

    x, y, z, f, s = mk.points
    np.testing.assert_almost_equal(x, np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0]))
    np.testing.assert_almost_equal(y, np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 0.0]))
    np.testing.assert_almost_equal(z, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(f, np.array([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 24.0]))
    np.testing.assert_almost_equal(s, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_meader_points_y_or(param) -> None:
    mk = Marker(**param)
    mk.meander([0, 0, 0], [-5, -1, -2], width=3, delta=1, orientation='y')

    x, y, z, f, s = mk.points
    np.testing.assert_almost_equal(
        x, np.array([0.0, 0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0, -4.0, -4.0, -5.0, -5.0, -5.0, 0.0])
    )
    np.testing.assert_almost_equal(
        y, np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_almost_equal(
        z, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_almost_equal(
        f, np.array([0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 24.0])
    )
    np.testing.assert_almost_equal(
        s, np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    )


def test_ablation_empty(param) -> None:
    mk = Marker(**param)
    assert mk.ablation([]) is None


def test_ablation_shift_default(param) -> None:
    mk = Marker(**param)
    mk.ablation([[0, 0, 0], [5, 0, 0]], shift=None)
    x, y, *_ = mk.points

    assert len(x) == 6
    np.testing.assert_almost_equal(x, np.array([0.0, 0.0, 0.0, 5.0, 5.0, 0.0]))
    np.testing.assert_almost_equal(y, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # test None values


def test_ablation_shift_custom(param) -> None:
    mk = Marker(**param)
    mk.ablation([[0, 0, 0], [5, 0, 0]], shift=0.1)
    x, y, *_ = mk.points

    assert len(x) == 22
    np.testing.assert_almost_equal(
        x,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                5.0,
                5.0,
                0.0,
                0.0,
                5.0,
                5.0,
                0.1,
                0.1,
                5.1,
                5.1,
                0.0,
                0.0,
                5.0,
                5.0,
                -0.1,
                -0.1,
                4.9,
                4.9,
                0.0,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        y,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.1,
                0.1,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_box_empty(param) -> None:
    mk = Marker(**param)
    mk.box(lower_left_corner=None)

    np.testing.assert_almost_equal(mk.x, np.array([]))
    np.testing.assert_almost_equal(mk.y, np.array([]))
    np.testing.assert_almost_equal(mk.z, np.array([]))


@pytest.mark.parametrize(
    'w, h, exp_x, exp_y, exp_z',
    [
        (0, 0, np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])),
        (1, 0, np.array([0, 0, 0, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])),
        (0, 1, np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])),
        (
            1,
            1,
            np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 1, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (
            1,
            2,
            np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 2, 2, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (
            5,
            5,
            np.array([0, 0, 0, 5, 5, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 5, 5, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (
            -5,
            -5,
            np.array([0, 0, 0, 5, 5, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 5, 5, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),
    ],
)
def test_box(w, h, exp_x, exp_y, exp_z, param) -> None:
    mk = Marker(**param)
    mk.box(lower_left_corner=[0, 0, 0], height=h, width=w)

    np.testing.assert_almost_equal(mk.x, exp_x)
    np.testing.assert_almost_equal(mk.y, exp_y)
    np.testing.assert_almost_equal(mk.z, exp_z)
