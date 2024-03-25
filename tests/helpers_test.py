from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml
from femto.helpers import almost_equal
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.helpers import grouped
from femto.helpers import listcast
from femto.helpers import load_parameters
from femto.helpers import nest_level
from femto.helpers import pairwise
from femto.helpers import sign
from femto.helpers import split_mask
from femto.helpers import swap
from femto.helpers import unique_filter
from femto.waveguide import Waveguide
from shapely.geometry import Point
from shapely.geometry import Polygon


@pytest.mark.parametrize(
    'it,n, exp',
    [
        ([], 5, []),
        ([], 0, []),
        ((), 3, []),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9), 3, [(1, 2, 3), (4, 5, 6), (7, 8, 9)]),
        ([1, 2, 3, 4], 1, [(1,), (2,), (3,), (4,)]),
        ([1, 2, [3, 4], 5], 2, [(1, 2), ([3, 4], 5)]),
        ([1, 2, 3, 4, 5, 6], 5, [(1, 2, 3, 4, 5)]),
    ],
)
def test_grouped(it, n, exp):
    assert list(grouped(it, n)) == exp


@pytest.mark.parametrize(
    'it, exp',
    [
        ([], []),
        ((), []),
        ([1, 2, 3, 4], [(1, 2), (3, 4)]),
        ([1, 2, 3], [(1, 2)]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]),
    ],
)
def test_pairwise(it, exp) -> None:
    assert list(pairwise(it)) == exp


def test_pairwise_wg() -> None:
    wg1 = Waveguide()
    wg2 = Waveguide()
    wg3 = Waveguide()
    wg4 = Waveguide()
    assert list(pairwise([wg1, wg2, wg3, wg4])) == [(wg1, wg2), (wg3, wg4)]


@pytest.mark.parametrize(
    'lst, swp, exp',
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [(0, 9)], [10, 2, 3, 4, 5, 6, 7, 8, 9, 1]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [(1, 2), (2, 3), (3, 4)], [1, 3, 4, 5, 2, 6, 7, 8, 9, 10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [(1, 2)], [1, 3, 2, 4, 5, 6, 7, 8, 9, 10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [(1, 2), (8, 3)], [1, 3, 2, 9, 5, 6, 7, 8, 4, 10]),
    ],
)
def test_swap(lst, swp, exp):
    assert swap(lst, swp) == exp


@pytest.mark.parametrize(
    'inp, exp',
    [
        ([], []),
        ((), []),
        (None, [None]),
        (1, [1]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([[[[[[1, 2, [3, [[[[4, 5]]]]]]]]]]], [[[[[[1, 2, [3, [[[[4, 5]]]]]]]]]]]),
        ((1, 2, 3, 4, [5, 6, 7]), [1, 2, 3, 4, [5, 6, 7]]),
        ((1, 2, 3, 4, [5, 6, 7, (8, 9)]), [1, 2, 3, 4, [5, 6, 7, (8, 9)]]),
        ('string', ['string']),
        (['str', 'ing'], ['str', 'ing']),
        (('str', 'ing'), ['str', 'ing']),
        (('str', ['ing']), ['str', ['ing']]),
        ({'a': 'str', 'b': 'ing'}, ['a', 'b']),
    ],
)
def test_listcast(inp, exp):
    assert listcast(inp) == exp


@pytest.mark.parametrize(
    'inp, exp',
    [
        (1, 0),
        (None, 0),
        ([], 1),
        ([1, 2, 3], 1),
        ([1, 2, [3]], 2),
        ([[1, 2], [3, 4]], 2),
        ([[1], [2], [3, [4]], [5]], 3),
        ([[[[[[6]]]]], 7], 6),
        ([[[[[[[7]]]]]], [[[[[[[[[[5]]]]]]]]]]], 11),
    ],
)
def test_nest_level(inp, exp):
    assert nest_level(inp) == exp


@pytest.mark.parametrize(
    'inp, exp',
    [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, [3]], [1, 2, 3]),
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([[1], [2], [3, [4]], [5]], [1, 2, 3, 4, 5]),
        ([[[[[[6]]]]], 7], [6, 7]),
        ([[[[[[[7]]]]]], [[[[[[[[[[5]]]]]]]]]]], [7, 5]),
        ([(3, 4, 5, 6), [5, 4]], [3, 4, 5, 6, 5, 4]),
    ],
)
def test_flatten(inp, exp):
    assert flatten(inp) == exp


@pytest.mark.parametrize('n, exp', [(0, 0), (1, 1), (2, -1), (99, 1), (106, -1), (334, -1)])
def test_sign(n, exp) -> None:
    s = sign()
    sig = 0
    for _ in range(n):
        sig = next(s)
        print(sig)
    assert sig == exp


@pytest.mark.parametrize(
    'inp, m, exp',
    [
        (np.array([1, 2, 3, 4, 5]), [0, 0, 0, 0, 0], [np.array([])]),
        (np.array([1, 2, 3, 4, 5]), np.array([0, 1, 1, 1, 1]), [np.array([2, 3, 4, 5])]),
        (np.array([1, 2, 3, 4, 5]), np.array([1, 0, 0, 0, 0]), [np.array([1])]),
        (np.array([1, 2, 3, 4, 5]), np.array([1, 0, 1, 0, 1]), [np.array([1]), np.array([3]), np.array([5])]),
        (np.array([1, 2, 3, 4, 5]), [1, 0, 1, 0, 1], [np.array([1]), np.array([3]), np.array([5])]),
        (np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1, 1, 1]), [np.array([1, 2, 3, 4, 5])]),
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]),
            [
                np.array([1, 2]),
                np.array([4, 5]),
                np.array([7, 8]),
                np.array([0, 1]),
                np.array([3, 4]),
                np.array([6, 7]),
                np.array([9]),
            ],
        ),
    ],
)
def test_split_mask(inp, m, exp):
    for comp, expected in zip(split_mask(inp, m), exp):
        np.testing.assert_array_equal(comp, expected)


@pytest.mark.parametrize(
    'pol, oth, tol',
    [
        (Point(0, 0).buffer(1), Point(0, 0).buffer(1), 1e-9),
        (Point(0, 0).buffer(1), Point(0, 0).buffer(1), 1e-8),
        (Point(0, 0).buffer(1), Point(0, 0).buffer(1), 1e-7),
        (Point(0, 0).buffer(1.005), Point(0, 0).buffer(1), 1e-1),
        (Point(0, 0).buffer(1), Point(0, 0).buffer(1.0000001), 1e-6),
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 1e-6),
        (
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).buffer(-0.1),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).buffer(-0.1),
            1e-6,
        ),
    ],
)
def test_almost_equals(pol, oth, tol):
    assert almost_equal(pol, oth, tol)


@pytest.mark.parametrize(
    'inp, exp',
    [
        ([], np.array([])),
        ([np.array([])], np.array([])),
        ([np.array([1, 2, 3])], np.array([1, 2, 3])),
        (
            [
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        6,
                        6,
                    ]
                )
            ],
            np.array([1, 2, 3, 4, 5, 6]),
        ),
        (
            [np.array([1, 2, 3, 4, 4, 5, 6]), np.array([0, 1, 0, 1, 0, 1, 0])],
            np.array([[1, 2, 3, 4, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0]]),
        ),
        ([np.array([1, 2, 3, 4, 4, 4, 4]), np.array([1, 1, 1, 1, 1, 1, 1])], np.array([[1, 2, 3, 4], [1, 1, 1, 1]])),
        (
            [np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]), np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0])],
            np.array([[1, 1, 2, 2, 3, 3, 4, 4], [1, 0, 1, 0, 1, 0, 1, 0]]),
        ),
        (
            [np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]).T, np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]).T],
            np.array([[1, 1, 2, 2, 3, 3, 4, 4], [1, 0, 1, 0, 1, 0, 1, 0]]),
        ),
    ],
)
def test_unique_filter(inp, exp):
    np.testing.assert_array_equal(unique_filter(inp), exp)


def test_dotdict():
    def_dict = dict(a=3, b='str', c=9.0)
    dd = dotdict(a=3, b='str', c=9.0)

    def_dict['new_val'] = 99
    dd['new_val'] = 99

    del def_dict['a']
    del dd['a']

    def_dict['b'] = 'string'
    dd['b'] = 'string'

    for k, e in dd.items():
        assert def_dict[k] == dd[k]
    del dd

    vv = {'lst': [1, 2, 3]}
    dd = dotdict(vv)

    lst = dd.lst
    llst = [lst]
    tpl = tuple(lst)
    dd.lst = llst
    dd.tpl = tpl

    for k in dd:
        print(dd.k)

    del dd.tpl
    del dd


def test_load_param_empty_default():

    PARAM_WG = dict(
        scan=6,
        speed=15,
        speed_closed=60,
        radius=15,
        y_init=1.25,
        depth=0.035,
        pitch_fa=0.127,
        shrink_correction_factor=0.9993,
        pitch=0.080,
        int_length=0.0,
        arm_length=0.0,
        ltrench=0.0,
        dz_bridge=0.000,
        lsafe=2,
    )
    PARAM_MK = dict(
        scan=1,
        speed=1,
        depth=0.000,
        speed_pos=5,
        lx=1.0,
        ly=0.040,
    )
    PARAM_GC = dict(filename='UPP8.pgm', laser='PHAROS', aerotech_angle=0.0, rotation_angle=0.0)
    p_dicts = {'DEFAULT': {}, 'wg': PARAM_WG, 'mk': PARAM_MK, 'gc': PARAM_GC}

    with open('test.yaml', 'w') as f:
        yaml.dump(p_dicts, f, sort_keys=False)

    pw, pm, pg = load_parameters('test.yaml')
    assert pw == PARAM_WG
    assert pm == PARAM_MK
    assert pg == PARAM_GC
    pathlib.Path('test.yaml').unlink()


def test_load_param_no_default():

    PARAM_WG = dict(
        scan=6,
        speed=15,
        speed_closed=60,
        radius=15,
        y_init=1.25,
        depth=0.035,
        pitch_fa=0.127,
        shrink_correction_factor=0.9993,
        pitch=0.080,
        int_length=0.0,
        arm_length=0.0,
        ltrench=0.0,
        dz_bridge=0.000,
        lsafe=2,
    )
    PARAM_MK = dict(
        scan=1,
        speed=1,
        depth=0.000,
        speed_pos=5,
        lx=1.0,
        ly=0.040,
    )
    PARAM_GC = dict(
        filename='UPP8.pgm',
        laser='PHAROS',
        aerotech_angle=0.0,
        rotation_angle=0.0,
    )
    p_dicts = {'wg': PARAM_WG, 'mk': PARAM_MK, 'gc': PARAM_GC}

    with open('test.yaml', 'w') as f:
        yaml.dump(p_dicts, f, sort_keys=False)

    pw, pm, pg = load_parameters('test.yaml')
    assert pw == PARAM_WG
    assert pm == PARAM_MK
    assert pg == PARAM_GC
    pathlib.Path('test.yaml').unlink()


def test_load_param_pathlib():

    PARAM_WG = dict(
        scan=6,
        speed=15,
        speed_closed=60,
        radius=15,
        y_init=1.25,
        depth=0.035,
        pitch_fa=0.127,
        shrink_correction_factor=0.9993,
        pitch=0.080,
        int_length=0.0,
        arm_length=0.0,
        ltrench=0.0,
        dz_bridge=0.000,
        lsafe=2,
    )
    PARAM_MK = dict(
        scan=1,
        speed=1,
        depth=0.000,
        speed_pos=5,
        lx=1.0,
        ly=0.040,
    )
    PARAM_GC = dict(filename='UPP8.pgm', laser='PHAROS', aerotech_angle=0.0, rotation_angle=0.0)
    p_dicts = {'wg': PARAM_WG, 'mk': PARAM_MK, 'gc': PARAM_GC}

    fp = pathlib.Path('test.yaml')
    with open(fp, 'w') as f:
        yaml.dump(p_dicts, f, sort_keys=False)

    pw, pm, pg = load_parameters(fp)
    assert pw == PARAM_WG
    assert pm == PARAM_MK
    assert pg == PARAM_GC
    pathlib.Path('test.yaml').unlink()


def test_load_param_empty():
    p_dicts = {}
    fp = pathlib.Path('test.yaml')
    with open(fp, 'w') as f:
        yaml.dump(p_dicts, f, sort_keys=False)

    assert load_parameters(fp) == []
    pathlib.Path('test.yaml').unlink()
