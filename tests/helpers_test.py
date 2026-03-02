from __future__ import annotations

import os

import numpy as np
import pytest
from femto.helpers import almost_equal
from femto.helpers import flatten
from femto.helpers import grouped
from femto.helpers import listcast
from femto.helpers import lookahead
from femto.helpers import nest_level
from femto.helpers import pairwise
from femto.helpers import remove_repeated_coordinates
from femto.helpers import sign
from femto.helpers import split_mask
from femto.helpers import swap
from femto.helpers import unique_filter
from femto.helpers import walklevel
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
        ('pgm', ['pgm']),
        (['pgm', 1], ['pgm', 1]),
        (['pgm', 1, [1, [[4], 8, 9]]], ['pgm', 1, 1, 4, 8, 9]),
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


@pytest.mark.parametrize(
    'arr, exp',
    [
        ([], np.array([])),
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.array([1, 2, 2]), np.array([1, 2, np.nan])),
        (
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        ),
        (np.array([1, 2, 2, 2, 3, 3, 4]), np.array([1, 2, np.nan, np.nan, 3, np.nan, 4])),
    ],
)
def test_remove_doubles(arr, exp) -> None:
    a = remove_repeated_coordinates(arr)
    assert len(a) == len(exp)
    np.testing.assert_array_equal(a, exp)


def test_look_ahead() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    last_elem = [False] * 7 + [True]

    assert list(zip(a, last_elem)) == list(lookahead(a))


def test_walklevel_0() -> None:
    res = walklevel(os.getcwd(), depth=0)
    assert list(res) == []


def test_walklevel_minus1() -> None:
    from pathlib import Path

    root = Path('.') / '..'
    wl_gen = walklevel(root, depth=-1)
    wk_gen = os.walk(root)
    all(a == b for a, b in zip(wl_gen, wk_gen))


def test_walklevel() -> None:
    from pathlib import Path

    root = Path('.').absolute().parent
    wl_gen = walklevel(root, depth=1)
    wk_gen = os.walk(root)
    all(a == b for a, b in zip(wl_gen, wk_gen))
    assert next(wk_gen)  # wl_gen is exhaused but the wk_gen not, we can explore other levels
