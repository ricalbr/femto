from __future__ import annotations

import functools
import itertools
import os
import pathlib
from typing import Any
from typing import Generator
from typing import Iterable
from typing import Iterator

import numpy as np
import numpy.typing as npt
from shapely import geometry


def grouped(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """Group an iterable in sub-groups of n elements.

    The returned iterable have `len(iterable)//n` tuples containing n elements. If the number of elements of the input
    iterable are not a multiple of n, the remaining elements will not be returned.

    ``s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...``

    Parameters
    ----------
    iterable: iterable
        Iterable to group in subset.

    n: int
        Size of the sub-groups.

    Returns
    -------
    iterable
        Grouped iterable in n-sized subsets.
    """
    return zip(*[iter(iterable)] * n)


# s -> (s0,s1), (s2,s3), (s3, s4), ...
pairwise = functools.partial(grouped, n=2)


def swap(
    array: list[Any] | npt.NDArray[Any],
    swap_pos: list[tuple[int, int]],
) -> list[Any] | npt.NDArray[Any]:
    """Swaps elements.

    Swap elements of an array given as input.

    Parameters
    ----------
    array: list, numpy.ndarray
        Input array.
    swap_pos: list[tuple[int, int]]
        List of tuple containing indexes of the element of the array to swap.

    Returns
    -------
    list, numpy.ndarray
        Swapped array.
    """
    # in case of a single swap, swap_pos can be (pos1, pos2).

    for pos1, pos2 in swap_pos:
        array[pos1], array[pos2] = array[pos2], array[pos1]
    return array


def listcast(x: Any) -> list[Any]:
    """Cast to list.

    Cast any input object to a list.
    If `x` is a Python dictionary, the output will be the list of all the dict-keys.

    Code example:
    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> e = listcast(d)
    >>> e
    >>> ['a', 'b', 'c']

    Parameters
    ----------
    x: Any
        Input element to be casted.

    Returns
    -------
    list
        x input casted to list.
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def nest_level(lst: list[Any] | Any) -> int:
    """Nest level.

    Compute the neseting level of a list.

    Parameters
    ----------
    lst: list[Any] | Any
        Input list with a certain nested level.

    Returns
    -------
    int
        Number of nested lists, a flattened list has ``nest_level`` of 1.
    """
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    return max(nest_level(item) for item in lst) + 1


def collapse(iterable: Iterable[Any]) -> Iterable[Any]:
    """Collapse Generator.

    Flatten an iterable with multiple levels of nesting (e.g., a list of lists of tuples) into non-iterable types.

        >>> iterable = [(1, 2), ([3, 4], [[5], [6]])]
        >>> list(collapse(iterable))
        [1, 2, 3, 4, 5, 6]

    Binary and text strings are not considered iterable and will not be collapsed.

    Parameters
    ----------
    iterable: Iterable[Any]
        Input list with an arbitrary nesting level.

    Yields
    ------
    Iterable[Any]
        Flattened version of input iterable.
    """

    if isinstance(iterable, (str, bytes)):
        yield iterable
        return

    for x in iterable:
        if isinstance(x, (list, tuple)):
            yield from collapse(x)
        else:
            yield x


def flatten(iterable: Iterable[Any]) -> list[Any]:
    """Flatten list.

    Flatten an arbitrarily nested list. Returns a new list, the original list is unchanged.

    >>> flatten([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]])
    [1, 2, 3, 4, 5]
    >>> flatten([[1, 2], 3]
    [1, 2, 3]

    Parameters
    ----------
    iterable: list
        Input list with an arbitrary nesting level.

    Returns
    -------
    list
        List with the same elements of the input list but a single nesting level.
    """
    return list(collapse(iterable))


def sign() -> Iterator[int]:
    """Sign iterator.

    A generator that cycles through +1 and -1.

    Code example:
    >>> s = sign()
    >>> next(s)
    1
    >>> next(s)
    -1
    >>> next(s)
    1
    >>> next(s)
    -1
    ...

    Returns
    -------
    Iterator
        Iterator cycling through +1 and -1.
    """
    return itertools.cycle([1, -1])


def remove_repeated_coordinates(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove repeated coordinates.

    Parameters
    ----------
    array: numpy.ndarray
        Coordinate array.

    Returns
    -------
    numpy.ndarray
        Returns an array of the same dimensions as the input one in which repeated points (redundant movements) are
        substituted with NaN values.
    """

    mask = np.diff(array).astype(bool)
    mask = np.insert(mask, 0, True)
    return np.where(~mask, np.nan, array).astype(np.float64)


# Filtering adjacent identical points from a list of arrays.
def unique_filter(arrays: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Remove duplicate subsequent points.

    Filtering adjacent identical points from a list of arrays.
    The function is different from other unique functions such as numpy's `unique` function. Indeed, `unique` return
    (a sorted list of) the unique elements of the whole array. For example:

    >>> x = np.array([1, 2, 3, 3, 3, 4, 3, 3])
    >>> np.unique(x)
    np.array([1, 2, 3, 4])
    >>> unique_filter([x])
    np.array([1, 2, 3, 4, 3])

    Duplicates can be selected by creating a boolean index mask as follows:
        - make a row-wise diff (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.diff.html>`_)
        - compute absolute value of all elements in order to work only with positive numbers
        - make a column-wise sum (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_)
        - mask is converted to boolean values
    In this way consecutive duplicates correspond to a 0 value in the latter array.
    Converting this array to boolean (all non-zero values are True) the index mask can be retrieved.
    The first element is set to True by default since it is lost by the diff operation.

    `unique_filter` works also with multiple arrays. If the input list contains several elements, the arrays are
    stacked together to form a [length_array, length_list] matrix. Each row of this matrix represents the coordinates of
    a point in a space with `length_list` dimensions.
    Finally, filtering operation is applied to the newly-constructed matrix to filter all the identical adjacent points.
    For example:

    >>> x = np.array([1, 2, 3, 3, 3, 4, 3, 3])
    >>> y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    >>> unique_filter([x, y])
    np.array([1, 2, 3, 3, 4, 3, 3])

    Parameters
    ----------
    arrays: list[numpy.ndarray]
        List of arrays to filter.

    Returns
    -------
    numpy.ndarray
        Modified numpy matrix without adjacent duplicates.
    """

    # arrays list is empty
    if not arrays:
        return np.array([])

    # data matrix
    if len(arrays) == 1:
        data = arrays[0]
    else:
        data = np.stack(arrays, axis=-1).astype(np.float64)
    data.reshape(-1, len(arrays))

    # mask
    try:
        mask = np.sum(np.diff(data, axis=0), axis=1, dtype=bool)
    except np.exceptions.AxisError:  # handle 1D-data matrix case
        mask = np.diff(data).astype(bool)
    mask = np.insert(mask, 0, True)

    # filtered data
    if data.size == 0:
        return np.array([])
    else:
        return np.array(data[mask]).T


def split_mask(arr: npt.NDArray[Any], mask: npt.NDArray[np.generic]) -> list[npt.NDArray[Any]]:
    """Split maks.

    Splits an array into sub-arrays based on a mask.
    The function return the list of sub-arrays correspoding to True values.

    Parameters
    ----------

    arr: numpy.ndarray
        Input array.
    maks: numpy.ndarray[bool]
        Boolean array used as mask to split the input array.

    Returns
    -------
    list[numpy.ndarray]
        List of arrays associated to True (1) values of mask.
    """
    arr, mask = np.array(arr), np.array(mask)
    indices = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    sp = np.split(arr, indices)
    sp = sp[0::2] if mask[0] else sp[1::2]
    return sp


def pad_infinite(iterable: Iterable[Any], padding: Any = None) -> Iterator[Any]:
    return itertools.chain(iterable, itertools.repeat(padding))


def pad(iterable: Iterable[Any], size: int, padding: Any = None) -> Iterator[Any]:
    return itertools.islice(pad_infinite(iterable, padding), size)


def almost_equal(
    polygon: geometry.polygon.Polygon,
    other: geometry.polygon.Polygon,
    tol: float = 1e-6,
) -> bool:
    """
    Compute equality between polygons using the shapely builtin function symmetric_difference to build a new polygon.
    The more similar the two input polygons, the smaller the are of the new computed polygon.

    :param polygon: shaperly Polygon object
    :param other: second shapely Polygon object
    :param tol: tolerance controlling the similarity
    :return: boolean value True if the polygon are almost equal, False otherwise
    """
    return bool(polygon.symmetric_difference(other).area < tol)


def normalize_polygon(poly: geometry.Polygon) -> geometry.Polygon:
    """Normalize polygon.

    The function standardize the input polygon. It set a given orientation and set a definite starting point for
    the inner and outer rings of the polygon.

    Parameters
    ----------
    poly: geometry.Polygon
        Input ``Polygon`` object.

    Returns
    -------
    geometry.Polygon
        New ``Polygon`` object constructed with the new ordered sequence of points.

    See Also
    --------
    `This <https://stackoverflow.com/a/63402916>`_ stackoverflow answer.
    """

    def normalize_ring(ring: geometry.polygon.LinearRing) -> list[tuple[float, ...]]:
        """Normalize ring.

        It takes the exterior ring (a list of coordinates) of a ``Polygon`` object and returns the same ring,
        but with the sorted coordinates.

        Parameters
        ----------
        ring : geometry.LinearRing
            List of coordinates of a ``Polygon`` object.

        Returns
        -------
            The coordinates of the ring, sorted from the minimum value to the maximum.

        See Also
        --------
        shapely.geometry.LinearRing : ordered sequence of (x, y[, z]) point tuples.
        """
        coords = ring.coords[:-1]
        start_index = min(range(len(coords)), key=coords.__getitem__)
        return coords[start_index:] + coords[:start_index]

    poly = geometry.polygon.orient(poly)
    normalized_exterior = normalize_ring(poly.exterior)
    normalized_interiors = list(map(normalize_ring, poly.interiors))
    return geometry.Polygon(normalized_exterior, normalized_interiors)


def lookahead(iterable: Iterable[Any]) -> Generator[tuple[Any, bool]]:
    """Lookahead.

    Pass through all values from the given iterable, augmented by the information if there are more values to come
    after the current one (False), or if it is the last value (True).

    Parameters
    ----------
    iterable: iterable[Any]
        Any iterable.

    Returns
    -------
    Generator[tuple[Any, bool]]
        Returns a tuple of values, the i-th element of the iterable and a boolean value, if True the i-th element is
        the last element of the iterable.

    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    next_item = next(it)

    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield next_item, False
        next_item = val

    # Report the last value.
    yield next_item, True


def walklevel(path: str | pathlib.Path, depth: int = 1) -> Generator[tuple[Any, Any, Any]]:
    """Walklevel.

    It works just like os.walk, but you can pass it a level parameter that indicates how deep the recursion will go.
    If depth is 1, the current directory is listed.
    If depth is 0, nothing is returned.
    If depth is -1 (or less than 0), the full depth is walked.

    Parameters
    ----------
    path: str | pathlib.Path
        Path of the directory to explore.
    depth: int
        Number of directory-tree levels to traverse.

    Returns
    -------
    Generator[tuple]
        root, subdirectories, files.
    """

    # If depth is negative, just walk
    # Not using yield from for python2 compat and copy dirs to keep consistant behavior for depth = -1 and depth = inf
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs[:], files
        return
    elif depth == 0:
        return
    base_depth = str(path).rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs[:], files
        cur_depth = root.count(os.path.sep)
        if base_depth + depth <= cur_depth:
            del dirs[:]


def delete_folder(path: str | pathlib.Path) -> None:
    """Delete folder.

    Empty and remove the folder given as input.

    Parameters
    ----------
    path: str | pathlib.Path
        Directory to remove.

    Returns
    -------
    None
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    for sub in path.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    path.rmdir()


def normalize_phase(phase, zero_to_two_pi=False):
    """Normalize a phase to be within +/- pi.

    Parameters
    ----------
    phase: float
        Phase to normalize.
    zero_to_two_pi: bool
        True ->  0 to 2*pi, False -> +/- pi.

    Returns
    -------
    float
        Normalized phase within +/- pi or 0 to 2*pi
    """

    if not zero_to_two_pi:
        return (phase + np.pi) % (2 * np.pi) - np.pi
    else:
        return (phase + 2 * np.pi) % (2 * np.pi)
