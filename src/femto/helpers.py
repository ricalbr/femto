from __future__ import annotations

import functools
import itertools
import pathlib
import warnings
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator

import numpy as np
import numpy.typing as npt
import yaml
from shapely import geometry


def grouped(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """
    Gruoup an iterable in sub-groups of n elements.
    The returned iterable have `len(iterable)//n` tuples containing n elements. If the number of elements of the input
    iterable are not a multiple of n, the remaining elements will not be returned.

    ``s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...``

    :param iterable: iterable to group
    :param n: size of the sub-groups
    :return: grouped iterable
    """
    return zip(*[iter(iterable)] * n)


# s -> (s0,s1), (s2,s3), (s3, s4), ...
pairwise = functools.partial(grouped, n=2)


def swap(
    array: list[Any],
    swap_pos: list[tuple[int, int]],
) -> list[Any]:
    """
    Swaps elements

    :param array:
    :param swap_pos:
    :return:
    """
    # in case of a single swap, swap_pos can be (pos1, pos2).

    for pos1, pos2 in swap_pos:
        array[pos1], array[pos2] = array[pos2], array[pos1]
    return array


def listcast(x: Any) -> list[Any]:
    """
    Cast any input object to a list.
    If `x` is a Python dictionary, the output will be the list of all the dict-keys.

    Code example

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> e = listcast(d)
    >>> e
    >>> ['a', 'b', 'c']

    :param x: input object
    :return: list
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


class dotdict(Dict[Any, Any]):
    """dot.notation access to dictionary attributes"""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            f'{self.__class__.__name__} is deprecated and it will be removed in future versions of the library. Use '
            'Dict() from the addict library instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]


def nest_level(lst: list[Any]) -> int:
    """
    Compute the neseting level of a list.

    :param lst: input object
    :return: number of nested lists, a flatten list has nest_level of 1.
    """
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    return max(nest_level(item) for item in lst) + 1


def flatten(items):
    """
    A recursive function that flattens a list.

    :param items: input list
    :return: list with the same elements of the input list but a single nesting level.
    """
    try:
        for i, x in enumerate(items):
            while isinstance(x, (list, tuple)) and not isinstance(x, (str, bytes)):
                items[i : i + 1] = x
                x = items[i]
    except IndexError:
        pass
    return items


def sign() -> Iterator[int]:
    """
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

    :return: iterator cycling through +1 and -1
    """
    return itertools.cycle([1, -1])


# Filtering adjacent identical points from a list of arrays.
def unique_filter(arrays: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Remove duplicate subsequent points.

    It takes a list of numpy arrays and returns a numpy array of unique rows. At least one coordinate have to
    change between two consecutive lines of the [X,Y,Z,F,S] matrix.

    Duplicates can be selected by creating a boolean index mask as follows:
        - make a row-wise diff (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.diff.html>`_)
        - compute absolute value of all elements in order to work only with positive numbers
        - make a column-wise sum (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_)
        - mask is converted to boolean values
    In this way consecutive duplicates correspond to a 0 value in the latter array.
    Converting this array to boolean (all non-zero values are True) the index mask can be retrieved.
    The first element is set to True by default since it is lost by the diff operation.

    Returns
    -------
    numpy.ndarray
        Modified coordinate matrix (x, y, z, f, s) without duplicates.


    Filtering adjacent identical points from a list of arrays.

    Filter adjacent identical point from array. The function is different from other unique functions such as numpy's
    `unique` function. Indeed, `unique` return (a sorted list of) the unique elements of the whole array.
    For example:

    >>> x = np.array([1, 2, 3, 3, 3, 4, 3, 3])
    >>> np.unique(x)
    np.array([1, 2, 3, 4])
    >>> unique_filter([x])
    np.array([1, 2, 3, 4, 3])

    `unique_filter` works also with multiple arrays. If the input list contains several elements, the arrays are
    stacked together to form a [length_array, length_list] matrix. Each row of this matrix represents the coordinates of
    a point in a space with `length_list` dimensions.
    Finally, filtering operation is applied to the newly-constructed matrix to filter all the identical adjacent points.
    For example:

    >>> x = np.array([1, 2, 3, 3, 3, 4, 3, 3])
    >>> y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    >>> unique_filter([x, y])
    np.array([1, 2, 3, 3, 4, 3, 3])

    :param arrays: list of arrays
    :return: matrix of arrays, each row has the filtered points on a given coordinate-axis.
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
    except np.AxisError:  # handle 1D-data matrix case
        mask = np.diff(data).astype(bool)
    mask = np.insert(mask, 0, True)

    # filtered data
    if data.size == 0:
        return np.array([])
    else:
        return np.array(data[mask]).T


def split_mask(arr: npt.NDArray[Any], mask: npt.NDArray[np.generic]) -> list[npt.NDArray[Any]]:
    """
    Splits an array into sub-arrays based on a mask.

     The function return the list of sub-arrays corresponding to True values.

    :param arr: Input array
    :param mask: Boolean array used as mask to split the input array
    :return: List of arrays associated to True (1) values of mask.
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


def load_parameters(param_file: str | pathlib.Path) -> list[dict]:
    """
    The `load_param` function loads a YAML configuration file and returns a list of dictionaries containing the
    parameters.
    The function first opens the YAML file in binary mode and uses the `tomli` library to load the configuration into a
    dictionary. The function then removes the `DEFAULT` dictionary from the configuration and merges its contents
    with the other dictionaries in the configuration. Finally, the function returns a list of dictionaries,
    where each dictionary contains the merged contents of the `DEFAULT` dictionary and one of the other dictionaries
    in the configuration.

    Parameters
    ----------
    param_file: str, pathlib.Path
        Path to the YAML parameter file.

    Returns
    -------
    List of the parameters dictionaries.
    """

    fp = pathlib.Path(param_file).stem + '.yaml'
    with open(fp, mode='rb') as f:
        config = yaml.safe_load(f)

    if not config:
        return []

    # remove the DEFAULT dictionary and merge it to all the other dictionaries
    try:
        default_dict = config.pop('DEFAULT')
    except KeyError:
        default_dict = {}

    dc = [dict(config[s]) for s in config.keys()]
    return [{**default_dict, **param_dict} for param_dict in dc]


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

    def normalize_ring(ring: geometry.polygon.LinearRing):
        """Normalize ring
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


def lookahead(iterable):
    """Lookahead.
    Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (False), or if it is the last value (True).
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
