from copy import deepcopy
from typing import Generator, Iterator, List

import numpy as np
import shapely.geometry
from descartes import PolygonPatch
from shapely.geometry import LineString, Polygon, polygon

from femto.helpers import dotdict, flatten, listcast
from femto.Parameters import TrenchParameters
from femto.Waveguide import _Waveguide, Waveguide


class Trench:
    """
    Class representing a single trench block.
    """

    def __init__(self, block: Polygon, delta_floor: float = 0.001):

        self.block = block
        self.delta_floor = delta_floor
        self.floor_length = 0.0
        self.wall_length = 0.0

    def __lt__(self, other):
        return self.block.exterior.coords.xy[1][0] < other.block.exterior.coords.xy[1][0]

    @property
    def center(self) -> list:
        """
        Returns the (x, y) coordinates of the trench block baricenter point.

        :return: (x, y) coordinates of the block's center point
        :rtype: list
        """
        return [self.block.centroid.x, self.block.centroid.y]

    @property
    def patch(self, kwargs=None) -> PolygonPatch:
        """
        Return a Patch obj representing the trench block for plotting.

        :param kwargs: Plotting options supported for matplotlib.patches.Polygon class.
        :type kwargs: dict
        :return: Patch obj for plotting the trench block
        :rtype: descartes.PolygonPatch
        """
        if kwargs is None:
            kwargs = {}
        default_kwargs = {'facecolor': 'k', 'edgecolor': None, 'alpha': 1, 'zorder': 1}
        kwargs = {**default_kwargs, **kwargs}
        return PolygonPatch(self.block, **kwargs)

    @property
    def xmin(self):
        return self.block.bounds[0]

    @property
    def ymin(self):
        return self.block.bounds[1]

    @property
    def xmax(self):
        return self.block.bounds[2]

    @property
    def ymax(self):
        return self.block.bounds[3]

    def trench_paths(self) -> Generator[np.ndarray, None, None]:
        """
        Generator of the inset paths of the trench block.

        First, the outer trench polygon obj is insert in the trench ``polygon_list``. While the list is not empty
        we can extract the outer polygon from the list and compute the ``inset_polygon`` and insert it back to the list.
        ``inset_polygon`` can be:
        ``Polygon`` obj
            The obj is appended to the ``inset_polygon`` list and the exterior (x, y) coordinates are yielded.
        ``MultiPolygon`` obj
            All the single ``Polygon`` objects composing the ``MultiPolygon`` are appended to the ``inset_polygon``
            list as ``Polygon`` objects and the exterior (x, y) coordinates are yielded.
        ``None``
            In this case, we cannot extract a ``inset_polygon`` from the ``Polygon`` obj extracted from the
            ``inset_polygon``. Nothing is appended to the ``polygon_list`` and its size is reduced.

        :return: (x, y) coordinates of the inset path.
        :rtype: Generator[numpy.ndarray]
        """
        polygon_list = [self.block]
        self.wall_length = self.block.length
        while polygon_list:
            current_poly = polygon_list.pop(0)
            inset_polygon = self._buffer_polygon(current_poly)
            if inset_polygon and inset_polygon.type == 'MultiPolygon':
                polygon_list.extend(list(inset_polygon.geoms))
                for poly in list(inset_polygon.geoms):
                    self.floor_length += poly.length
                    yield np.array(poly.exterior.coords).T
            elif inset_polygon and inset_polygon.type == 'Polygon':
                self.floor_length += inset_polygon.length
                polygon_list.append(inset_polygon)
                yield np.array(inset_polygon.exterior.coords).T
            elif inset_polygon:
                raise ValueError(f'Trench block should be either Polygon or Multipolygon. Given {inset_polygon.type}')

    # Private interface
    def _buffer_polygon(self, shape: Polygon, inset: bool = True) -> shapely.geometry.shape:
        """
        Compute a buffer operation of shapely ``Polygon`` obj.

        :param shape: ``Polygon`` of the trench block
        :type shape: shapely.geometry.Polygon
        :param inset: If ``True`` the eroded polygon is computed (inset), if ``False`` the dilated polygon is
        computed (outset).
        :type inset: bool
        :return: Buffered polygon
        :rtype: shapely.geometry.shape

        .. note::
        The buffer operation returns a polygonal result. The new polygon is checked for validity using
        ``obj.is_valid`` in the sense of [#]_.

        For a reference, read the buffer operations `here
        <https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods>`_
        .. [#] John R. Herring, Ed., “OpenGIS Implementation Specification for Geographic information - Simple feature
        access - Part 1: Common architecture,” Oct. 2006
        """
        if inset:
            new_polygon = shape.buffer(-self.delta_floor)
        else:
            new_polygon = shape.buffer(self.delta_floor)
        return new_polygon if new_polygon.is_valid else None


class TrenchColumn(TrenchParameters):
    """
    Class representing a column of trenches.
    """

    def __init__(self, param: dict):
        super().__init__(**param)
        self._trench_list = []

    def __iter__(self) -> Iterator[Trench]:
        """
        Iterator that yields the single trench blocks of the column.

        :return: Trench obj of the trench column
        :rtype: Trench
        """
        return iter(self._trench_list)

    @property
    def fabrication_time(self) -> float:
        l_tot = 0.0
        for trench in self._trench_list:
            l_tot += self.nboxz * (self.n_repeat * trench.wall_length + trench.floor_length)
        return l_tot / self.speed

    @property
    def trenches(self) -> list:
        return self._trench_list

    def get_trench(self, waveguides: List, remove=None):
        """
        Compute the trench blocks from the waveguide of the optical circuit.
        To get the trench blocks, the waveguides are used as mold matrix for the trenches. The waveguides are
        converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference, the rectangle (minx, miny, maxx, maxy) = (x_c - l, y_min, x_c + l, y_max) is cut
        obtaining a ``MultiPolygon`` with all the trench blocks.

        All the blocks are treated individually. Each block is then buffered to obtain an outset polygon with rounded
        corners a Trench obj is created with the new polygon box and the trenches are appended to the
        ``trench_list``.

        :param waveguides: List of the waveguides composing the optical circuit.
        :type waveguides: List[_Waveguide]
        :param remove: List of trench to remove.
        :type remove: List[int]
        """

        if remove is None:
            remove = []

        waveguides = flatten(deepcopy(waveguides))
        trench_block = self.rect
        for wg in waveguides:
            x, y = self._extract_path(wg)
            dilated = (LineString(list(zip(x, y))).buffer(self.adj_bridge, cap_style=1))
            trench_block = trench_block.difference(dilated)

        for block in listcast(sorted(trench_block.geoms, key=Trench)):
            block = (polygon.orient(block).buffer(self.round_corner, resolution=250, cap_style=1))
            trench = Trench(block, self.delta_floor)
            self._trench_list.append(trench)

        for index in sorted(listcast(remove), reverse=True):
            del self._trench_list[index]

    @staticmethod
    def _extract_path(waveguide):
        """
        Extract the x, y path from Waveguide or numpy.ndarray object.

        The input object get parsed and if it is a Waveguide object the points are extracted and the shutter-closed
        points are removed from the 2D matrix.
        Alterntively, if the input object is a numpy.ndarray, it is implicity assumed that it is a 2D matrix with
        just the shutter-open path. (x, y) coordinates are extracted and returned.

        :param waveguide: Waveguide object or 2D numpy.ndarray with the path xy-points.
        :type waveguide: _Waveguide or numpy.ndarray
        """

        if isinstance(waveguide, _Waveguide):
            x, y = waveguide.path
        elif isinstance(waveguide, np.ndarray):
            x, y = waveguide.T
        else:
            raise TypeError('Elements circuit list must be of type _Waveguide or 2D numpy.array.')
        return x, y


def _example():
    import matplotlib.pyplot as plt

    # Data
    x_mid = None

    PARAMETERS_WG = dotdict(
            scan=6,
            speed=20,
            radius=15,
            pitch=0.080,
            int_dist=0.007,
    )

    PARAMETERS_TC = dotdict(
            length=1.0,
            nboxz=4,
            deltaz=0.0015,
            h_box=0.075,
            base_folder=r'',
            y_min=-0.1,
            y_max=19 * PARAMETERS_WG['pitch'] + 0.1
    )

    # Calculations
    coup = [Waveguide(PARAMETERS_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035]).sin_acc((-1) ** i * wg.dy_bend)
        x_mid = wg.x[-1]
        wg.sin_acc((-1) ** i * wg.dy_bend).end()

    PARAMETERS_TC.x_center = x_mid
    trench_col = TrenchColumn(PARAMETERS_TC)
    trench_col.get_trench(coup)

    fig, ax = plt.subplots()
    for wg in coup:
        ax.plot(wg.x[:-1], wg.y[:-1], 'b')
    for t in trench_col:
        ax.add_patch(t.patch)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    _example()
