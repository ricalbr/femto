from __future__ import annotations

import copy
import math
import pathlib
from functools import cached_property
from typing import Any
from typing import Generator
from typing import Iterator

import attrs
import dill
import largestinteriorrectangle as lir
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.curves import sin
from femto.helpers import almost_equal
from femto.helpers import flatten
from femto.helpers import listcast
from femto.helpers import normalize_polygon
from femto.waveguide import Waveguide
from shapely import geometry
from shapely.ops import unary_union

# Define array type
nparray = npt.NDArray[np.float64]


class Trench:
    """Class that represents a trench block and provides methods to compute the toolpath of the block."""

    def __init__(
        self,
        block: geometry.Polygon,
        delta_floor: float = 0.001,
        height: float = 0.300,
        safe_inner_turns: int = 5,
        step: float | None = None,
    ) -> None:
        self.block: geometry.Polygon = block  #: Polygon shape of the trench.
        self.delta_floor: float = delta_floor  #: Offset distance between buffered polygons in the trench toolpath.
        self.height: float = height  #: Depth of the trench box.
        self.safe_inner_turns: int = safe_inner_turns  #: Number of spiral turns before zig-zag filling.
        self.step: float | None = step  #: Step between adjacent points in the trench toolpath, [mm/s].

        self._floor_length: float = 0.0  #: Length of the floor path.
        self._wall_length: float = 0.0  #: Length of the wall path.

        self._id: str = 'TR'  #: Trench identifier.

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return almost_equal(self.block, other.block)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] < other.yborder[0])

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] <= other.yborder[0])

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] > other.yborder[0])

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Trench):
            logger.error(f'Trying comparing Trench with {other.__class__.__name__}')
            raise TypeError(f'Trying comparing Trench with {other.__class__.__name__}')
        return bool(self.yborder[0] >= other.yborder[0])

    @property
    def id(self) -> str:
        """Object ID.

        The property returns the ID of a given object.

        Returns
        -------
        str
            The ID of the object.
        """
        return self._id

    @property
    def border(self) -> tuple[nparray, nparray]:
        """Border of the trench.

        It returns the border of the block as a tuple of two numpy arrays, one for the `x` coordinates and one for
        the `y` coordinates.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            `x` and `y`-coordinates arrays of the trench border.
        """
        xx, yy = self.block.exterior.coords.xy
        rx, ry = self.resample_polygon(
            x=np.asarray(xx, dtype=np.float64),
            y=np.asarray(yy, dtype=np.float64),
            step=self.step,
        )
        logger.debug('Extracting (x,y) from trench block.')
        return np.asarray(rx, dtype=np.float64), np.asarray(ry, dtype=np.float64)

    @property
    def xborder(self) -> nparray:
        """`x`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `x`-coordinates arrays of the trench border.
        """
        x, _ = self.border
        logger.debug('Return x-coordinate of the border.')
        return x

    @property
    def yborder(self) -> nparray:
        """`y`-coordinates of the trench border.

        Returns
        -------
        numpy.ndarray
            `y`-coordinates arrays of the trench border.
        """
        _, y = self.border
        logger.debug('Return y-coordinate of the border.')
        return y

    @property
    def xmin(self) -> float:
        """Minimum `x` value of the trench boundary.

        Returns
        -------
        float
            Minimum `x` value of the block border.
        """
        logger.debug('Return minimum x-value for trench block.')
        return float(self.block.bounds[0])

    @property
    def ymin(self) -> float:
        """Minimum `y` value of the trench boundary.

        Returns
        -------
        float
            Minimum `y` value of the block border.
        """
        logger.debug('Return minimum y-value for trench block.')
        return float(self.block.bounds[1])

    @property
    def xmax(self) -> float:
        """Maximum `x` value of the trench boundary.

        Returns
        -------
        float
            Maximum `x` value of the block border.
        """
        logger.debug('Return maximun x-value for trench block.')
        return float(self.block.bounds[2])

    @property
    def ymax(self) -> float:
        """Maximum `y` value of the trench boundary.

        Returns
        -------
        float
            Maximum `y` value of the block border.
        """
        logger.debug('Return maximun y-value for trench block.')
        return float(self.block.bounds[3])

    @property
    def center(self) -> tuple[float, float]:
        """Baricenter of the trench block.

        Returns
        -------
        tuple(float, float)
            `x` and `y` coordinates of the centroid of the block.
        """
        logger.debug('Return block center point.')
        return self.block.centroid.x, self.block.centroid.y

    @property
    def floor_length(self) -> float:
        """Total length of the floor path."""
        logger.debug(f'Total length of the floor path {self._floor_length} mm.')
        return self._floor_length

    @property
    def wall_length(self) -> float:
        """Length of a single layer of the wall path."""
        logger.debug(f'Total length of the wall path {self._wall_length} mm.')
        return self._wall_length

    @cached_property
    def orientation(self) -> str:
        """Orientation of the trench block."""
        (xmin, ymin, xmax, ymax) = self.block.bounds
        if (xmax - xmin) <= (ymax - ymin):
            logger.debug('The block orientation is vertical.')
            return 'v'
        else:
            logger.debug('The block orientation is horizontal.')
            return 'h'

    @cached_property
    def num_insets(self) -> int:
        """Number of spiral turns."""
        if self.block.contains(self.block.convex_hull.buffer(-0.01 * self.delta_floor)):
            logger.debug(f'The number of spiral turns is {self.safe_inner_turns}.')
            return self.safe_inner_turns
        else:
            # External rectangle
            (xmin_ext, ymin_ext, xmax_ext, ymax_ext) = self.block.bounds

            # Internal rectangle
            buffer_length = 2 * self.delta_floor
            p = np.array([[[x, y] for (x, y) in self.block.buffer(-buffer_length).exterior.coords]], np.float64) * 1e3
            xmin_int, ymin_int, dx_int, dy_int = lir.lir(p.astype(np.int32), np.int32) / 1e3
            xmax_int, ymax_int = xmin_int + dx_int, ymin_int + dy_int

            if self.orientation == 'h':
                d_upper = np.abs(ymax_ext - ymax_int)
                d_lower = np.abs(ymin_ext - ymin_int)
            else:
                d_upper = np.abs(xmax_ext - xmax_int)
                d_lower = np.abs(xmin_ext - xmin_int)

            # Distinguish concave / biconcave
            if d_upper <= buffer_length or d_lower <= buffer_length:
                n_turns = math.ceil((d_upper + d_lower) / self.delta_floor) + self.safe_inner_turns
            else:
                n_turns = math.ceil((d_upper + d_lower) / (2 * self.delta_floor)) + self.safe_inner_turns
            logger.debug(f'The number of spiral turns is {n_turns}.')
            return int(n_turns)

    def zigzag_mask(self) -> geometry.MultiLineString:
        """Zig-zag mask.

        The function returns a Shapely geometry (MultiLineString, or more rarely, GeometryCollection) for a simple
        hatched rectangle. The spacing between the lines is given by ``self.delta_floor`` while the rectangle is the
        minimum rotated rectangle containing the trench block.

        The lines of the zig-zag pattern are along the longest dimension of the rectangular envelope of the trench
        block.

        Returns
        -------
        shapely.MultiLineString | shapely.GeometryCollection
            Collection of hatch lines (in case a hatch line intersects with the corner of the clipping
            rectangle, which produces a point along with the usual lines).
        """

        (xmin, ymin, xmax, ymax) = self.block.bounds
        number_of_lines = 2 + int((xmax - xmin) / self.delta_floor)

        coords = []
        logger.debug('Create rectangular zig-zag MultiString line pattern.')
        if self.orientation == 'v':
            # Vertical hatching
            for i in range(0, number_of_lines, 2):
                coords.extend([((xmin + i * self.delta_floor, ymin), (xmin + i * self.delta_floor, ymax))])
                coords.extend([((xmin + (i + 1) * self.delta_floor, ymax), (xmin + (i + 1) * self.delta_floor, ymin))])
        else:
            # Horizontal hatching
            for i in range(0, number_of_lines, 2):
                coords.extend([((xmin, ymin + i * self.delta_floor), (xmax, ymin + i * self.delta_floor))])
                coords.extend([((xmax, ymin + (i + 1) * self.delta_floor), (xmin, ymin + (i + 1) * self.delta_floor))])
        return geometry.MultiLineString(coords)

    def zigzag(self, poly: geometry.Polygon) -> nparray:
        """Zig-zag filling pattern.

        The function `zigzag` takes a polygon as input, applies a zig-zag filling pattern to it, and returns the
        coordinates of the resulting zigzag pattern.

        Parameters
        ----------
        poly: geometry.Polygon
            The parameter `poly` is of type `geometry.Polygon`. It represents a polygon object that you want to
            apply the zig-zag filling pattern to.

        Returns
        -------
        numpy.ndarray
            Coordinates array of the zig-zag filling pattern.
        """
        mask = self.zigzag_mask()
        path = poly.intersection(mask)
        logger.debug('Intersect rectangular zig-zag line pattern with trench block.')

        # Extract single lines (distinguish between LineString and MultiLineString)
        if isinstance(path, geometry.LineString):
            lines = [path]
        elif isinstance(path, (geometry.MultiLineString, geometry.GeometryCollection)):
            lines = [x for x in path.geoms if isinstance(x, geometry.LineString)]
        else:
            lines = []

        coords = []
        for line in lines:
            self._floor_length += line.length + self.delta_floor
            coords.extend(line.coords)
        return np.array(coords).T

    def toolpath(self) -> Generator[nparray]:
        """Toolpath generator.

        The function takes a polygon and computes the filling toolpath.
        Such path is obtained with two strategy:
        -   First, the outer border is added to the ``polygon_list``. The functions pops polygon objects from this list,
            buffers it, and yields the exterior coordinates of the buffered polygon.
            Before yielding, the new polygon is added to the list as the buffered inset will be computed in the next
            iteration. If the buffering operation returns polygons composed of different non-touching parts (`i.e.`
            ``MultiPolygon``), each part is added to the list as a single ``Polygon`` object.
            If no inset can be computed from the starting polygon, no object is added to the list. The generator
            terminates when no more buffered polygons can be computed.

        -   Second, after a number of insets the center of the trench floor is filled with a zig-zag pattern to avoid
            harsh acceleration and small displacements.
            The zig-zag hatching pattern is computed for each polygon in the ``polygon_list``.

        Yields
        ------
        numpy.ndarray
            (`x`, `y`) coordinates of each of the buffered polygons.

        See Also
        --------
        geometry.Polygon : shapely polygon object.
        geometry.Multipolygon : collections of shapely polygon objects.
        """

        self._wall_length = self.block.length
        polygon_list = [self.block]

        for _ in range(self.num_insets):
            current_poly = polygon_list.pop(0)
            if not current_poly.is_empty:
                polygon_list.extend(self.buffer_polygon(current_poly, offset=-np.fabs(self.delta_floor)))
                self._floor_length += current_poly.length
                logger.debug('Yield buffered contour path.')
                x, y = np.array(current_poly.exterior.coords).T
                yield self.resample_polygon(x=x, y=y, step=self.step)

        for poly in polygon_list:
            logger.debug('Yield inner zig-zag path.')
            yield self.zigzag(poly.buffer(1.05 * self.delta_floor))

    @staticmethod
    def buffer_polygon(shape: geometry.Polygon, offset: float) -> list[geometry.Polygon]:
        """Buffer a polygon.

        It takes a polygon and returns a list of polygons that are offset by a given distance.

        Parameters
        ----------
        shape : geometry.Polygon
            Shape of the trench block to buffer.
        offset : float
            The offset to buffer the polygon by [mm].

        Returns
        -------
        list(geometry.Polygon)
            List of buffered polygons. If the buffered polygon is still a ``Polyon`` object the list contains just a
            single polygon. If the buffered polygon is ``MultiPolygon``, the list contais all the single ``Polygon``
            objects that compose the multipolygon. Finally, if the buffered polygon cannot be computed the list
            contains just the empty polygon ``Polygon()``.

        Notes
        -----
        The buffer operation returns a polygonal result. The new polygon is checked for validity using
        ``obj.is_valid`` in the sense of [#]_.

        For a reference, read the buffer operations `here
        <https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods>`_

        .. [#] John R. Herring, Ed., “OpenGIS Implementation Specification for Geographic information - Simple
            feature access - Part 1: Common architecture,” Oct. 2006

        See Also
        --------
        geometry.Polygon.buffer : buffer operations on ``Polygon`` objects.
        geometry.Polygon : shapely polygon object.
        geometry.Multipolygon : collections of shapely polygon objects.
        """

        if not shape.is_valid:
            return [geometry.Polygon()]

        # split single polygon components
        if isinstance(shape, geometry.MultiPolygon):
            parts = shape.geoms
        else:
            parts = [shape]

        out = []
        for part in parts:
            # buffer each single part separately
            buff = part.buffer(offset).simplify(
                tolerance=1e-5,
                preserve_topology=True,
            )

            if isinstance(buff, geometry.Polygon) and not buff.is_empty:
                out.append(buff)
            elif isinstance(buff, geometry.MultiPolygon) and not buff.is_empty:
                out.extend(list(buff.geoms))
            else:
                out.append(geometry.Polygon())  # collapsed or non-area geometry
        return out

    @staticmethod
    def resample_polygon(x: nparray, y: nparray, step: float | None = 0.005) -> nparray:
        """Resample a polygon border by a specified number of points.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            x-coordinate of the polygon border.
        y : npt.NDArray[np.float64]
            y-coordinate of the polygon border.
        step : float
            Step between two adjacent points, [mm]. Default value is 0.005 mm.

        Returns
        -------
        npt.NDArray[np.float64]
            xy-coordinates of the re-sampled polygon border.
        """
        if step is None:
            return np.array([x, y]).astype(np.float64)

        # Cumulative Euclidean distance between successive polygon points. This will be later be used for interpolation
        xy = np.stack([x, y], axis=1)
        d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

        # Get linearly spaced points along the cumulative Euclidean distance
        num_points = int(d.max() / step + 1)
        d_sampled = np.linspace(0, d.max(), num_points)

        # Interpolate x and y coordinates
        return np.array([np.interp(d_sampled, d, x).astype(np.float64), np.interp(d_sampled, d, y).astype(np.float64)])


@attrs.define(kw_only=True, repr=False, init=False)
class TrenchColumn:
    """Class representing a column of isolation trenches."""

    x_center: float  #: Center of the trench blocks, [mm].
    y_min: float  #: Minimum `y` coordinates of the trench blocks, [mm].
    y_max: float  #: Maximum `y` coordinates of the trench blocks, [mm].
    bridge: float = 0.026  #: Separation length between nearby trench blocks, [mm].
    length: float = 1  #: Lenght of the trench along the `x` axis, [mm].
    h_box: float = 0.075  #: Height of the single trench box, [mm].
    nboxz: int = 4  #: Number of stacked box along the `z` axis.
    z_off: float = -0.020  #: Starting offset in `z` with respect to the sample's surface, [mm].
    deltaz: float = 0.0015  #: Offset distance between countors paths of the trench wall, [mm].
    delta_floor: float = 0.001  #: Offset distance between buffered polygons in the trench toolpath [mm].
    n_pillars: int | None = None  #: number of sustaining pillars.
    pillar_width: float = 0.040  #: width of the pillars.
    safe_inner_turns: int = 5  #: Number of spiral turns befor zig-zag filling
    u: list[float] = attrs.field(
        factory=list
    )  #: List of U coordinate to change irradiation power automatically, [deg].
    speed_wall: float = 4.0  #: Translation speed of the wall section, [mm/s].
    speed_floor: float = attrs.field(factory=float)  #: Translation speed of the floor section, [mm/s].
    speed_closed: float = 5.0  #: Translation speed with closed shutter, [mm/s].
    speed_pos: float = 2.0  #: Positioning speed with closed shutter, [mm/s].
    base_folder: str = ''  #: Location where PGM files are stored in lab PC. If empty, load files with relative path.
    beam_waist: float = 0.004  #: Diameter of the laser beam-waist, [mm].
    round_corner: float = 0.010  #: Radius of the blocks round corners, [mm].
    step: float | None = None  #: Step between adjacent points in the trench toolpath, [mm/s].

    _id: str = attrs.field(alias='_id', default='TC')  #: TrenchColumn ID.
    _trench_list: list[Trench] = attrs.field(alias='_trench_list', factory=list)  #: List of trench objects.
    _trenchbed: list[Trench] = attrs.field(alias='_trenchbed', factory=list)  #: List of beds blocks.

    CWD: pathlib.Path = attrs.field(default=pathlib.Path.cwd())  #: Current working directory

    def __init__(self, **kwargs: Any) -> None:
        filtered: dict[str, Any] = {
            att.name: kwargs[att.name]
            for att in self.__attrs_attrs__  # type: ignore[attr-defined]
            if att.name in kwargs
        }
        self.__attrs_init__(**filtered)  # type: ignore[attr-defined]

    def __attrs_post_init__(self) -> None:
        if not self.speed_floor:
            self.speed_floor = self.speed_wall
            logger.debug(f'Floor speed is set to {self.speed_floor} mm/s.')

    def __iter__(self) -> Iterator[Trench]:
        """Iterator that yields single trench blocks of the column.

        Yields
        ------
        Trench
            Single trench block of the TrenchColumn.
        """
        return iter(self._trench_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @classmethod
    def from_dict(cls: type[TrenchColumn], param: dict[str, Any], **kwargs: Any | None) -> TrenchColumn:
        """Create an instance of the class from a dictionary.

        It takes a class and a dictionary, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        param: dict()
            Dictionary mapping values to class attributes.
        kwargs: optional
            Series of keyword arguments that will be used to update the param file before the instantiation of the
            class.

        Returns
        -------
        Instance of class.
        """
        # Update parameters with kwargs
        p = copy.deepcopy(param)
        if kwargs:
            p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

    @classmethod
    def load(cls: type[TrenchColumn], pickle_file: str) -> TrenchColumn:
        """Create an instance of the class from a pickle file.

        The load function takes a class and a pickle file name, and returns an instance of the class with the
        dictionary's keys as the instance's attributes.

        Parameters
        ----------
        pickle_file: str
            Filename of the pickle_file.

        Returns
        -------
        Instance of class.
        """

        logger.info(f'Load {cls.__name__} object from pickle file.')
        with open(pickle_file, 'rb') as f:
            tmp = dill.load(f)
            logger.debug(f'{f} file loaded.')
        return cls.from_dict(tmp)

    @property
    def id(self) -> str:
        """Object ID.

        The property returns the ID of a given object.

        Returns
        -------
        str
            The ID of the object.
        """
        return self._id

    @property
    def trench_list(self) -> list[Trench]:
        """List of Trench objects.

        Returns
        -------
        list[Trench]
            A list of trenches.
        """
        return self._trench_list

    @property
    def bed_list(self) -> list[Trench]:
        """Trench bed list.

        Returns
        -------
        list[Trench]
            List of Trench objects that constitute the "bed" under the waveguides of the U-Trench structure.
        """
        return self._trenchbed

    @property
    def adj_bridge(self) -> float:
        """Bridge length adjusted for the laser beam waist.

        Returns
        -------
        float
            Adjustted bridge size considering the size of the laser focus [mm].
        """
        adj_b = self.bridge / 2 + self.beam_waist + self.round_corner
        logger.debug(f'The beidge size adjusted for the beam size is {adj_b} mm.')
        return adj_b

    @property
    def adj_pillar_width(self) -> float:
        """Pillar size adjusted for the laser beam waist.

        Returns
        -------
        float
            Adjustted pillar size considering the size of the laser focus [mm].
        """
        adj_p = self.pillar_width / 2 + self.beam_waist
        logger.debug(f'The adjusted pillar size is {adj_p} mm.')
        return adj_p

    @property
    def n_repeat(self) -> int:
        """Number of laser passes required to cover the vertical height of the trench box.

        Returns
        -------
        int
            The number of times the border path is repeated in the `z` direction.
        """
        n_repeat = int(abs(math.ceil((self.h_box - self.z_off) / self.deltaz)))
        logger.debug(f'The number of laser vertical trench layers is {n_repeat}.')
        return n_repeat

    @property
    def fabrication_time(self) -> float:
        """Total fabrication time.

        The fabrication time is the sum of the lengths of all the walls, floors and bedfloors of all the trenches,
        divided by the translation speed.

        Returns
        -------
        float
            Total fabrication time [s].
        """
        t_box: float = np.sum(
            [
                self.nboxz * (self.n_repeat * t.wall_length / self.speed_wall + t.floor_length / self.speed_floor)
                for t in self._trench_list
            ]
        )
        t_bed: float = np.sum([b.floor_length / self.speed_floor for b in self._trenchbed])
        fab_time: float = t_box + t_bed
        logger.debug(f'The total fabrication time for the trench column is {fab_time} s.')
        return fab_time

    @property
    def total_height(self) -> float:
        """Total trench height.

        Returns
        -------
        float
            Total trench height [um].
        """
        h_tot = float(self.nboxz * self.h_box)
        logger.debug(f'The total height of the trench block is {h_tot} mm.')
        return h_tot

    @property
    def rect(self) -> geometry.Polygon:
        """Area of the trench column.

        The rectangular box is centered in ``x_c`` along the `x` axis, while the `y`-borders are ``y_min`` and
        ``y_max``. ::

            ┌─────┐  ► y_max
            │     │
            │     │
            │     │
            └─────┘  ► y_min
               ▲
               x_c

        Returns
        -------
        geometry.box
            Rectangular box polygon.
        """
        if not self.length:
            return geometry.Polygon()
        logger.debug(f'Return rectangle of sides {self.y_max - self.y_min} mm and {self.length} mm.')
        return geometry.box(self.x_center - self.length / 2, self.y_min, self.x_center + self.length / 2, self.y_max)

    def define_trench_bed(self, n_pillars: int) -> None:
        """Trenchbed shape.

        This method is used to calculate the shape of the plane beneath the column of trenches based on a list of
        trenches.

        The trench bed is defined as a rectangular-ish shape with the top and bottom part with the same shape of the
        top and bottom trench (respectively) of the column of trenches.
        This trench bed can be divided into several `beds` divided by structural pillars of a given `x`-width. The
        width and the number of the pillars can be defined by the user when the `TrenchColumn`` object is created.

        This method populated the ``self._trenchbed`` attribute of the ``TrenchColumn`` object.

        Parameters
        ----------
        n_pillars: int
            Number of pillars

        Returns
        -------
        None.
        """
        if not self._trench_list:
            logger.critical('No trench is present. Trenchbed cannot be defined.')
            return None

        # Define the trench bed shape as union of a rectangle and the first and last trench of the column.
        # Automatically convert the bed polygon to a MultiPolygon to keep the code flexible to word with the
        # no-pillar case.
        logger.debug(f'Divide bed in {n_pillars + 1} parts.')
        t1, t2 = self._trench_list[0], self._trench_list[-1]
        tmp_rect = geometry.box(t1.xmin, t1.center[1], t2.xmax, t2.center[1])

        unioned = unary_union([t1.block, t2.block, tmp_rect])
        if isinstance(unioned, geometry.Polygon):
            tmp_bed_polygons = [unioned]
        elif isinstance(unioned, geometry.MultiPolygon):
            tmp_bed_polygons = list(unioned.geoms)
        elif isinstance(unioned, geometry.GeometryCollection):
            # extract polygons
            tmp_bed_polygons = [g for g in unioned.geoms if isinstance(g, geometry.Polygon)]
        else:
            # fallback for pathological cases
            tmp_bed_polygons = [geometry.Polygon()]

        tmp_bed = geometry.MultiPolygon(tmp_bed_polygons)

        # Add pillars and define the bed layer as a Trench object
        xmin, ymin, xmax, ymax = tmp_bed.bounds
        x_pillars = np.linspace(xmin, xmax, n_pillars + 2)[1:-1]
        for x in x_pillars:
            tmp_pillar = geometry.LineString([[x, ymin], [x, ymax]]).buffer(self.adj_pillar_width)
            tmp_bed = tmp_bed.difference(tmp_pillar)

        # Extract polygons
        polygons = []
        if isinstance(tmp_bed, geometry.Polygon):
            polygons = [tmp_bed]
        elif isinstance(tmp_bed, geometry.MultiPolygon):
            polygons = list(tmp_bed.geoms)

        # Add bed blocks
        logger.debug('Add trench beds as trench objects with height = 0.015 mm.')
        self._trenchbed = [
            Trench(
                block=normalize_polygon(p.buffer(-self.round_corner).buffer(self.round_corner)),
                height=0.015,
                delta_floor=self.delta_floor,
                safe_inner_turns=self.safe_inner_turns,
                step=self.step,
            )
            for p in polygons
        ]
        return None

    def dig_from_waveguide(
        self,
        waveguides: list[Waveguide],
        remove: list[int] | None = None,
    ) -> None:
        """Dig trenches from waveguide input.

        The function uses a list of ``Waveguide`` objects as a mold to define the trench shapes. It populates
        `self.trech_list` with ``Trench`` objects.
        If some of the generated trenches are not needed they can be removed from the list is a ``remove`` list of
        indeces is given as input. Trenches are numbered such that the one with lowest `y` coordinate has index 0,
        the one with second-lowest `y` coordinate has index 1 and so on. If ``remove`` is empty or ``None`` all the
        generated trenches are added to the `self.trench_list`.

        Parameters
        ----------
        waveguides : list(Waveguide)
            List of ``Waveguide`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None.
        """

        if not all(isinstance(wg, Waveguide) for wg in waveguides):
            logger.debug(f'All the input objects must be of type Waveguide.\nGiven {[type(wg) for wg in waveguides]}')
            raise ValueError(
                f'All the input objects must be of type Waveguide.\nGiven {[type(wg) for wg in waveguides]}'
            )

        coords = []
        for wg in waveguides:
            x, y = wg.path
            coords.extend([list(zip(x, y))])
        self._dig(coords, remove)

    def dig_from_array(
        self,
        waveguides: list[nparray],
        remove: list[int] | None = None,
    ) -> None:
        """Dig trenches from array-like input.

        The function uses a list of `array-like` objects as a mold to define the trench shapes. It populates
        `self.trech_list` with ``Trench`` objects.
        If some of the generated trenches are not needed they can be removed from the list is a ``remove`` list of
        indeces is given as input. Trenches are numbered such that the one with lowest `y` coordinate has index 0,
        the one with second-lowest `y` coordinate has index 1 and so on. If ``remove`` is empty or ``None`` all the
        generated trenches are added to the `self.trench_list`.

        Parameters
        ----------
        waveguides : list(numpy.ndarray)
            List of ``numpy.ndarray`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None.
        """
        if not all(isinstance(wg, np.ndarray) for wg in waveguides):
            logger.debug(f'All the input objects must be numpy arrays. Given {[type(wg) for wg in waveguides]}')
            raise ValueError(f'All the input objects must be numpy arrays. Given {[type(wg) for wg in waveguides]}')

        coords = []
        for wg in waveguides:
            x, y = wg.T if wg.shape[1] == 2 else wg
            coords.extend([list(zip(x, y))])
        self._dig(coords, remove)

    def _dig(
        self,
        coords_list: list[list[tuple[float, float]]],
        remove: list[int] | None = None,
    ) -> None:
        """Compute the trench blocks from the waveguide of the optical circuit.

        Trench blocks shapes are defined using a list of paths (``coords_list``) as mold matrix.
        The waveguides are converted to ``LineString`` and buffered to be as large as the adjusted bridge width.

        Using polygon difference operation, the rectangular area of the ``TrenchColumn`` is cut obtaining a
        ``MultiPolygon`` made of all the trench blocks.

        All the blocks are then treated individually. Each block is buffered to obtain an outset polygon with
        rounded corners a ``Trench`` object is created with the new polygon box and appended to the ``trench_list``,
        if their index is not present in the ``remove`` list.

        Parameters
        ----------
        coords_list : list(list(tuple(float, float)))
            List of ``numpy.ndarray`` objects that will be used as a mold to define trench shapes.
        remove : list[int], optional
            List of indides of trench to be removed from the ``TrenchColumn``.

        Returns
        -------
        None.
        """
        if remove is None:
            remove = []

        trench_blocks = self.rect
        for coords in coords_list:
            dilated = geometry.LineString(coords).buffer(self.adj_bridge, cap_style='round')
            trench_blocks = trench_blocks.difference(dilated)

        # if coordinates are empty or coordinates do not intersect the trench column rectangle box
        if isinstance(trench_blocks, geometry.Polygon) and almost_equal(trench_blocks, self.rect, tol=1e-8):
            logger.critical('No trench found intersecting waveguides with trench area.\n')
            return None

        # Extract trench blocks
        logger.debug('Trenches found.')
        blocks = []
        if isinstance(trench_blocks, geometry.Polygon):
            blocks = [trench_blocks]
        elif isinstance(trench_blocks, geometry.MultiPolygon):
            blocks = list(trench_blocks.geoms)
        for block in listcast(sorted(blocks, key=Trench)):
            # buffer to round corners
            block = block.buffer(self.round_corner, resolution=256, cap_style='round')
            # simplify the shape to avoid path too much dense of points
            block = block.simplify(tolerance=5e-7, preserve_topology=True)
            self._trench_list.append(
                Trench(
                    normalize_polygon(block),
                    delta_floor=self.delta_floor,
                    safe_inner_turns=self.safe_inner_turns,
                    step=self.step,
                )
            )
        logger.debug('Finished append trenches.')

        for index in sorted(listcast(remove), reverse=True):
            del self._trench_list[index]

        if self.n_pillars is not None:
            self.define_trench_bed(int(self.n_pillars))


def main() -> None:
    """The main function of the script."""
    from addict import Dict as ddict

    # Data
    param_wg = ddict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    param_tc = ddict(length=1.0, base_folder='', y_min=-0.1, y_max=19 * param_wg['pitch'] + 0.1, u=[30.339, 32.825])

    # Calculations
    x_c = 0
    coup = [Waveguide(**param_wg) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_c = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()

    # Trench
    utc = TrenchColumn(x_center=x_c, n_pillars=3, **param_tc)
    utc.dig_from_waveguide(flatten([coup]))

    import matplotlib.pyplot as plt

    # b = T._trench_list[0]
    # b = T._trenchbed[0]
    for tr in utc.trench_list:
        for x, y in tr.toolpath():
            plt.plot(x, y)

    plt.axis('equal')


if __name__ == '__main__':
    main()
