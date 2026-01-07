from __future__ import annotations

import copy
import pathlib
from typing import Any
from typing import Sequence

import attrs
import dill
import numpy as np
import numpy.typing as npt
from femto import logger
from femto.helpers import unique_filter

# Define array type
nparray = npt.NDArray[np.float64]


@attrs.define(kw_only=True, repr=False, init=False)
class LaserPath:
    """Class that computes and stores the coordinates of a laser path."""

    radius: float = 15  #: Curvature radius
    scan: int = attrs.field(validator=attrs.validators.instance_of(int), default=1)  #: Number of overlapped scans.
    speed: float = 1.0  #: Opened shutter translation speed `[mm/s]`.
    samplesize: tuple[float, float] = (100, 50)  #: Dimensions of the sample (x `[mm]`, y `[mm]`).
    x_init: float = -2.0  #: Initial x-coordinate for the laser path `[mm]`.
    y_init: float = 0.0  #: Initial y-coordinate for the laser path `[mm]`
    z_init: float = attrs.field(default=float('nan'))  #: Initial z-coordinate for the laser path `[mm]`.
    shrink_correction_factor: float = 1.0  #: Correcting factor for glass shrinking.
    lsafe: float = 2.0  #: Safe margin length `[mm]`.
    speed_closed: float = 5.0  #: Closed shutter translation speed `[mm/s]`.
    speed_pos: float = 0.5  #: Positioning speed (shutter closed)`[mm/s]`.
    cmd_rate_max: float = 1200.0  #: Maximum command rate `[cmd/s]`.
    acc_max: float = 500.0  #: Maximum acceleration/deceleration `[m/s^2]`.
    end_off_sample: bool = True  #: Flag to end laserpath off of the sample. (See `x_end`).
    warp_flag: bool = False  #: Flag to toggle the glass warp compensation.
    metadata: dict[str, str] = attrs.field(factory=dict)  #: Dictionary with laserpath metadata.

    _id: str = attrs.field(alias='_id', default='LP')
    _x: nparray = attrs.field(alias='_x', factory=lambda: np.array([], dtype=np.float64))
    _y: nparray = attrs.field(alias='_y', factory=lambda: np.array([], dtype=np.float64))
    _z: nparray = attrs.field(alias='_z', factory=lambda: np.array([], dtype=np.float64))
    _f: nparray = attrs.field(alias='_f', factory=lambda: np.array([], dtype=np.float64))
    _s: nparray = attrs.field(alias='_s', factory=lambda: np.array([], dtype=np.float64))
    logger.debug('Initialized all the coordinates arrays.')

    def __init__(self, **kwargs: Any) -> None:
        filtered: dict[str, Any] = {
            att.name: kwargs[att.name]
            for att in self.__attrs_attrs__  # type: ignore[attr-defined]
            if att.name in kwargs
        }
        self.__attrs_init__(**filtered)  # type: ignore[attr-defined]

    def __attrs_post_init__(self) -> None:
        if 'name' not in self.metadata.keys():
            self.metadata['name'] = type(self).__name__

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @classmethod
    def from_dict(cls: type[LaserPath], param: dict[str, Any], **kwargs: Any | None) -> LaserPath:
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
    def load(cls: type[LaserPath], pickle_file: str | pathlib.Path) -> LaserPath:
        """Create an instance of the class from a pickle file.

        It takes a class and a pickle file name, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        pickle_file: str or pathlib.Path
            Filename of the pickle_file.

        Returns
        -------
        Instance of class.
        """
        logger.info(f'Load {cls.__name__} object from pickle file.')
        with open(pickle_file, 'rb') as f:
            tmp = dill.load(f)
            logger.debug(f'Load {f} file.')
        return cls.from_dict(tmp)

    @property
    def init_point(self) -> tuple[float, float, float]:
        """Initial point of the trajectory.

        Returns
        -------
        tuple(float, float, float)
            `[x0, y0, z0]` coordinates.
        """

        z0 = self.z_init if self.z_init is not None else 0.0
        logger.debug(f'Return init point: ({self.x_init}, {self.y_init}, {z0}).')
        return self.x_init, self.y_init, z0

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
    def lvelo(self) -> float:
        """Compute the length needed to reach the translation speed.

        The length needed to reach the writing speed is computed as 3 times the length needed to accelerate from
        0 to the translation speed.

        Returns
        -------
        float
            Length needed to accelerate to translation speed [mm].
        """

        logger.debug(f'Return translation lvelo = {3 * (0.5 * self.speed**2 / self.acc_max)}.')
        return 3 * (0.5 * self.speed**2 / self.acc_max)

    @property
    def dl(self) -> float:
        """Compute the minimum spatial separation between two points.

        The minimum spatial separation between two points is the speed divided by the maximum command rate.

        Returns
        -------
        float
            The minimum separation between two points [mm].
        """

        logger.debug(f'Return minimum spatial separation between two points dl = {self.speed / self.cmd_rate_max}.')
        return self.speed / self.cmd_rate_max

    @property
    def x_end(self) -> float | None:
        """Compute the `x` coordinate of the laserpth outside the sample, if the sample size is not None.

        Returns
        -------
        float, optional
            The end of the laser path outside the sample [mm].
        """

        if not self.samplesize[0]:
            logger.debug('Return None, check samplesize.')
            logger.error(f'Error, sample size is {self.samplesize[0]}.')
            return None
        if self.end_off_sample:
            logger.debug(f'Return x_end = {self.samplesize[0] + self.lsafe}.')
            return self.samplesize[0] + self.lsafe
        else:
            logger.debug(f'Return x_end = {self.samplesize[0] - self.lsafe}.')
            return self.samplesize[0] - self.lsafe

    @property
    def points(self) -> nparray:
        """Matrix of the unique points in the trajectory.

        The matrix of points is parsed through a unique functions that removes all the subsequent identical points in
        the set.

        See Also
        --------
        femto.helpers.unique_filter : Filter trajectory points to remove subsequent identical points.

        Returns
        -------
        numpy.ndarray
            `[X, Y, Z, F, S]` points of the laser trajectory.
        """
        logger.debug('Return matrix with filtered coordinate. See femto.helpers.unique_filter().')
        return np.array(unique_filter([self._x, self._y, self._z, self._f, self._s]))

    @property
    def x(self) -> nparray:
        """`x`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        See Also
        --------
        femto.helpers.unique_filter : Filter trajectory points to remove subsequent identical points.

        Returns
        -------
        numpy.ndarray
            The `x`-coordinates of the points in the laser path.
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            logger.debug('Return x-coordinates.')
            return np.array(coords[0])
        logger.debug('x-coordinate array is empty. Return empty array.')
        return np.array([])

    @property
    def lastx(self) -> float:
        """Last `x` value in the trajectory points matrix, if any.

        Returns
        -------
        float
            The last value of the `x` array.
        """

        arrx = self.x
        logger.debug(f'Return x-coordinate of last point: {float(arrx[-1])}.')
        return float(arrx[-1])

    @property
    def y(self) -> nparray:
        """`y`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            The `y`-coordinates of the points in the laser path.

        See Also
        --------
        unique_filter : Filter trajectory points to remove subsequent identical points.
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            logger.debug('Return y-coordinates.')
            return np.array(coords[1])
        logger.debug('y-coordinate array is empty. Return empty array.')
        return np.array([])

    @property
    def lasty(self) -> float:
        """Last `y` value in the trajectory points matrix, if any.

        Returns
        -------
        float
            The last value of the `y` array.
        """

        arry = self.y
        logger.debug(f'Return y-coordinate of last point: {float(arry[-1])}.')
        return float(arry[-1])

    @property
    def z(self) -> nparray:
        """`z`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            The `z`-coordinates of the points in the laser path.

        See Also
        --------
        unique_filter: Filter trajectory points to remove subsequent identical points.
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            logger.debug('Return z-coordinates.')
            return np.array(coords[2])
        logger.debug('z-coordinate array is empty. Return empty array.')
        return np.array([])

    @property
    def lastz(self) -> float:
        """Last `z` value in the trajectory points matrix, if any.

        Returns
        -------
        float
            The last value of the `z` array.
        """

        arrz = self.z
        logger.debug(f'Return z-coordinate of last point: {float(arrz[-1])}.')
        return float(arrz[-1])

    @property
    def lastpt(self) -> nparray:
        """Last point of the laser path, if any.

        Returns
        -------
        numpy.ndarray
            Final `[x, y, z]` pointof the laser path.
        """

        if self._x.size > 0:
            logger.debug(f'Return last point ({self._x[-1]}, {self._y[-1]}, {self._z[-1]}).')
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        logger.debug('No points present, return empty array.')
        return np.array([])

    @property
    def path(self) -> tuple[nparray, nparray]:
        """List of `x` and `y`-coordinates of the laser path written with open shutter.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            `x`, `y` coordinates arrays.

        See Also
        --------
        path3d: List of `x`, `y` and `z`-coordinates of the laser path written with open shutter.
        """

        logger.debug('Return 2D path with shutter = ON.')
        x, y, _ = self.path3d
        return x, y

    @property
    def path3d(self) -> tuple[nparray, nparray, nparray]:
        """List of `x`, `y` and `z`-coordinates of the laser path written with open shutter.

        It takes the `x`, `y` and `z`, and shutter values `s` values from the path trajectory and filters out the
        points written at closed shutter (``s = 0``).

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            `x`, `y` and `z` coordinates arrays.

        See Also
        --------
        path: List of `x` and `y`-coordinates of the laser path written with open shutter.
        """

        if self._x.size:
            # filter 3D points without F
            x, y, z, s = unique_filter([self._x, self._y, self._z, self._s])
            # mask and select just those with s = 1
            x = np.delete(x, np.where(np.invert(s.astype(bool))))
            y = np.delete(y, np.where(np.invert(s.astype(bool))))
            z = np.delete(z, np.where(np.invert(s.astype(bool))))
            logger.debug('Return 3D path with shutter = ON.')
            return x, y, z
        logger.debug('No points present, return empty arrays.')
        return np.array([]), np.array([]), np.array([])

    @property
    def length(self) -> float:
        """Length of the laser path trajectory.

        Returns
        -------
        float
            The length of the path [mm].
        """

        x, y, z = self.path3d
        length = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))
        logger.debug(f'Return total length, l = {length}.')
        return length

    @property
    def fabrication_time(self) -> float:
        """Total fabrication time in seconds.

        It takes the x, y and z of the laser path and calculates the distance between each point and the next,
        computes the element-wise divison between that distance and the f values of the path to get the time it takes
        to travel that distance. Finally, it sums all the contribution to get the total fabrication time.

        Returns
        -------
        float
            Fabrication time [s].
        """

        x = np.tile(self._x, self.scan)
        y = np.tile(self._y, self.scan)
        z = np.tile(self._z, self.scan)
        f = np.tile(self._f, self.scan)

        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        time = float(sum(dists / f[1:]))
        logger.debug(f'Return total fabrication time, {time} s.')
        return time

    def curvature(self) -> tuple[nparray, nparray]:
        """Curvature.

        The curvature vector, denoted as k, is a vector that points in the direction of the center of curvature of a
        curve in three-dimensional space. Its components are related to the curvature of the curve along each of the
        coordinate axes.

        In more detail, let's say the curvature vector is given by k = (kx, ky, kz). Here's what each component
        signifies:
            kx: the component of the curvature along the x-axis.
            ky: the component of the curvature along the y-axis (same as a flat, 2D, curve).
            kz: the component of the curvature along the z-axis.

        So, each component of the curvature vector indicates the curvature of the curve in the corresponding plane
        formed by two of the three coordinate axes. The sign of each component also indicates the direction of
        curvature along that axis: positive values indicate a leftward curvature, and negative values indicate a
        rightward curvature.
        Note this curvature is numerically computed so areas where the curvature jumps instantaneously (such as
        between an arc and a straight segment) will be slightly interpolated, and sudden changes in point density
        along the curve can cause discontinuities.

        Returns
        -------
        s: np.ndarray[N]
            The arc-length of the path.
        k: np.ndarray[3][N]
            The curvature of the path.
        """
        points = np.array(self.path3d).T

        ds = np.sqrt(np.sum(np.gradient(points, axis=0) ** 2, axis=1))
        s = np.cumsum(ds)

        T_vector = np.gradient(points, axis=0)
        norm_T = np.linalg.norm(T_vector, axis=1)
        unit_T = T_vector / norm_T[:, np.newaxis]

        # Compute the derivative of the unit tangent vector with respect to arc length
        dT_ds = np.gradient(unit_T, axis=0)
        k = dT_ds / norm_T[:, np.newaxis]
        return s, k

    @property
    def curvature_radius(self) -> nparray:
        """Curvature radius.

        Returns
        -------
        nparray
            Point-by-point curvature radius.
        """
        _, k = self.curvature()
        k_magnitude = np.sqrt(np.sum(k**2, axis=1))
        return np.divide(
            np.ones_like(k_magnitude),
            k_magnitude,
            where=~np.isclose(k_magnitude, np.zeros_like(k_magnitude), atol=1e-3),
        )

    @property
    def cmd_rate(self) -> nparray:
        """Point-to-point command rate of the laser path.

        Returns
        -------
        numpy.ndarray
            Array of point-to-point command rate values of the trajectory.
        """

        # exclude last point, it's there just to close the shutter
        x, y, z, f, _ = unique_filter([self._x, self._y, self._z, self._f, self._s])[:, :-1]

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)
        dt = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

        # only divide nonzeros else Inf
        default_zero = np.zeros(np.size(dt))
        cmd_rate = np.divide(f, dt, out=default_zero, where=(dt != 0)).astype(np.float64)
        logger.debug(f'Return point-to-point command rate values, avg_cmd_rate = {np.mean(cmd_rate)}.')
        return cmd_rate

    @staticmethod
    def get_sbend_parameter(dy: float, radius: float) -> tuple[float, float]:
        """Compute the rotation angle, and `x`-displacement for a circular S-bend.

        Parameters
        ----------
        dy: float, optional
            Displacement along `y`-direction [mm].
        radius: float, optional
            Curvature radius of the S-bend [mm]. The default value is `self.radius`.

        Returns
        -------
        tuple(float, float)
            Rotation angle [rad], `x`-displacement [mm].
        """

        if radius is None or radius <= 0:
            raise ValueError(f'Radius should be a positive value. Given {radius}.')
        if dy is None:
            raise ValueError('dy is None. Give a valid input valid.')

        a = np.arccos(1 - (np.abs(dy / 2) / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    # Methods
    def start(self, init_pos: list[float] | None = None, speed_pos: float | None = None) -> LaserPath:
        """Start a laser path.

        The function starts the laserpath in the optional initial position given as input. If the initial position
        is not given, the laser path starts in [`self.x_init`, `self.y_init`, `self.z_init`].

        Parameters
        ----------
        init_pos: list[float], optional
            [`x`, `y`, `z`] coordinates of the initial point [mm]. Default value is [`self.x_init`, `self.y_init`,
            `self.z_init`].
        speed_pos: float, optional
            Translation speed [mm/s]. Default value is `self.speed_pos`.

        Returns
        -------
        None.
        """

        if self._x.size != 0:
            raise ValueError('Coordinate matrix is not empty. Cannot start a new laser path in this point.')

        if init_pos is None:
            xi, yi, zi = self.init_point
        else:
            if np.size(init_pos) != 3:
                raise ValueError(f'Given initial position is not valid. 3 values required. {np.size(init_pos)} given.')
            xi, yi, zi = init_pos

        if speed_pos is None:
            speed_pos = self.speed_pos

        x0 = np.array([xi, xi])
        y0 = np.array([yi, yi])
        z0 = np.array([zi, zi])
        f0 = np.array([speed_pos, speed_pos])
        s0 = np.array([0.0, 1.0])

        self.add_path(x0, y0, z0, f0, s0)
        return self

    def end(self) -> None:
        """Ends a laser path.

        The function automatically returns to the initial point of the laser path with a translation speed of
        `self.speed_close`.

        Returns
        -------
        None.
        """

        if not self._x.size:
            raise IndexError('Try to access an empty array. Use the start() method before the end() method.')

        # append the transformed path and add the coordinates to return to the initial point
        x = np.array([self._x[-1], self._x[0]])
        y = np.array([self._y[-1], self._y[0]])
        z = np.array([self._z[-1], self._z[0]])
        f = np.array([self._f[-1], self.speed_closed])
        s = np.array([0, 0])
        self.add_path(x, y, z, f, s)

    def add_path(
        self,
        x: nparray,
        y: nparray,
        z: nparray,
        f: nparray,
        s: nparray,
    ) -> None:
        """Appends the given arrays to the end of the existing coordinates.

        Parameters
        ----------
        x: numpy.ndarray
            Array of `x`-coordinate values.
        y: numpy.ndarray
            Array of `y`-coordinate values.
        z: numpy.ndarray
            Array of `z`-coordinate values.
        f: numpy.ndarray
            Array of translation speed values.
        s: numpy.ndarray
            Array of shutter state values.

        Returns
        -------
        None.
        """
        self._x = np.append(self._x, x.astype(np.float64))
        self._y = np.append(self._y, y.astype(np.float64))
        self._z = np.append(self._z, z.astype(np.float64))
        self._f = np.append(self._f, f.astype(np.float64))
        self._s = np.append(self._s, s.astype(np.float64))

    def linear(
        self,
        increment: Sequence[float | None],
        mode: str = 'INC',
        shutter: int = 1,
        speed: float | None = None,
    ) -> LaserPath:
        """Add a linear increment to the current laser path.

        Parameters
        ----------
        increment: list[float]
            List of increments [`dx`, `dy`, `dz`] in incremental (**INC**) mode [mm].
            New position [`x_f`, `y_f, `z_f`] in absolute (**ABS**) mode [mm].
        mode: str
            Mode selector. Default value is `INC`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.
        """

        if mode.lower() not in ['abs', 'inc']:
            raise ValueError(f'Mode should be either ABS or INC. {mode.upper()} was given.')

        if len(increment) != 3:
            raise ValueError(f'Increment should be a list of three values. Increment has {len(increment)} entries.')

        if speed is None and self.speed is None:
            raise ValueError('Speed is None. Set LaserPath\'s "speed" attribute or give a speed as input.')

        if mode.lower() == 'abs':
            # If increment is None use the last value on the coordinate-array
            x_inc = self._x[-1] if increment[0] is None else increment[0]
            y_inc = self._y[-1] if increment[1] is None else increment[1]
            z_inc = self._z[-1] if increment[2] is None else increment[2]
        else:
            x, y, z = map(lambda k: k or 0, increment)
            x_inc = self._x[-1] + x
            y_inc = self._y[-1] + y
            z_inc = self._z[-1] + z

        l_curve = np.sqrt((x_inc - self._x[-1]) ** 2 + (y_inc - self._y[-1]) ** 2 + (z_inc - self._z[-1]) ** 2)
        f_val = self.speed if speed is None else speed
        if l_curve <= 1e-6 or self.warp_flag is False:
            x_arr = np.array(x_inc)
            y_arr = np.array(y_inc)
            z_arr = np.array(z_inc)
        else:
            num = self.num_subdivisions(l_curve=l_curve, speed=f_val)
            x_arr = np.linspace(self._x[-1], x_inc, num)
            y_arr = np.linspace(self._y[-1], y_inc, num)
            z_arr = np.linspace(self._z[-1], z_inc, num)

        f_arr = f_val * np.ones_like(x_arr)
        s_arr = float(shutter) * np.ones_like(x_arr)

        self.add_path(x_arr, y_arr, z_arr, f_arr, s_arr)
        return self

    def num_subdivisions(self, l_curve: float = 0, speed: float | None = None) -> int:
        """Compute the number of points required to work at the maximum command rate.

        Parameters
        ----------
        l_curve: float, optional
            Length of the laser path segment [mm]. The default value is 0.0 mm.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        num: int
            The number of subdivisions.
        """

        f = self.speed if speed is None else speed
        if f < 1e-6:
            raise ValueError('Speed set to 0.0 mm/s. Check speed parameter.')

        dl = f / self.cmd_rate_max
        num = int(np.ceil(l_curve / dl))
        if num <= 1:
            logger.critical('I had to add use an higher instruction rate.')
            return 3
        else:
            return num

    def export(self, filename: str, as_dict: bool = False) -> None:
        """Export the object as a pickle file.

        Parameters
        ----------
        filename: str
            Name of (or path to) the file to be saved.
        as_dict: bool, optional
            Flag varibale to export the object as dictionary. The default value is False.

        Returns
        -------
        None.
        """

        fn = pathlib.Path(filename)
        if fn.suffix not in ['.pickle', 'pkl']:
            fn = pathlib.Path(fn.stem + '.pickle')
        with open(fn, 'wb') as p:
            if as_dict:
                dill.dump(attrs.asdict(self), p)
            else:
                dill.dump(self, p)
            logger.info(f'{self.__class__.__name__} exported to {fn}.')


def main() -> None:
    """The main function of the script."""
    import matplotlib.pyplot as plt
    from addict import Dict as ddict
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    parameters_lp = ddict(scan=6, speed=20, lsafe=3)

    lpath = LaserPath(**parameters_lp)
    path_x = np.array([0, 1, 1, 2, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_y = np.array([0, 0, 2, 3, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4])
    path_z = np.array([0, 0, 0, 3, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_f = np.array([1, 2, 3, 4, 3, 1, 1, 1, 1, 3, 1, 1, 1, 6, 1])
    path_s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 0])
    lpath.add_path(path_x, path_y, path_z, path_f, path_s)
    print(lpath.points.T)

    # Export Laserpath
    lpath.export('LP.pickle')

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.plot(lpath.x, lpath.y, lpath.z, '-k', linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f'Expected writing time {lpath.fabrication_time:.3f} seconds')
    print(f'Laser path length {lpath.length:.6f} mm')

    print(lpath.metadata)


if __name__ == '__main__':
    main()
