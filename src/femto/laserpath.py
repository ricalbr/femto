from __future__ import annotations

import dataclasses
import inspect
import pathlib
from typing import Any
from typing import Sequence
from typing import TypeVar

import dill
import numpy as np
import numpy.typing as npt
from femto.helpers import unique_filter

LP = TypeVar('LP', bound='LaserPath')


@dataclasses.dataclass(repr=False)
class LaserPath:
    """Class that computes and stores the coordinates of a laser path."""

    name: str | None = None  # Name of the laser path.
    scan: int = 1  #: Number of overlapped scans.
    speed: float = 1.0  #: Opened shutter translation speed `[mm/s]`.
    samplesize: tuple[float, float] = (100, 50)  #: Dimensions of the sample (x `[mm]`, y `[mm]`).
    x_init: float = -2.0  #: Initial x-coordinate for the laser path `[mm]`.
    y_init: float = 0.0  #: Initial y-coordinate for the laser path `[mm]`
    z_init: float | None = None  #: Initial z-coordinate for the laser path `[mm]`.
    shrink_correction_factor: float = 1.0  #: Correcting factor for glass shrinking.
    lsafe: float = 2.0  #: Safe margin length `[mm]`.
    speed_closed: float = 5  #: Closed shutter translation speed `[mm/s]`.
    speed_pos: float = 0.5  #: Positioning speed (shutter closed)`[mm/s]`.
    cmd_rate_max: float = 1200  #: Maximum command rate `[cmd/s]`.
    acc_max: float = 500  #: Maximum acceleration/deceleration `[m/s^2]`.
    end_off_sample: bool = True  #: Flag to end laserpath off of the sample. (See `x_end`).
    warp_flag: bool = False  #: Flag to toggle the glass warp compensation.

    _x: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
    _y: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
    _z: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
    _f: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))
    _s: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))

    def __post_init__(self) -> None:
        if not isinstance(self.scan, int):
            raise ValueError(f'Number of scan must be integer. Given {self.scan}.')

        if self.name is None:
            self.name = type(self).__name__

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @classmethod
    def from_dict(cls: type[LP], param: dict[str, Any]) -> LP:
        """Create an instance of the class from a dictionary.

        It takes a class and a dictionary, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        param, dict()
            Dictionary mapping values to class attributes.

        Returns
        -------
        Instance of class
        """
        return cls(**{k: v for k, v in param.items() if k in inspect.signature(cls).parameters})

    @property
    def init_point(self) -> tuple[float, float, float]:
        """Initial point of the trajectory.

        Returns
        -------
        tuple(float, float, float)
            `[x0, y0, z0]` coordinates.
        """

        z0 = self.z_init if self.z_init is not None else 0.0
        return self.x_init, self.y_init, z0

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

        return 3 * (0.5 * self.speed**2 / self.acc_max)

    @property
    def dl(self) -> float:
        """Compute the minimum spatial separation between two points.

        The minimum spatial separation between two points is the speed divided by the maximum command rate.

        Returns
        -------
        float
            The minimum separation between two points [mm]
        """

        return self.speed / self.cmd_rate_max

    @property
    def x_end(self) -> float | None:
        """Compute the `x` coordinate of the laserpth outside the sample, if the sample size is not None.

        Returns
        -------
        float, optional
            The end of the laser path outside the sample [mm].
        """

        if self.samplesize[0] is None:
            return None
        if self.end_off_sample:
            return self.samplesize[0] + self.lsafe
        else:
            return self.samplesize[0] - self.lsafe

    @property
    def points(self) -> npt.NDArray[np.float64]:
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

        return np.array(unique_filter([self._x, self._y, self._z, self._f, self._s]))

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """`x`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        See Also
        --------
        femto.helpers.unique_filter : Filter trajectory points to remove subsequent identical points.

        Returns
        -------
        numpy.ndarray
            The `x`-coordinates of the points in the laser path
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            return np.array(coords[0])
        return np.array([])

    @property
    def lastx(self) -> float | None:
        """Last `x` value in the trajectory points matrix, if any.

        Returns
        -------
        float, optional
            The last value of the `x` array.
        """

        arrx = self.x
        if arrx.size:
            return float(arrx[-1])
        return None

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """`y`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            The `y`-coordinates of the points in the laser path

        See Also
        --------
        unique_filter : Filter trajectory points to remove subsequent identical points.
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            return np.array(coords[1])
        return np.array([])

    @property
    def lasty(self) -> float | None:
        """Last `y` value in the trajectory points matrix, if any.

        Returns
        -------
        float, optional
            The last value of the `y` array.
        """

        arry = self.y
        if arry.size:
            return float(arry[-1])
        return None

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """`z`-coordinate vector as a numpy array.

        The subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            The `z`-coordinates of the points in the laser path

        See Also
        --------
        unique_filter: Filter trajectory points to remove subsequent identical points.
        """

        coords = unique_filter([self._x, self._y, self._z, self._f, self._s])
        if coords.ndim == 2:
            return np.array(coords[2])
        return np.array([])

    @property
    def lastz(self) -> float | None:
        """Last `z` value in the trajectory points matrix, if any.

        Returns
        -------
        float, optional
            The last value of the `z` array.
        """

        arrz = self.z
        if arrz.size:
            return float(arrz[-1])
        return None

    @property
    def lastpt(self) -> npt.NDArray[np.float64]:
        """Last point of the laser path, if any.

        Returns
        -------
        numpy.ndarray
            Final `[x, y, z]` pointof the laser path.
        """

        if self._x.size > 0:
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        return np.array([])

    @property
    def path(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """List of `x` and `y`-coordinates of the laser path written with open shutter.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            `x`, `y` coordinates arrays.

        See Also
        --------
        path3d: List of `x`, `y` and `z`-coordinates of the laser path written with open shutter.
        """

        x, y, _ = self.path3d
        return x, y

    @property
    def path3d(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
            return x, y, z
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
        return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))

    @property
    def fabrication_time(self) -> float:
        """Total fabrication time in seconds.

        It takes the x, y and z of the laser path and calculates the distance between each point and the next,
        computes the element-wise division between that distance and the f values of the path to get the time it takes
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
        times = dists / f[1:]
        return float(sum(times))

    @property
    def curvature_radius(self) -> npt.NDArray[np.float64]:
        """Point-to-point curvature radius of the trajectory.

        The curvature radius is computed as the radius of the circle that best fits the curve at a given point.

        Returns
        -------
        numpy.ndarray
            Array of curvature radii of the trajectory.
        """

        points = np.array(self.path3d).T

        # Compute the first and second derivatives of the curve
        t = np.linspace(0, 1, len(points))
        r1 = np.gradient(points, t, axis=0, edge_order=2)
        r2 = np.gradient(r1, t, axis=0, edge_order=2)

        # Compute the cross product and norm of the cross product
        cross = np.cross(r1, r2)
        norm_cross = np.linalg.norm(cross, axis=1)
        norm_r1 = np.linalg.norm(r1, axis=1)

        # Return the local curvature radius, where cannot divide by 0 return inf
        return np.divide(norm_r1**3, norm_cross, out=np.full_like(norm_r1, fill_value=np.inf), where=~(norm_cross == 0))

    @property
    def cmd_rate(self) -> npt.NDArray[np.float64]:
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
        cmd_rate = np.divide(f, dt, out=default_zero, where=(dt != 0))
        return np.array(cmd_rate, dtype=np.float64)

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
        None
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
        None
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
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ):
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
        None
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
            x_inc = self._x[-1] if increment[0] is None else np.array([increment[0]])
            y_inc = self._y[-1] if increment[1] is None else np.array([increment[1]])
            z_inc = self._z[-1] if increment[2] is None else np.array([increment[2]])
        else:
            x, y, z = map(lambda k: k or 0, increment)
            x_inc = np.array([self._x[-1] + x])
            y_inc = np.array([self._y[-1] + y])
            z_inc = np.array([self._z[-1] + z])

        l_curve = np.sqrt((x_inc - self._x[-1]) ** 2 + (y_inc - self._y[-1]) ** 2 + (z_inc - self._z[-1]) ** 2)
        f_val = self.speed if speed is None else speed
        if l_curve <= 1e-6 or self.warp_flag is False:
            x_arr = np.array([x_inc])
            y_arr = np.array([y_inc])
            z_arr = np.array([z_inc])
        else:
            num = self.num_subdivisions(l_curve=l_curve, speed=f_val)
            x_arr = np.linspace(self._x[-1], x_inc, num)
            y_arr = np.linspace(self._y[-1], y_inc, num)
            z_arr = np.linspace(self._z[-1], z_inc, num)

        f_arr = f_val * np.ones_like(x_arr)
        s_arr = shutter * np.ones_like(x_arr)

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
            print('I had to add use an higher instruction rate.\n')
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
            Flag variable to export the object as dictionary. The default value is False.

        Returns
        -------
        None
        """

        fn = pathlib.Path(filename)
        if fn.suffix not in ['.pickle', 'pkl']:
            fn = pathlib.Path(fn.stem + '.pkl')
        with open(fn, 'wb') as p:
            if as_dict:
                dill.dump(self.__dict__, p)
            else:
                dill.dump(self, p)
            print(f'{self.__class__.__name__} exported to {fn}.')


def main() -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from femto.helpers import dotdict

    # Data
    PARAMETERS_LP = dotdict(scan=6, speed=20, lsafe=3)

    lpath = LaserPath(**PARAMETERS_LP)
    path_x = np.array([0, 1, 1, 2, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_y = np.array([0, 0, 2, 3, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4])
    path_z = np.array([0, 0, 0, 3, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_f = np.array([1, 2, 3, 4, 3, 1, 1, 1, 1, 3, 1, 1, 1, 6, 1])
    path_s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 0])
    lpath.add_path(path_x, path_y, path_z, path_f, path_s)
    print(lpath.points.T)

    # Export Laserpath
    lpath.export('LP.pkl')

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]'), ax.set_zlabel('Z [mm]')
    ax.plot(lpath.x, lpath.y, lpath.z, '-k', linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f'Expected writing time {lpath.fabrication_time:.3f} seconds')
    print(f'Laser path length {lpath.length:.6f} mm')


if __name__ == '__main__':
    main()
