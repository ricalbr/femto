from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from femto.helpers import unique_filter
from femto.Parameters import LaserPathParameters


@dataclass(kw_only=True)
class LaserPath(LaserPathParameters):
    """
    Class of irradiated paths. It manages all the coordinates of the laser path and computes the fabrication writing
    time. It is the parent of all other classes through thier *ClassParameter*
    """

    _x: Optional[np.ndarray] = None
    _y: Optional[np.ndarray] = None
    _z: Optional[np.ndarray] = None
    _f: Optional[np.ndarray] = None
    _s: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        self._x = np.array([])
        self._y = np.array([])
        self._z = np.array([])
        self._f = np.array([])
        self._s = np.array([])

    @property
    def points(self) -> np.ndarray:
        """
        Getter for the coordinates' matrix as a numpy.ndarray matrix. The dataframe is parsed through a unique functions
        that removes all the subsequent identical points in the set.

        :return: [X, Y, Z, F, S] unique point matrix
        :rtype: numpy.ndarray
        """
        return self._unique_points()

    @property
    def x(self) -> np.ndarray:
        """
        Getter for the x-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the x-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[0]

    @property
    def lastx(self) -> float:
        return self.x[-1]

    @property
    def y(self) -> np.ndarray:
        """
        Getter for the y-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the y-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[1]

    @property
    def lasty(self) -> float:
        return self.y[-1]

    @property
    def z(self) -> np.ndarray:
        """
        Getter for the z-coordinate vector as a numpy array. The subsequent identical points in the vector are removed.

        :return: Array of the z-coordinates
        :rtype: numpy.ndarray
        """
        coords = self._unique_points().T
        return coords[2]

    @property
    def lastz(self) -> float:
        return self.z[-1]

    @property
    def lastpt(self) -> np.ndarray:
        """
        Getter for the last point of the waveguide.

        :return: Final point [x, y, z]
        :rtype: numpy.ndarray
        """
        if self._x.size > 0:
            return np.array([self._x[-1], self._y[-1], self._z[-1]])
        return np.array([])

    @property
    def path(self) -> List:
        x, y, _ = self.path3d
        return [x, y]

    @property
    def path3d(self) -> List:
        # filter 3D points without F
        x, y, z, s = unique_filter([self._x, self._y, self._z, self._s]).T
        # mask and select just those with s = 1
        x = np.delete(x, np.where(np.invert(s.astype(bool))))
        y = np.delete(y, np.where(np.invert(s.astype(bool))))
        z = np.delete(z, np.where(np.invert(s.astype(bool))))
        return [x, y, z]

    @property
    def length(self) -> float:
        x, y, z = self.path3d
        return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))

    def add_path(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, f: np.ndarray, s: np.ndarray):
        """
        Takes [x, y, z, f, s] numpy.ndarrays and adds it to the class coordinates.

        :param x: array of x-coordinates.
        :type x: numpy.ndarray
        :param y: array of y-coordinates.
        :type y: numpy.ndarray
        :param z: array of z-coordinates.
        :type z: numpy.ndarray
        :param f: array of feed rates (speed) coordinates.
        :type f: numpy.ndarray
        :param s: array of shutter coordinates.
        :type s: numpy.ndarray
        """
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        self._z = np.append(self._z, z)
        self._f = np.append(self._f, f)
        self._s = np.append(self._s, s)

    @property
    def fabrication_time(self) -> float:
        """
        Getter for the laserpath fabrication time.

        :return: Fabrication time in seconds
        :rtype: float
        """
        x = np.tile(self._x, self.scan)
        y = np.tile(self._y, self.scan)
        z = np.tile(self._z, self.scan)
        f = np.tile(self._f, self.scan)

        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        times = dists / f[1:]
        return sum(times)

    # Private interface
    def _unique_points(self):
        """
        Remove duplicate subsequent points. At least one coordinate have to change between two consecutive lines of the
        (X,Y,Z,F,S) matrix.

        Duplicates can be selected by creating a boolean index mask as follows:
            - make a row-wise diff (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.diff.html>`_)
            - compute absolute value of all elements in order to work only with positive numbers
            - make a column-wise sum (`numpy.diff <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_)
            - mask is converted to boolean values
        In this way consecutive duplicates correspond to a 0 value in the latter array.
        Converting this array to boolean (all non-zero values are True) the index mask can be retrieved.
        The first element is set to True by default since it is lost by the diff operation.

        :return: Modified coordinate matrix (x, y, z, f, s) without duplicates.
        :rtype: numpy.ndarray
        """
        return unique_filter([self._x, self._y, self._z, self._f, self._s])

    def _get_num(self, l_curve: float = 0, speed: float = 0) -> int:
        """
        Utility function that, given the length of a segment and the fabrication speed, computes the number of points
        required to work at the maximum command rate (attribute of _Waveguide obj).

        :param l_curve: Length of the waveguide segment [mm]. The default is 0 mm.
        :type l_curve: float
        :param speed: Fabrication speed [mm/s]. The default is 0 mm/s.
        :type speed: float
        :return: Number of subdivisions.
        :rtype: int

        :raise ValueError: Speed is set too low.
        """
        f = self.speed if speed is None else speed
        if f < 1e-6:
            raise ValueError('Speed set to 0.0 mm/s. Check speed parameter.')

        dl = f / self.cmd_rate_max
        num = int(np.ceil(l_curve / dl))
        if num <= 1:
            print('I had to add use an higher instruction rate.\n')
            return 3
        return num


def _example():
    # Example usefull to test Waveguide, but not Laserpath as stand alone

    from femto.helpers import dotdict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAMETERS_LP = dotdict(
            scan=6,
            speed=20,
            lsafe=3,
    )

    lpath = LaserPath(**PARAMETERS_LP)

    path_x = np.array([0, 1, 1, 2, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_y = np.array([0, 0, 2, 3, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4])
    path_z = np.array([0, 0, 0, 3, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4])
    path_f = np.array([1, 2, 3, 4, 3, 1, 1, 1, 1, 3, 1, 1, 1, 6, 1])
    path_s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 0])
    lpath.add_path(path_x, path_y, path_z, path_f, path_s)

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

    print("Expected writing time {:.3f} seconds".format(lpath.fabrication_time))
    print("Laser path length {:.3f} mm".format(lpath.length))


if __name__ == '__main__':
    _example()
