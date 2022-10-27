from operator import add
from typing import List
import os
import pickle
from dataclasses import dataclass
from itertools import product
from math import ceil, radians
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from scipy.interpolate import interp2d
from shapely.geometry import box

import numpy as np

from femto.helpers import dotdict

@dataclass
class LaserPathParameters:
    """
    Class containing the parameters for generic FLM written structure fabrication.
    """

    scan: int =1
    speed: float =1.0
    x_init: float = 0
    y_init: float = None
    z_init: float = None
    lsafe: float = 2.0
    speed_closed: float = 5
    speed_pos: float = 0.5
    cmd_rate_max: float = 1200
    acc_max: float = 500
    samplesize: Tuple[float, float] = (None, None)
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    _x: np.ndarray = np.asarray([])
    _y: np.ndarray = np.asarray([])
    _z: np.ndarray = np.asarray([])
    _f: np.ndarray = np.asarray([])
    _s: np.ndarray = np.asarray([])
    
    def __post_init__(self):
        if not isinstance(self.scan, int):
            raise ValueError('Number of scan must be integer.')
    
    @property
    def init_point(self):
        if self.y_init is None:
            y0 = 0.0
        else:
            y0 = self.y_init
        if self.z_init is None:
            z0 = 0.0
        else:
            z0 = self.z_init
        return [self.x_init, y0, z0]
    
    @property
    def lvelo(self) -> float:
        # length needed to acquire the writing speed [mm]
        return 3 * (0.5 * self.speed ** 2 / self.acc_max)
    
    
    @property
    def dl(self) -> float:
        # minimum separation between two points [mm]
        return self.speed / self.cmd_rate_max
    
    
    @property
    def x_end(self) -> float:
        # end of laser path (outside the sample)
        return self.samplesize[0] + self.lsafe


class LaserPath(LaserPathParameters):
    """
    Class of irradiated paths. It manages all the coordinates of the laser path and computes the fabrication writing
    time. It is the parent of all other classes through thier *ClassParameter*
    """

    def __init__(self, param: dict):
        super().__init__(**param)
        self._wtime = float(0)

        # # Points
        # self._x = np.asarray([])
        # self._y = np.asarray([])
        # self._z = np.asarray([])
        # self._f = np.asarray([])
        # self._s = np.asarray([])

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
    def wtime(self) -> float:
        """
        Getter for the laserpath fabrication time.

        :return: Fabrication time in seconds
        :rtype: float
        """
        
        self.fabrication_time()
        return self._wtime

    @property
    def path(self) -> List:
        x, y, z, _, s = self.points.T
        x = np.delete(x, np.where(np.invert(s.astype(bool))))
        y = np.delete(y, np.where(np.invert(s.astype(bool))))
        return [x, y]

    @property
    def path3d(self) -> List:
        x, y, z, _, s = self.points.T
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

    def fabrication_time(self):
        """
        Computes the time needed to travel along the line. It assumes 
        """
        x, y, z , f= self._x, self._y, self._z, self._f
        
        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        times = dists/f[1:]
        self._wtime = (sum(times))*self.scan

    def flip_path(self):
        """
        Flip the laser path along the x-, y- and z-coordinates
        :return: None
        """

        m = np.array([1, 1, 1])
        # reverse the coordinates arrays to flip
        if self.flip_x:
            m[0] = -1
            xc = np.flip(self._x)
        else:
            xc = self._x
        if self.flip_y:
            m[1] = -1
            yc = np.flip(self._y)
        else:
            yc = self._y
        if self.flip_z:
            m[2] = -1
            zc = np.flip(self._z)
        else:
            zc = self._z

        # create flip matrix (+1 -> no flip, -1 -> flip)
        M = np.diag(m)

        # create the displacement matrix to map the transformed min/max coordinates to the original min/max coordinates)
        C = np.array([xc, yc, zc])
        d = np.array([np.max(xc) + np.min(xc),
                      np.max(yc) + np.min(yc),
                      np.max(zc) + np.min(zc)])
        S = np.multiply((1 - m) / 2, d)

        # matrix multiplication and sum element-wise
        flip_x, flip_y, flip_z = map(add, M @ C, S)

        # update coordinates
        if self.flip_x:
            self._x = np.flip(flip_x)
        else:
            self._x = flip_x
        if self.flip_y:
            self._y = np.flip(flip_y)
        else:
            self._y = flip_y
        if self.flip_z:
            self._z = np.flip(flip_z)
        else:
            self._z = flip_z

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
        data = np.stack((self._x, self._y, self._z, self._f, self._s), axis=-1).astype(np.float32)
        mask = np.diff(data, axis=0)
        mask = np.sum(np.abs(mask), axis=1, dtype=bool)
        mask = np.insert(mask, 0, True)
        return np.delete(data, np.where(mask is False), 0).astype(np.float32)


def _example():
    #### example usefull to test Waveguide, but not Laserpath as stand alone
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

     # Data
    PARAMETERS_LP = dotdict(
            scan=6,
            speed=20,
            lsafe=3,
    )

    LP_instance=LaserPath(PARAMETERS_LP)

    path_x=np.array([0,1,1,2])
    path_y=np.array([0,0,2,3])
    path_z=np.array([0,0,0,3])
    path_f=np.array([1,2,3,4])
    path_s=np.array([1,1,1,1])
    LP_instance.add_path(path_x,path_y,path_z,path_f,path_s)
    # increment = [PARAMETERS_WG.lsafe, 0, 0]


    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.plot(LP_instance.x, LP_instance.y, LP_instance.z, '-k', linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print("Expected writing time {:.3f} seconds".format(LP_instance.wtime))
    print("Laser path length {:.3f} mm".format(LP_instance.length))

if __name__ == '__main__':
    _example()
