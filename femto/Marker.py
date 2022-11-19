from dataclasses import dataclass
from typing import List

from dacite import from_dict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import numpy as np

from femto.Parameters import MarkerParameters
from femto import LaserPath
from femto.helpers import sign


@dataclass(kw_only=True)
class _Marker(LaserPath, MarkerParameters):
    """
    Class representing an ablation marker.
    """

    def __post_init__(self):
        super().__post_init__()

    def start(self, init_pos: List[float] = None, speedpos: float = None) -> Self:
        """
        Starts a laserpath in the initial position given as input.
        The coordinates of the initial position are the first added to the matrix that describes the waveguide.

        :param init_pos: Ordered list of coordinate that specifies the waveguide starting point [mm].
            init_pos[0] -> X
            init_pos[1] -> Y
            init_pos[2] -> Z
        :type init_pos: List[float]
        :param speedpos: Translation speed [mm/s].
        :type speedpos: float
        :return: Self
        :rtype: _Waveguide
        """

        if init_pos is None:
            init_pos = self.init_point
        elif len(init_pos) == 2:
            init_pos.append(self.depth)
        elif len(init_pos) == 3:
            init_pos[2] = self.depth
        else:
            raise ValueError('Given invalid position.')
        if self._x.size != 0:
            raise ValueError('Coordinate matrix is not empty. Cannot start a new waveguide in this point.')
        if speedpos is None:
            speedpos = self.speed_pos

        x0, y0, z0 = init_pos
        f0 = np.asarray(speedpos, dtype=np.float32)
        s0 = np.asarray(0.0, dtype=np.float32)
        s1 = np.asarray(1.0, dtype=np.float32)
        self.add_path(x0, y0, z0, f0, s0)
        self.add_path(x0, y0, z0, f0, s1)
        return self

    def end(self):
        """
        Ends a waveguide. The function automatically return to the initial point of the waveguide with a translation
        speed specified by the user.

        :return: Self
        :rtype: _Waveguide
        """
        x = np.array([self._x[-1]]).astype(np.float32)
        y = np.array([self._y[-1]]).astype(np.float32)
        z = np.array([self._z[-1]]).astype(np.float32)
        f = np.array([self.speed_closed]).astype(np.float32)
        s = np.array([0]).astype(np.float32)
        self.add_path(x, y, z, f, s)

    def linear(self, increment: list, mode: str = 'INC', shutter: int = 1, speed: float = None) -> Self:
        """
        Adds a linear increment to the last point of the current waveguide.

        :param increment: Ordered list of coordinate that specifies the increment if mode is INC or new position if
        mode is ABS.
            increment[0] -> X-coord [mm]
            increment[1] -> Y-coord [mm]
            increment[2] -> Z-coord [mm]
        :type increment: List[float, float, float]
        :param mode: Select incremental or absolute mode. The default is 'INC'.
        :type mode: str
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.param.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide

        :raise ValueError: Mode is neither INC nor ABS.
        """
        if mode.upper() not in ['ABS', 'INC']:
            raise ValueError(f'Mode should be either ABS or INC. {mode.upper()} was given.')
        x_inc, y_inc, z_inc = increment
        f = self.speed if speed is None else speed
        if mode.upper() == 'ABS':
            self.add_path(x_inc, y_inc, z_inc, np.asarray(f), np.asarray(shutter))
        else:
            self.add_path(self._x[-1] + x_inc, self._y[-1] + y_inc, self._z[-1] + z_inc, np.asarray(f),
                          np.asarray(shutter))
        return self

    def cross(self, position: List[float], lx: float = 1, ly: float = 0.05):
        """
        Computes the points of a cross marker of given widht along x- and y-direction.

        :param position: 2D ordered coordinate list that specifies the cross position [mm].
            position[0] -> X
            position[1] -> Y
        :type position: List[float]
        :param lx: Length of the cross marker along x [mm]. The default is 1 mm.
        :type lx: float
        :param ly: Length of the cross marker along y [mm]. The default is 0.05 mm.
        :type ly: float
        :return: None
        """
        if len(position) == 2:
            position.append(self.depth)
        elif len(position) != 3:
            raise ValueError('Given invalid position.')

        # start_pos = np.add(position, [-lx / 2, 0, 0])
        xi, yi, zi = position
        self.start([xi - lx / 2, yi, zi], speedpos=5.0)
        self.linear([xi + lx / 2, yi, zi], mode='ABS')
        self.linear([xi + lx / 2, yi, zi], mode='ABS', shutter=0)
        self.linear([xi, yi - ly / 2, zi], mode='ABS', shutter=0)
        self.linear([xi, yi - ly / 2, zi], mode='ABS')
        self.linear([xi, yi + ly / 2, zi], mode='ABS')
        self.linear(position, mode='ABS', shutter=0)
        self.end()

    def ruler(self, y_ticks: List, lx: float, lx_short: float = None, x_init: float = -2, speedpos: float = None):
        """
        Computes the points of a ruler marker. The y-coordinates of the ticks are specified by the user as well as
        the length of the ticks in the x-direction.

        :param y_ticks: y-coordinates of the ruler's ticks [mm]
        :type y_ticks: List[float]
        :param lx: Long tick length along x [mm]. The default is 1 mm.
        :type lx: float
        :param lx_short: Short tick length along x [mm]. The default is 0.75 mm.
        :type lx_short: float
        :param x_init: Starting x-coordinate of the laser [mm]. The default is -2 mm.
        :type x_init: float
        :param speedpos: Transition speed with the shutter closes [mm/s]. The default is self.speed.
        :type speedpos: float
        :return: None
        """

        if speedpos is None:
            speedpos = self.speed_pos

        if lx_short is None:
            lx_short = 0.75 * lx
        tick_len = lx_short * np.ones_like(y_ticks)
        tick_len[0] = lx

        self.start([x_init, y_ticks[0], self.depth])
        for y, tlen in zip(y_ticks, tick_len):
            self.linear([x_init, y, self.depth], speed=speedpos, mode='ABS', shutter=0)
            self.linear([0, 0, 0], shutter=1)
            self.linear([tlen, 0, 0], speed=self.speed, shutter=1)
            self.linear([0, 0, 0], speed=self.speed, shutter=0)
        self.end()

    def meander(self, init_pos: List[float], final_pos: List[float], width: float = 1, delta: float = 0.001,
                orientation: str = 'x', speedpos: float = None):

        if speedpos is None:
            speedpos = self.speed_pos

        if orientation.lower() not in ['x', 'y']:
            raise ValueError('Orientation must be either "x" (parallel lines along x) or "y" (parallel lines along y).'
                             f'Given {orientation}.')
        s = sign()
        if orientation.lower() == 'x':
            num_passes = int(np.abs(init_pos[1] - final_pos[1]) / delta)
            delta = np.sign(final_pos[1] - init_pos[1]) * delta

            self.start(init_pos, speedpos=speedpos)
            for _ in range(num_passes):
                self.linear([next(s) * width, 0, 0], mode='INC')
                self.linear([0, delta, 0], mode='INC')
            self.linear([next(s) * width, 0, 0], mode='INC')
            self.end()

        else:
            num_passes = int(np.abs(init_pos[0] - final_pos[0]) / delta)
            delta = np.sign(final_pos[0] - init_pos[0]) * delta

            self.start(init_pos, speedpos=speedpos)
            for _ in range(num_passes):
                self.linear([0, next(s) * width, 0], mode='INC')
                self.linear([delta, 0, 0], mode='INC')
            self.linear([0, next(s) * width, 0], mode='INC')
            self.end()

    def ablation(self, points: List[List[float]] = None, shift: float = None, speedpos: float = None):
        if points is None:
            return

        if speedpos is None:
            speedpos = self.speed_closed

        self.start(points.pop(0), speedpos=speedpos)
        for p in points:
            if p[0] is None:
                p[0] = self.lastx
            if p[1] is None:
                p[1] = self.lasty
            if p[2] is None:
                p[2] = self.lastz
            self.linear(p, mode='ABS')

        if shift is not None:
            points = np.asarray(points)
            shifted_points = [np.add(points, [shift, 0, 0]), np.add(points, [-shift, 0, 0]),
                              np.add(points, [0, shift, 0]), np.add(points, [0, -shift, 0])]

            for shift in shifted_points:
                self.linear(shift[0], mode='ABS', shutter=0)
                for p in shift:
                    self.linear(p, mode='ABS')
                self.linear(shift[-1], mode='ABS', shutter=0)
        self.end()


def Marker(param):
    return from_dict(data_class=_Marker, data=param)


def _example():
    from femto.helpers import dotdict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    PARAMETERS_MK = dotdict(
            scan=1,
            speed=1,
            speed_pos=5,
            speed_closed=5,
            depth=0.000
    )

    PARAMETERS_GC = dotdict(
            filename='testMarker.pgm',
            lab='CAPABLE',
            samplesize=(25, 25),
            rotation_angle=0.0,
    )

    c = Marker(PARAMETERS_MK)
    c.cross([2.5, 1], 5, 2)
    print(c.points)

    from femto import PGMCompiler
    with PGMCompiler(PARAMETERS_GC) as gc:
        gc.write(c.points)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.plot(c.x, c.y, c.z, '-k', linewidth=2.5)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()


if __name__ == '__main__':
    _example()
