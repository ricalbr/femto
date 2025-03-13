from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
from femto.helpers import sign
from femto.laserpath import LaserPath


@dataclasses.dataclass(repr=False)
class Marker(LaserPath):
    """Class that computes and stores the coordinates of a superficial abletion marker."""

    depth: float = 0.0
    lx: float = 1.0
    ly: float = 0.060

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.z_init is None:
            self.z_init = self.depth

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def cross(self, position: list[float], lx: float | None = None, ly: float | None = None) -> None:
        """Cross marker.

        The function takes the position (`x`, `y`, `z`) of the center of the marker and two lengths of the arms
        (`lx`, `ly`) and draws a cross with the given lengths.

        position : list(float)
            2D (`x`, `y`) or 3D (`x`, `y`, `z`) coordinates of the center of the cross [mm]. In case of 2D position the
            `z` coordinate is set to `self.depth`.
        lx : float, optional
            Length of the cross arm along the `x` direction [mm]. The default value is ``self.lx``.
        ly : float, optional
            Length of the cross arm along the `y` direction [mm]. The default value is ``self.ly``.

        Returns
        -------
        None
        """

        if len(position) == 2:
            position.append(self.depth)

        if len(position) != 3:
            raise ValueError(
                'The cross position is not valid. The valid number of elements are 2 or 3. The given '
                f'position has {len(position)} elements.'
            )

        if lx is None and self.lx is None:
            raise ValueError(
                "The x-length of the cross is None. Set the Marker's 'lx' attribute or give a valid number as " 'input.'
            )
        lx = self.lx if lx is None else lx
        if ly is None and self.ly is None:
            raise ValueError(
                "The y-length of the cross is None. Set the Marker's 'ly' attribute or give a valid number as " 'input.'
            )
        ly = self.ly if ly is None else ly

        # start_pos = np.add(position, [-lx / 2, 0, 0])
        xi, yi, zi = position
        self.start([xi - lx / 2, yi, zi])
        self.linear([lx, None, None], mode='INC')
        self.linear([None, None, None], mode='INC', shutter=0)
        self.linear([-lx / 2, -ly / 2, None], mode='INC', shutter=0)
        self.linear([None, None, None], mode='INC')
        self.linear([None, ly, None], mode='INC')
        self.linear([None, None, None], mode='INC', shutter=0)
        self.linear(position, mode='ABS', shutter=0)

    def ruler(
        self,
        y_ticks: list[float] | npt.NDArray[np.float64],
        lx: float | None = None,
        lx2: float | None = None,
        x_init: float | None = None,
    ) -> None:
        """Ticks marker.

        Draws a series of horizontal ablation lines parallel to the `x` axis. The `y`-coordinates of the ticks are
        given as input.

        y_ticks : array-like, list[float], npt.NDArray[np.float64]
            List of `y`-coordinate values for each tick in the marker.
        lx : float, optional
            Length of the longer tick marks [mm]. The default value is ``self.lx``.
        lx2 : float, optional
            Length of the tick marks [mm]. The default value is ``0.75 * self.lx``.
        x_init : float
            Starting position for the `x` coordinate [mm]. The default values is ``self.x_init``.

        Returns
        -------
        None
        """

        if y_ticks is None or len(y_ticks) == 0:
            return None

        # Sort the y-ticks and takes unique points
        y_ticks = np.unique(y_ticks)

        # Ticks x-length
        if lx is None and self.lx is None:
            raise ValueError('The tick length is None. Set the "lx" attribute or give a valid number as input.')
        lx = self.lx if lx is None else lx
        lx2 = 0.75 * self.lx if lx2 is None else lx2

        x_ticks = np.repeat(lx2, len(y_ticks))
        x_ticks[0] = lx

        # Set x_init
        if x_init is None and self.x_init is None:
            raise ValueError(
                'The x-init value of the ruler is None. Set the "x_init" attribute or give a valid number as input.'
            )
        x_init = self.x_init if x_init is None else x_init

        # Add straight segments
        self.start([x_init, y_ticks[0], self.depth])
        for xt, yt in zip(x_ticks, y_ticks):
            self.linear([x_init, yt, self.depth], mode='ABS', shutter=0)
            self.linear([None, None, None], mode='ABS')
            self.linear([xt, yt, None], mode='ABS')
            self.linear([None, None, None], mode='ABS', shutter=0)
        self.end()

    def meander(
        self,
        init_pos: list[float],
        final_pos: list[float],
        width: float = 1,
        delta: float = 0.001,
        orientation: str = 'x',
    ) -> None:
        """Parallel serpentine marker.

        The function takes in the initial and final positions, the width of the meander, the distance between the
        parallel lines and connects the two points with a serpentine-like pattern.
        The orientation of the meander (parallel to `x` or `y` axis) can be given as input.

        init_pos : list(float)
            Starting position of the meander [mm].
        final_pos : list(float)
            Ending position of the meander [mm].
        width : float
            Longitudinal width of the meander [mm]. The default value is 1 mm.
        delta : float
            The distance between the parallel lines of the meander [mm]. The default value is 0.001 mm.
        orientation : str
            Orientation of the meander, either parallel to the 'x'-axis or 'y'-axis. The default value is 'x'.

        Returns
        -------
        None
        """

        if len(init_pos) not in [2, 3]:
            raise ValueError(
                'Initial position must be a list of 2 (x,y) or 3 (x,y,z) values.'
                f"init_pos has {len(init_pos)} elements. Give a valid 'init_pos' list."
            )
        xi, yi, *_ = init_pos

        if len(final_pos) not in [2, 3]:
            raise ValueError(
                'Final position must be a list of 2 (x,y) or 3 (x,y,z) values.'
                f'final_pos has {len(final_pos)} elements. Give a valid "final_pos" list.'
            )
        xf, yf, *_ = final_pos

        if orientation.lower() not in ['x', 'y']:
            raise ValueError(
                f'Orientation must be either "x" (parallel to x axis) or "y" (parallel to y axis). Given {orientation}.'
            )

        s = sign()
        self.start(init_pos)
        if orientation.lower() == 'x':
            num_passes = math.floor(np.abs(yf - yi) / delta)
            delta = np.sign(yf - yi) * delta

            for _ in range(num_passes):
                self.linear([next(s) * width, 0, 0], mode='INC')
                self.linear([0, delta, 0], mode='INC')
            self.linear([next(s) * width, 0, 0], mode='INC')
        else:
            num_passes = math.floor(np.abs(xf - xi) / delta)
            delta = np.sign(xf - xi) * delta

            for _ in range(num_passes):
                self.linear([0, next(s) * width, 0], mode='INC')
                self.linear([delta, 0, 0], mode='INC')
            self.linear([0, next(s) * width, 0], mode='INC')
        self.end()

    def ablation(
        self,
        points: list[list[float]],
        shift: float | None = None,
    ) -> None:
        """Ablation line.

        The function takes a list of points and connects them with linear segments. The first point is the starting
        point of the ablation line.
        The line can shifted by a value given as input. Given a shift of `ds` the line is rigidly shifted along `x`
        and `y` of `ds` and `-ds`. In the default behaviour `ds` is set to ``None`` (no shift is applied).

        points: list[list[float]]
            List of points representing the vertices of a polygonal chain line.
        shift: float
            Amount of shift between different lines. The default value is ``None``.
        """

        if not points:
            return

        # shift the path's points by shift value
        pts = np.asarray(points)
        if shift is None:
            path_list = [pts]
        else:
            path_list = [
                np.add(pts, [0, 0, 0]),
                np.add(pts, [shift, 0, 0]),
                np.add(pts, [-shift, 0, 0]),
                np.add(pts, [0, shift, 0]),
                np.add(pts, [0, -shift, 0]),
            ]

        # Start and add first path
        p_init = path_list.pop(0)
        self.start(p_init[0])
        for p in p_init:
            self.linear(p, mode='ABS')
        self.linear(p_init[-1], mode='ABS', shutter=1)
        self.linear(p_init[-1], mode='ABS', shutter=0)

        # Add all shifted paths
        for path in path_list:
            self.linear(path[0], mode='ABS', shutter=0)
            self.linear(path[0], mode='ABS', shutter=1)
            for p in path:
                self.linear(p, mode='ABS')
            self.linear(path[-1], mode='ABS', shutter=1)
            self.linear(path[-1], mode='ABS', shutter=0)
        self.end()

    def box(self, lower_left_corner: list[float], width: float = 1.0, height: float = 0.06) -> None:
        """Box.
        The function creates a rectangular ablation pattern. It takes in the lower left corner of the rectangle,
        the width and height of the rectangle as input and generates a list of points representing the vertices of
        the rectangle. The `ablation` method is then called with this list of points to create the ablation
        pattern.

        lower_left_corner: list[float]
            List of points representing the lower left corner of the rectangle.
        width: float
            Width of the rectangle [mm]. The default value is 1.0 mm.
        height: float
            Height of the rectangle [mm]. The default value is 0.060 mm.
        """

        if not lower_left_corner:
            return
        lower_left_corner = np.array(lower_left_corner)

        pts = [
            lower_left_corner,
            lower_left_corner + np.array([abs(width), 0, 0]),
            lower_left_corner + np.array([abs(width), abs(height), 0]),
            lower_left_corner + np.array([0, abs(height), 0]),
            lower_left_corner,
        ]
        self.ablation(points=pts, shift=None)


def main() -> None:
    import matplotlib.pyplot as plt

    from femto.helpers import dotdict, split_mask

    PARAMETERS_MK = dotdict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    PARAMETERS_GC = dotdict(filename='testPGM.pgm', laser='PHAROS', samplesize=(10, 10))

    c = Marker(**PARAMETERS_MK)
    # c.cross([0, 0], 5, 2)
    # c.ablation([[0, 0, 0], [5, 0, 0], [5, 1, 0], [2, 2, 0]], shift=0.001)
    c.box([1, 2, 3], width=5.0, height=0.01)
    print(c.points)

    from femto.pgmcompiler import PGMCompiler

    with PGMCompiler(**PARAMETERS_GC) as gc:
        gc.write(c.points)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]')
    x, y, *_, s = c.points
    for x_seg, y_seg in zip(split_mask(x, s.astype(bool)), split_mask(y, s.astype(bool))):
        ax.plot(x_seg, y_seg, '-k', linewidth=2.5)
    for x_seg, y_seg in zip(split_mask(x, ~s.astype(bool)), split_mask(y, ~s.astype(bool))):
        ax.plot(x_seg, y_seg, ':b', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    main()
