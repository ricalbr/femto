from dataclasses import dataclass
from typing import List

import numpy as np
from dacite import from_dict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from functools import partialmethod
from femto.helpers import dotdict
from femto import LaserPath
from femto.Parameters import WaveguideParameters


@dataclass(kw_only=True)
class _Waveguide(LaserPath, WaveguideParameters):
    """
    Class representing an optical waveguide.
    """

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return "{cname}@{id:x}".format(cname=self.__class__.__name__, id=id(self) & 0xFFFFFF)

    # Methods
    def start(self, init_pos: List[float] = None, speedpos: float = None) -> Self:
        """
        Starts a waveguide in the initial position given as input.
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
            x0, y0, z0 = self.init_point
        else:
            if np.size(init_pos) != 3:
                raise ValueError(f'Given initial position is not valid. 3 values required. {np.size(init_pos)} given.')
            x0, y0, z0 = init_pos
        if self._x.size != 0:
            raise ValueError('Coordinate matrix is not empty. Cannot start a new waveguide in this point.')
        if speedpos is None:
            speedpos = self.speed_pos

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

        # append the transformed path and add the coordinates to return to the initial point
        x = np.array([self._x[-1], self._x[0]]).astype(np.float32)
        y = np.array([self._y[-1], self._y[0]]).astype(np.float32)
        z = np.array([self._z[-1], self._z[0]]).astype(np.float32)
        f = np.array([self._f[-1], self.speed_closed]).astype(np.float32)
        s = np.array([0, 0]).astype(np.float32)
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
            self.add_path(self._x[-1] + x_inc, self._y[-1] + y_inc, self._z[-1] + z_inc, np.asarray(f), np.asarray(
                    shutter))
        return self

    def circ(self, initial_angle: float, final_angle: float, radius: float = None, shutter: int = 1,
             speed: float = None) -> Self:
        """
        Computes the points in the xy-plane that connects two angles (initial_angle and final_angle) with a circular
        arc of a given radius. The user can set the transition speed, the shutter state during the movement and the
        number of points of the arc.

        :param initial_angle: Starting rotation_angle of the circular arc [radians].
        :type initial_angle: float
        :param final_angle: Ending rotation_angle of the circular arc [radians].
        :type final_angle: float
        :param radius: Radius of the circular arc [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: float
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: int
        :return: Self
        :rtype: _Waveguide
        """
        if radius is None:
            radius = self.radius
        f = self.speed if speed is None else speed

        delta_angle = abs(final_angle - initial_angle)
        num = self._get_num(delta_angle * radius, f)

        t = np.linspace(initial_angle, final_angle, num)
        new_x = self._x[-1] - radius * np.cos(initial_angle) + radius * np.cos(t)
        new_y = self._y[-1] - radius * np.sin(initial_angle) + radius * np.sin(t)
        new_z = self._z[-1] * np.ones(new_x.shape)

        # update coordinates
        self.add_path(new_x, new_y, new_z, f * np.ones(new_x.shape), shutter * np.ones(new_x.shape))
        return self

    def arc_bend(self, dy: float, radius: float = None, shutter: int = 1, speed: float = None) -> Self:
        """
        Concatenates two circular arc to make a circular S-bend.
        The user can specify the amplitude of the S-bend (height in the y direction) and the curvature radius.
        Starting and ending rotation_angle of the two arcs are computed automatically.

        The sign of dy encodes the direction of the S-bend:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if radius is None:
            radius = self.radius
        a, _ = self.get_sbend_parameter(dy, radius)

        if dy > 0:
            self.circ(np.pi * (3 / 2), np.pi * (3 / 2) + a, radius=radius, speed=speed, shutter=shutter)
            self.circ(np.pi * (1 / 2) + a, np.pi * (1 / 2), radius=radius, speed=speed, shutter=shutter)
        else:
            self.circ(np.pi * (1 / 2), np.pi * (1 / 2) - a, radius=radius, speed=speed, shutter=shutter)
            self.circ(np.pi * (3 / 2) - a, np.pi * (3 / 2), radius=radius, speed=speed, shutter=shutter)
        return self

    def arc_acc(self, dy: float, radius: float = None, int_length: float = None, shutter: int = 1,
                speed: float = None) -> Self:
        """
        Concatenates two circular S-bend to make a single mode of a circular directional coupler.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Length of the coupler straight arm [mm]. The default is self.int_length.
        :type int_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if int_length is None:
            int_length = self.int_length

        self.arc_bend(dy, radius=radius, speed=speed, shutter=shutter)
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_bend(-dy, radius=radius, speed=speed, shutter=shutter)
        return self

    def arc_mzi(self, dy: float, radius: float = None, int_length: float = None, arm_length: float = None,
                shutter: int = 1, speed: float = None) -> Self:
        """
        Concatenates two circular couplers to make a single mode of a circular MZI.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the S-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param arm_length: Length of the coupler straight arm [mm]. The default is self.arm_length.
        :type arm_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if arm_length is None:
            arm_length = self.arm_length

        self.arc_acc(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        self.linear([arm_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_acc(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        return self

    def sin_bridge(self, dy: float, dz: float = None, radius: float = None, shutter: int = 1,
                   speed: float = None) -> Self:
        """
        Computes the points in the xy-plane of a Sin-bend curve and the points in the xz-plane of a
        Sin-bridge of height Dz. The distance between the initial and final point is the same of the equivalent
        (circular) S-bend of given radius.
        The user can specify the amplitude of the Sin-bend (height in the y direction) and the curvature radius as
        well as the transition speed, the shutter state during the movement and the number of points of the arc.

        The sign of dy encodes the direction of the Sin-bend:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        The sign of dz encodes the direction of the Sin-bridge:
            - dz > 0, top bridge
            - dz < 0, under bridge

        .. notes
            The radius is an effective radius. The radius of curvature of the
            overall curve will be lower (in general) than the specified
            radius.

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param dz: Amplitude of the S-bend along the y direction [mm].
        :type dz: float
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """

        if radius is None:
            radius = self.radius
        f = self.speed if speed is None else speed

        _, dx = self.get_sbend_parameter(dy, radius)
        num = self._get_num(dx, f)

        new_x = np.linspace(self._x[-1], self._x[-1] + dx, num)
        new_y = self._y[-1] + 0.5 * dy * (1 - np.cos(np.pi / dx * (new_x - self._x[-1])))
        if dz:
            new_z = self._z[-1] + 0.5 * dz * (1 - np.cos(2 * np.pi / dx * (new_x - self._x[-1])))
        else:
            new_z = self._z[-1] * np.ones(new_x.shape)

        # update coordinates
        self.add_path(new_x, new_y, new_z, f * np.ones(new_x.shape), shutter * np.ones(new_x.shape))
        return self

    sin_bend = partialmethod(sin_bridge, dz=None)

    def sin_bend_comp(self, dx: float, dy: float, shutter: int = 1, speed: float = None) -> Self:

        f = self.speed if speed is None else speed
        num = self._get_num(dx, f)

        new_x = np.linspace(self._x[-1], self._x[-1] + dx, num)
        new_y = self._y[-1] + 0.5 * dy * (1 - np.cos(2 * np.pi / dx * (new_x - self._x[-1])))
        new_z = self._z[-1] * np.ones(new_x.shape)

        # update coordinates
        self.add_path(new_x, new_y, new_z, f * np.ones(new_x.shape), shutter * np.ones(new_x.shape))
        return self

    def sin_acc(self, dy: float, radius: float = None, int_length: float = 0.0, shutter: int = 1,
                speed: float = None) -> Self:
        """
        Concatenates two Sin-bend to make a single mode of a sinusoidal directional coupler.
        The user can specify the amplitude of the coupler (height in the y direction) and the effective curvature
        radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward Sin-bend
            - dy < 0, downward Sin-bend

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if int_length is None:
            int_length = self.int_length

        self.sin_bend(dy, radius=radius, speed=speed, shutter=shutter)
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.sin_bend(-dy, radius=radius, speed=speed, shutter=shutter)
        return self

    def sin_mzi(self, dy: float, radius: float = None, int_length: float = None, arm_length: float = None,
                shutter: int = 1, speed: float = None) -> Self:
        """
        Concatenates two sinusoidal couplers to make a single mode of a sinusoidal MZI.
        The user can specify the amplitude of the coupler (height in the y direction) and the curvature radius.

        The sign of dy encodes the direction of the coupler:
            - dy > 0, upward S-bend
            - dy < 0, downward S-bend

        :param dy: Amplitude of the Sin-bend along the y direction [mm].
        :type dy: float
        :param radius: Curvature radius of the Sin-bend [mm]. The default is self.radius.
        :type radius: float
        :param int_length: Interaction distance of the MZI [mm]. The default is self.int_length.
        :type int_length: float
        :param arm_length: Length of the coupler straight arm [mm]. The default is self.arm_length.
        :type arm_length: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if arm_length is None:
            arm_length = self.arm_length

        self.sin_acc(dy, radius=radius, int_length=int_length, shutter=shutter, speed=speed)
        self.linear([arm_length, 0, 0], shutter=shutter, speed=speed)
        self.sin_acc(dy, radius=radius, int_length=int_length, shutter=shutter, speed=speed)
        return self

    def spline(self, dy: float, dz: float, init_pos: np.ndarray = None, radius: float = None, disp_x: float = 0,
               shutter: int = 1, speed: float = None,
               bc_y: tuple = ((1, 0.0), (1, 0.0)),
               bc_z: tuple = ((1, 0.0), (1, 0.0))) -> Self:
        """
        Function wrapper. It computes the x,y,z coordinates of spline curve starting from init_pos with Dy and Dz
        displacements.
        See :func:`~femto.Waveguide._get_spline_points`.
        The points are then appended to the Waveguide coordinate list.

        :param dy: Amplitude of the spline along the y direction [mm].
        :type dy: float
        :param dz: Amplitude of the spline along the y direction [mm].
        :type dz: float
        :param init_pos: Initial position of the spline. The default is last point of the waveguide (self.lastpt).
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is self.radius.
        :type radius: float
        :param disp_x: Displacement of the spline along the x direction [mm].
        :type disp_x: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :param bc_y:
        :type bc_y: tuple
        :param bc_z:
        :type bc_z: tuple
        :return: Self
        :rtype: _Waveguide
        """
        if radius is None:
            radius = self.radius
        x_spl, y_spl, z_spl = self._get_spline_points(**locals())
        f = self.speed if speed is None else speed

        # update coordinates or return
        self.add_path(x_spl, y_spl, z_spl, f * np.ones(x_spl.shape), shutter * np.ones(x_spl.shape))
        return self

    def spline_bridge(self, dy: float, dz: float, init_pos: list = None, radius: float = None, disp_x: float = 0,
                      shutter: int = 1, speed: float = None) -> Self:
        """
        Computes a spline bridge as a sequence of two spline segments. dy is the total displacement along the
        y-direction of the bridge and dz is the height of the bridge along z.
        First, the function computes the dx-displacement of the planar spline curve with a dy displacement. This
        datum is used to compute the value of the first derivative along the y-coordinate for the costruction of the
        spline bridge such that:

            df(x, y)/dx = dy/dx

        in the peak point of the bridge.

        The cubic spline bridge obtained in this way has a first derivatives df(x, y)/dx, df(x, z)/dx which are zero
        (for construction) in the initial and final point.
        However, the second derivatives are not null in principle. To cope with this the spline points are fitted
        with a spline of the 5-th order. In this way the final curve has second derivatives close to zero (~1e-4)
        while maintaining the first derivative to zero.

        :param dy: Amplitude of the spline along the y direction [mm].
        :type dy: float
        :param dz: Amplitude of the spline along the y direction [mm].
        :type dz: float
        :param init_pos: Initial position of the spline. The default is last point of the waveguide (self.lastpt).
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is self.radius.
        :type radius: float
        :param disp_x: Displacement of the spline along the x direction [mm].
        :type disp_x: float
        :param shutter: State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        :type shutter: int
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :return: Self
        :rtype: _Waveguide
        """
        if init_pos is None:
            init_pos = self.lastpt
        if radius is None:
            radius = self.radius
        if speed is None:
            f = self.speed
        else:
            f = speed

        dx, *_, l_curve = self.get_spline_parameter(init_pos, dy, 0, radius, disp_x)

        x1, y1, z1 = self._get_spline_points(dy / 2, dz, init_pos, radius, speed=speed,
                                             bc_y=((1, 0.0), (1, dy / dx)),
                                             bc_z=((1, 0.0), (1, 0.0)))
        init_pos2 = np.array([x1[-1], y1[-1], z1[-1]])
        x2, y2, z2 = self._get_spline_points(dy / 2, -dz, init_pos2, radius, speed=speed,
                                             bc_y=((1, dy / dx), (1, 0.0)),
                                             bc_z=((1, 0.0), (1, 0.0)))
        x = np.append(x1[:-1], x2)
        y = np.append(y1[:-1], y2)
        z = np.append(z1[:-1], z2)

        # Use CubicSpline as control point for higher order spline.
        us_y = InterpolatedUnivariateSpline(x, y, k=5)
        us_z = InterpolatedUnivariateSpline(x, z, k=5)

        # update speed array
        self.add_path(x, us_y(x), us_z(x), f * np.ones(x.shape), shutter * np.ones(x.shape))
        return self

    def curvature_radius(self) -> np.ndarray:
        """
        Computes the 3D point-to-point curvature radius of the waveguide shape.

        :return: Array of the curvature radii computed at each point of the curve.
        :rtype: numpy.ndarray
        """
        
        (x, y, z) = self.path3d
        
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        num = (dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2) ** 1.5
        den = np.sqrt((d2z_dt2 * dy_dt - d2y_dt2 * dz_dt) ** 2 +
                      (d2x_dt2 * dz_dt - d2z_dt2 * dx_dt) ** 2 +
                      (d2y_dt2 * dx_dt - d2x_dt2 * dy_dt) ** 2)
        default_zero = np.ones(np.size(num)) * np.inf

        # only divide nonzeros else Inf
        curvature_radius = np.divide(num, den, out=default_zero, where=(den != 0))
        return curvature_radius[2:-2]

    def cmd_rate(self) -> np.ndarray:
        """
        Computes the point-to-point command rate of the waveguide shape.

        :return: Average command rates computed at each point of the curve.
        :rtype: numpy.ndarray
        """
        # exclude last point, it's there just to close the shutter
        (x, y, z, f, _) = self._unique_points()[:-1, :].T

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)
        dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)

        default_zero = np.zeros(np.size(dt))
        # only divide nonzeros else Inf
        cmd_rate = np.divide(f, dt, out=default_zero, where=(dt != 0))
        return np.mean(cmd_rate)

    # Private interface
    def _get_spline_points(self, dy: float, dz: float, init_pos: np.ndarray = None, radius: float = 20,
                           disp_x: float = 0, speed: float = None,
                           bc_y: tuple = ((1, 0.0), (1, 0.0)),
                           bc_z: tuple = ((1, 0.0), (1, 0.0))) -> tuple:
        """
        Function for the generation of a 3D spline curve. Starting from init_point the function compute a 3D spline
        with a displacement dy in y-direction and dz in z-direction.
        The user can specify the length of the curve or (alternatively) provide a curvature radius that is used to
        compute the displacement along x-direction as the displacement of the equivalent circular S-bend.

        User can provide the boundary conditions for the derivatives in the y- and z-directions. Boundary conditions
        are a tuple of tuples in which we have:
            bc = ((initial point), (final point))
        where the (initial point) and (final point) tuples are specified as follows:
            (derivative order, value of derivative)
        the derivative order can be either 0, 1, 2.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param dz: Displacement along z-direction [mm].
        :type dz: float
        :param init_pos: Initial position of the curve.
        :type init_pos: np.ndarray
        :param radius: Curvature radius of the spline [mm]. The default is 20 mm.
        :type dz: radius
        :param disp_x: Displacement along x-direction [mm]. The default is 0 mm.
        :type disp_x: float
        :param speed: Transition speed [mm/s]. The default is self.speed.
        :type speed: float
        :param bc_y: Boundary conditions for the Y-coordinates. The default is ((1, 0.0), (1, 0.0)).
        :type bc_y: tuple
        :param bc_z: Boundary conditions for the z-coordinates. The default is ((1, 0.0), (1, 0.0)).
        :type bc_z: tuple
        :return: (x-coordinates, y-coordinates, z-coordinates) of the spline curve.
        :rtype: Tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        xd, yd, zd, l_curve = self.get_spline_parameter(init_pos, dy, dz, radius, disp_x)
        f = self.speed if speed is None else speed
        num = self._get_num(l_curve, f)

        xcoord = np.linspace(0, xd, num)
        cs_y = CubicSpline((0.0, xd), (0.0, yd), bc_type=bc_y)
        cs_z = CubicSpline((0.0, xd), (0.0, zd), bc_type=bc_z)

        return xcoord + init_pos[0], cs_y(xcoord) + init_pos[1], cs_z(xcoord) + init_pos[2]


def Waveguide(param):
    return from_dict(data_class=_Waveguide, data=param)


def coupler(param, d=None):
    p = dotdict(param.copy())

    if d is not None:
        p.int_dist = d

    if p.y_init is None:
        p.y_init = 0.0

    mode1 = Waveguide(p)
    mode1.start() \
        .linear([(mode1.samplesize[0] - mode1.dx_bend) / 2, mode1.lasty, mode1.lastz], mode='ABS') \
        .sin_acc(mode1.dy_bend) \
        .linear([mode1.x_end, mode1.lasty, mode1.lastz], mode='ABS') \
        .end()

    p.y_init += p.pitch
    mode2 = Waveguide(p)
    mode2.start() \
        .linear([(mode2.samplesize[0] - mode2.dx_bend) / 2, mode2.lasty, mode2.lastz], mode='ABS') \
        .sin_acc(-mode2.dy_bend) \
        .linear([mode2.x_end, mode2.lasty, mode2.lastz], mode='ABS') \
        .end()
    return [mode1, mode2]


def _example():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAMETERS_WG = dotdict(
            scan=6,
            speed=20,
            radius=15,
            pitch=0.080,
            int_dist=0.007,
            lsafe=3,
    )

    increment = [PARAMETERS_WG.lsafe, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        PARAMETERS_WG.y_init = -PARAMETERS_WG.pitch / 2 + index * PARAMETERS_WG.pitch

        wg = Waveguide(PARAMETERS_WG)
        wg.start() \
            .linear(increment) \
            .sin_mzi((-1) ** index * wg.dy_bend) \
            .spline_bridge((-1) ** index * 0.08, (-1) ** index * 0.015) \
            .sin_mzi((-1) ** (index + 1) * wg.dy_bend) \
            .sin_mzi((-1) ** (index + 1) * wg.dy_bend) \
            .linear(increment)
        wg.end()
        mzi.append(wg)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    for wg in mzi:
        ax.plot(wg.x[:-1], wg.y[:-1], wg.z[:-1], '-k', linewidth=2.5)
        ax.plot(wg.x[-2:], wg.y[-2:], wg.z[-2:], ':b', linewidth=1.0)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print("Expected writing time {:.3f} seconds".format(wg.fabrication_time))
    print("Laser path length {:.3f} mm".format(wg.length))


if __name__ == '__main__':
    _example()
