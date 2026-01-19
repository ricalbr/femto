from __future__ import annotations

import dataclasses
import functools
from typing import Any

import numpy as np
from femto.helpers import dotdict
from femto.laserpath import LaserPath


@dataclasses.dataclass(repr=False)
class Waveguide(LaserPath):
    """Class that computes and stores the coordinates of an optical waveguide."""

    depth: float = 0.035  #: Distance for sample's bottom facet
    radius: float = 15  #: Curvature radius
    pitch: float = 0.080  #: Distance between adjacent modes
    pitch_fa: float = 0.127  #: Distance between fiber arrays' adjacent modes (for fan-in, fan-out)
    int_dist: float | None = None  #: Directional Coupler's interaction distance
    int_length: float = 0.0  #: Directional Coupler's interaction length
    arm_length: float = 0.0  #: Mach-Zehnder interferometer's length of central arm
    dz_bridge: float = 0.007  #: Maximum `z`-height for 3D bridges
    ltrench: float = 0.0  #: Length of straight segment to accomodate trenches

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.z_init is None:
            self.z_init = self.depth

        # Adjust the pitch for the glass shrinking. Only change the external pitch in case of Fan-IN and Fan-out
        # segments.
        if self.pitch == self.pitch_fa:
            self.pitch /= self.shrink_correction_factor
        self.pitch_fa /= self.shrink_correction_factor

    @property
    def dy_bend(self) -> float:
        """`y`-displacement of an S-bend.

        Returns
        -------
        float
            The difference between the waveguide pitch and the interaction distance.
        """
        if self.pitch is None:
            raise ValueError('Waveguide pitch is set to None.')
        if self.int_dist is None:
            raise ValueError('Interaction distance is set to None.')
        return 0.5 * (self.pitch - self.int_dist)

    @property
    # Defining a function that calculates the length of a sbend.
    def dx_bend(self) -> float:
        """`x`-displacement of an S-bend.

        Returns
        -------
        float

        """

        return float(self.get_sbend_parameter(self.dy_bend, self.radius)[1])

    @property
    def dx_coupler(self) -> float:
        """`x`-displacement of a Directional Coupler.

        Returns
        -------
        float
            Sum of two `x`-displacements S-bend segments and the interaction distance straight segment.
        """
        return 2 * self.dx_bend + self.int_length

    @property
    def dx_mzi(self) -> float:
        """`x`-displacement of a Mach-Zehnder Interferometer.

        Returns
        -------
        float
            Sum of two `x`-displacements Directional Coupler segments and the ``arm_length`` distance straight segment.
        """
        return 4 * self.dx_bend + 2 * self.int_length + self.arm_length

    @staticmethod
    def get_sbend_parameter(dy: float | None, radius: float | None) -> tuple[float, float]:
        """Compute the rotation angle, and `x`-displacement for a circular S-bend.

        Parameters
        ----------
        dy: float, optional
            Displacement along `y`-direction [mm].
        radius: float, optional
            Curvature radius of the S-bend [mm]. The default value is `self.radius`

        Returns
        -------
        tuple(float, float)
            rotation angle [rad], `x`-displacement [mm].
        """

        if radius is None or radius <= 0:
            raise ValueError(f'Radius should be a positive value. Given {radius}.')
        if dy is None:
            raise ValueError('dy is None. Give a valid input valid.')

        a = np.arccos(1 - (np.abs(dy / 2) / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    # Methods
    def circ(
        self,
        initial_angle: float,
        final_angle: float,
        radius: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Add a circular curved path to the waveguide.

        Computes the points in the xy-plane that connects two angles (initial_angle and final_angle) with a circular
        arc of a given radius. The transition speed and the shutter state during the movement can be given as input.

        Parameters
        ----------
        initial_angle: float
            Starting angle of the circular arc [radians].
        final_angle: float
            Final angle of the circular arc [radians].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.
        """

        if radius is None and self.radius is None:
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        r = radius if radius is not None else self.radius
        if r < 0:
            raise ValueError('Radius is negative. Set Waveguide\'s "radius" attribute or give a radius as input.')

        f = speed if speed is not None else self.speed
        if f is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')

        delta_angle = final_angle - initial_angle
        num = self.num_subdivisions(np.fabs(delta_angle * r), f)

        t = np.linspace(initial_angle, final_angle, num)
        x_circ = self._x[-1] + np.fabs(r) * (-np.cos(initial_angle) + np.cos(t))
        y_circ = self._y[-1] + np.fabs(r) * (-np.sin(initial_angle) + np.sin(t))
        z_circ = np.repeat(self._z[-1], num)
        f_circ = np.repeat(f, num)
        s_circ = np.repeat(shutter, num)

        # update coordinates
        self.add_path(x_circ, y_circ, z_circ, f_circ, s_circ)
        return self

    def arc_bend(
        self,
        dy: float,
        radius: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Concatenate two circular arc to make a circular S-bend.

        The vertical displacement of the S-bend and the curvature radius are given as input.
        Starting and ending angles of the arcs are computed automatically.

        The sign of `dy` encodes the direction of the S-bend:

        - `dy` > 0, upward S-bend
        - `dy` < 0, downward S-bend

        Parameters
        ----------
        dy: float
            Vertical displacement of the waveguide of the S-bend [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.
        """

        r = radius if radius is not None else self.radius
        a, _ = self.get_sbend_parameter(dy, r)

        if dy > 0:
            self.circ(
                np.pi * (3 / 2),
                np.pi * (3 / 2) + a,
                radius=r,
                speed=speed,
                shutter=shutter,
            )
            self.circ(
                np.pi * (1 / 2) + a,
                np.pi * (1 / 2),
                radius=r,
                speed=speed,
                shutter=shutter,
            )
        else:
            self.circ(
                np.pi * (1 / 2),
                np.pi * (1 / 2) - a,
                radius=radius,
                speed=speed,
                shutter=shutter,
            )
            self.circ(
                np.pi * (3 / 2) - a,
                np.pi * (3 / 2),
                radius=radius,
                speed=speed,
                shutter=shutter,
            )
        return self

    def arc_coupler(
        self,
        dy: float,
        radius: float | None = None,
        int_length: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Concatenates two circular S-bend to make a single mode of a circular Directional Coupler.

        Parameters
        ----------
        dy: float
            Vertical displacement of the waveguide of the S-bend [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        circ : Add a circular curved path to the waveguide.
        arc_bend : Concatenate two circular arc to make a circular S-bend.
        """
        if int_length is None and self.int_length is None:
            raise ValueError(
                'Interaction length is None. Set Waveguide\'s "int_length" attribute or give a valid '
                'interaction length as input.'
            )

        int_length = int_length if int_length is not None else self.int_length
        self.arc_bend(dy, radius=radius, speed=speed, shutter=shutter)
        self.linear([np.fabs(int_length), 0, 0], speed=speed, shutter=shutter, mode='INC')
        self.arc_bend(-dy, radius=radius, speed=speed, shutter=shutter)
        return self

    def arc_mzi(
        self,
        dy: float,
        radius: float | None = None,
        int_length: float | None = None,
        arm_length: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Concatenates two circular Directional Couplers curves to make a single mode of a circular Mach-Zehnder
        Interferometer.

        Parameters
        ----------
        dy: float
            Vertical displacement of the waveguide of the S-bend [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        arm_length: float
            Length of the Mach-Zehnder Interferometer's straight arm [mm]. The default is `self.arm_length`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        circ : Add a circular curved path to the waveguide.
        arc_bend : Concatenate two circular arc to make a circular S-bend.
        arc_coupler : Concatenates two circular S-bend to make a single mode of a circular Directional Coupler.
        """

        if arm_length is None and self.arm_length is None:
            raise ValueError(
                'Arm length is None. Set Waveguide\'s "arm_length" attribute or give a valid ' 'arm length as ' 'input.'
            )

        arm_length = arm_length if arm_length is not None else self.arm_length

        self.arc_coupler(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        self.linear([np.fabs(arm_length), 0, 0], speed=speed, shutter=shutter, mode='INC')
        self.arc_coupler(dy, radius=radius, int_length=int_length, speed=speed, shutter=shutter)
        return self

    # TODO: test disp_x
    # TODO: togliere dz optional
    def sin_bridge(
        self,
        dy: float,
        dz: float | None = None,
        disp_x: float | None = None,
        flat_peaks: float = 0.0,
        omega: tuple[float, float] = (1.0, 2.0),
        radius: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Add a sinusoidal curved path to the waveguide.

        Combines sinusoidal bend curves in the `xy`- and `xz`-plane. The `x`-displacement between the initial and
        final point is identical to the one of the (circular) S-bend computed with the same radius.

        The sign of the `y`, `z` displacement encodes the direction of the sin-bend:

        - `d` > 0, upward sin-bend
        - `d` < 0, downward sin-bend

        Parameters
        ----------
        dy: float
            `y` displacement of the sinusoidal-bend [mm].
        dz: float, optional
            `z` displacement of the sinusoidal-bend [mm]. The default values is `self.dz_bridge.
        disp_x: float, optional
            `x`-displacement  for the sinusoidal bend. If the value is ``None`` (the default value),
            the `x`-displacement is computed with the formula for the circular `S`-bend.
        flat_peaks: float
            Parameter that regulates the flatness of the sine's peaks. The higher the parameter the more the sine
            function resembles a square function. If ``flat_peaks == 0`` the sine function does not have flat peaks.
            The default value is `0`.
        omega: tuple(float, float)
            Frequency of the Sin-bend oscillations for `y` and `z` coordinates, respectively.
            The deafult values are `fy` = 1, `fz` = 1.
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        Notes
        -----
            The radius is an *effective* radius. The given radius is used to compute the `x` displacement using the
            `get_sbend_parameter` method. The radius of curvature of the overall curve will be lower (in general)
            than the specified radius.

        See Also
        --------
        get_sbend_parameter : Compute the rotation angle, and `x`-displacement for a circular S-bend.
        """

        if radius is None and self.radius is None:
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        if speed is None and self.speed is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')
        if dz is None and self.dz_bridge is None:
            raise ValueError('dz bridge is None. Set Waveguide\'s "dz_bridge" attribute or give a valid dz as input.')
        if dy is None:
            raise ValueError('dy is None. Give a valid dy as input.')

        omega_y, omega_z = omega
        r = radius if radius is not None else self.radius
        f = speed if speed is not None else self.speed
        dzb = dz if dz is not None else self.dz_bridge

        dx = disp_x if disp_x is not None else self.get_sbend_parameter(dy, r)[-1]
        num = self.num_subdivisions(dx, f)

        x_sin = np.linspace(self._x[-1], self._x[-1] + dx, num)
        tmp_cos = np.cos(omega_y * np.pi / dx * (x_sin - self._x[-1]))
        y_sin = self._y[-1] + 0.5 * dy * (1 - np.sqrt((1 + flat_peaks**2) / (1 + flat_peaks**2 * tmp_cos**2)) * tmp_cos)
        z_sin = self._z[-1] + 0.5 * dzb * (1 - np.cos(omega_z * np.pi / dx * (x_sin - self._x[-1])))
        f_sin = np.repeat(f, num)
        s_sin = np.repeat(shutter, num)

        # update coordinates
        self.add_path(x_sin, y_sin, z_sin, f_sin, s_sin)
        return self

    sin_bend = functools.partialmethod(sin_bridge, dz=0.0, omega=(1.0, 2.0))
    sin_comp = functools.partialmethod(sin_bridge, dz=0.0, omega=(2.0, 2.0))

    def sin_coupler(
        self,
        dy: float,
        radius: float | None = None,
        flat_peaks: float = 0.0,
        int_length: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Concatenates two sinusoidal S-bend to make a single mode of a circular Directional Coupler.

        Parameters
        ----------
        dy: float
            Vertical displacement of the waveguide of the sinusoidal-bend [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        flat_peaks: float
            Parameter that regulates the flatness of the sine's peaks. The higher the parameter the more the sine
            function resembles a square function. If ``flat_peaks == 0`` the sine function does not have flat peaks.
            The default value is `0`.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        sin_bridge : Add a sinusoidal curved path to the waveguide.
        sin_bend : Concatenate two circular arc to make a sinusoidal S-bend.
        """

        if int_length is None and self.int_length is None:
            raise ValueError(
                'Interaction length is None.'
                'Set Waveguide\'s "int_length" attribute or give a valid "int_length" as input.'
            )

        int_length = int_length if int_length is not None else self.int_length

        self.sin_bend(dy, radius=radius, flat_peaks=flat_peaks, speed=speed, shutter=shutter)
        self.linear([np.fabs(int_length), 0, 0], speed=speed, shutter=shutter)
        self.sin_bend(-dy, radius=radius, flat_peaks=flat_peaks, speed=speed, shutter=shutter)
        return self

    def sin_mzi(
        self,
        dy: float,
        radius: float | None = None,
        flat_peaks: float = 0.0,
        int_length: float | None = None,
        arm_length: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Concatenates two sinusoidal Directional Couplers curves to make a single mode of a circular Mach-Zehnder
        Interferometer.

        Parameters
        ----------
        dy: float
            Vertical displacement of the waveguide of the sinusoidal-bend [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        flat_peaks: float
            Parameter that regulates the flatness of the sine's peaks. The higher the parameter the more the sine
            function resembles a square function. If ``flat_peaks == 0`` the sine function does not have flat peaks.
            The default value is `0`.
        int_length: float, optional
            Length of the Directional Coupler's straight interaction region [mm]. The default is `self.int_length`.
        arm_length: float, optional
            Length of the Mach-Zehnder Interferometer's straight arm [mm]. The default is `self.arm_length`.
        shutter: int, optional
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        sin_bridge : Add a sinusoidal curved path to the waveguide.
        sin_bend : Concatenate two circular arc to make a sinusoidal S-bend.
        sin_coupler : Concatenates two sinusoidal S-bend to make a single mode of a circular Directional Coupler.
        """

        if arm_length is None and self.arm_length is None:
            raise ValueError(
                'Arm length is None. Set Waveguide\'s "arm_length" attribute or give a valid "arm_length" as input.'
            )

        arm_length = arm_length if arm_length is not None else self.arm_length

        self.sin_coupler(dy, radius=radius, flat_peaks=flat_peaks, int_length=int_length, shutter=shutter, speed=speed)
        self.linear([np.fabs(arm_length), 0, 0], shutter=shutter, speed=speed)
        self.sin_coupler(dy, radius=radius, flat_peaks=flat_peaks, int_length=int_length, shutter=shutter, speed=speed)
        return self

    def spline(
        self,
        dy: float,
        dz: float = 0.0,
        disp_x: float | None = None,
        y_derivatives: tuple[tuple[float]] = ((0.0, 0.0), (0.0, 0.0)),
        z_derivatives: tuple[tuple[float]] = ((0.0, 0.0), (0.0, 0.0)),
        radius: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """
        The function construct a piecewise 3D polynomial in the Bernstein basis, compatible with the specified values
        and derivatives at breakpoints.
        The user can specify the `y` and `z` displacements as well as the values of the first and second derivatives
        at the initial and final points (separaterly for the `y` and `z` coordinates).

        Parameters
        ----------
        dy: float
            `y`-displacement from initial position [mm].
        dz: float, optional
            `z`-displacement from initial position [mm]. The default value is 0.0.
        disp_x: float, optional
            The displacement along the x-axis. If not specified, it is calculated using the`get_sbend_parameter`
            method of the `Waveguide` class.
        y_derivatives   : tuple(tuple(float))
            Tuple containing the derivates for the `y` coordinate for the initial and final point. The number of
            derivatives is arbitrary. For example ``y_derivatives=((0.0, 1.0, 2.0), (0.0,-1.0,-0.2))`` generates a
            polynomial spline curve `f(x)` such that `f'(x0) = 0.0`, `f''(x0) = 1.0`, `f'''(x0) = 2.0`, `f'(x0+dx) =
            0.0`, `f''(x0+dx) = -1.0` and `f'''(x0+dx) = -0.2`. The default value is `((0.0, 0.0), (0.0, 0.0))`.
        z_derivatives   : tuple(tuple(float))
            Tuple containing the derivates for the `z` coordinate for the initial and final point. The number of
            derivatives is arbitrary. For example ``z_derivatives=((0.0, 1.0, 2.0), (0.0,-1.0,-0.2))`` generates a
            polynomial spline curve `f(x)` such that `f'(x0) = 0.0`, `f''(x0) = 1.0`, `f'''(x0) = 2.0`, `f'(x0+dx) =
            0.0`, `f''(x0+dx) = -1.0` and `f'''(x0+dx) = -0.2`. The default value is `((0.0, 0.0), (0.0, 0.0))`.
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        scipy.interpolate.BPoly.from_derivatives : Construct piecewise polymonial from derivatives at breakpoints.
        """

        if radius is None and self.radius is None:
            raise ValueError('Radius is None. Set Waveguide\'s "radius" attribute or give a radius as input.')
        if speed is None and self.speed is None:
            raise ValueError('Speed is None. Set Waveguide\'s "speed" attribute or give a speed as input.')
        if dy is None:
            raise ValueError('dy is None. Give a valid dy as input.')
        if dz is None:
            raise ValueError('dz is None. Give a valid dz as input.')

        r = radius if radius is not None else self.radius
        f = speed if speed is not None else self.speed

        dx = disp_x if disp_x is not None else self.get_sbend_parameter(np.sqrt(dy**2 + dz**2), r)[-1]
        num = self.num_subdivisions(dx, f)

        # Define initial and final point of the curve
        x0, x1 = self._x[-1], self._x[-1] + dx
        y0, y1 = self._y[-1], self._y[-1] + dy
        z0, z1 = self._z[-1], self._z[-1] + dz

        from scipy.interpolate import BPoly

        y_poly = BPoly.from_derivatives([x0, x1], [[y0, *y_derivatives[0]], [y1, *y_derivatives[-1]]])
        z_poly = BPoly.from_derivatives([x0, x1], [[z0, *z_derivatives[0]], [z1, *z_derivatives[-1]]])

        x_poly = np.linspace(x0, x1, num)
        y_poly = y_poly(x_poly)
        z_poly = z_poly(x_poly)
        f_poly = np.repeat(f, num)
        s_poly = np.repeat(shutter, num)

        # update coordinates
        self.add_path(x_poly, y_poly, z_poly, f_poly, s_poly)
        return self

    poly_bend = functools.partialmethod(
        spline,
        y_derivatives=((0.0, 0.0), (0.0, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )

    def spline_bridge(
        self,
        dy: float,
        dz: float,
        disp_x: float | None = None,
        radius: float | None = None,
        shutter: int = 1,
        speed: float | None = None,
    ) -> Waveguide:
        """Connect two points in the `xy` plane with a Bezier curve bridge.

        It takes in an initial position (`x_0`, `y_0`, `z_0`) and linear displacements in the
        `x`, `y`, and `z` directions. The final point of the curved is computed as
        (`x_0 +d_x`, `y_0 +d_y`, `z_0 +d_z`). The points are connected with a sequence of two
        spline segments.

        The spline segments join at the peak of the bridge. In this point it is required to have a derivative
        along the `z` direction of `df(x, z)/dz = 0` and a derivative along the `y` direction that is contiunous. This
        latter value is fixed to be `df(x, y)/dx = disp_y/disp_y`.
        The values of the first derivatives df(x, y)/dx, df(x, z)/dx are set to zero in the initial and final point
        of the spline bridge.

        Parameters
        ----------
        dy: float
            `y`-displacement from initial position [mm].
        dz: float
            `z`-displacement from initial position [mm].
        disp_x: float, optional
            `x`-displacement from initial position [mm].
        radius: float, optional
            Curvature radius [mm]. The default value is `self.radius`.
        shutter: int
            State of the shutter during the transition (0: 'OFF', 1: 'ON'). The default value is 1.
        speed: float, optional
            Translation speed [mm/s]. The default value is `self.speed`.

        Returns
        -------
        The object itself.

        See Also
        --------
        CubicSpline : Cubic spline data interpolator.
        InterpolatedUnivariateSpline : 1-D interpolating spline for a given set of data points.
        spline : Connect the current position to a new point with a Bezier curve.
        """
        if dy is None:
            raise ValueError('dy is None. Give a valid dy as input.')
        if dz is None:
            raise ValueError('dz is None. Give a valid dz as input.')

        r = radius if radius is not None else self.radius
        dx = disp_x if disp_x is not None else self.get_sbend_parameter(np.sqrt(dy**2 + dz**2), r)[-1]

        # First half of the spline bridge
        self.spline(
            dy=dy / 2,
            dz=dz,
            disp_x=dx,
            y_derivatives=((0.0, 0.0), (dy / dx, 0.0)),
            z_derivatives=((0.0, 0.0), (0.0, 0.0)),
            radius=radius,
            shutter=shutter,
            speed=speed,
        )
        # Second half of the spline bridge
        self.spline(
            dy=dy / 2,
            dz=-dz,
            disp_x=dx,
            y_derivatives=((dy / dx, 0.0), (0.0, 0.0)),
            z_derivatives=((0.0, 0.0), (0.0, 0.0)),
            radius=radius,
            shutter=shutter,
            speed=speed,
        )
        return self


@dataclasses.dataclass
class NasuWaveguide(Waveguide):
    """Class that computes and stores the coordinates of a Nasu optical waveguide [#]_.

    References
    ----------
    .. [#] `Nasu Waveguides <https://opg.optica.org/ol/abstract.cfm?uri=ol-30-7-723>`_ on Optics Letters.
    """

    adj_scan_shift: tuple[float, float, float] = (0, 0.0004, 0)  #: (`x`, `y`, `z`)-shifts between adjacent passes
    adj_scan: int = 5  #: Number of adjacent scans

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.adj_scan, int):
            raise ValueError(f'Number of adjacent scan must be of type int. Given type is {type(self.scan).__name__}.')

    @property
    def adj_scan_order(self) -> list[float]:
        """Scan order.

        Given the number of adjacent scans it computes the list of adjacent scans as:

        * if `self.adj_scan` is odd

            ``[0, 1, -1, 2, -2, ...]``

        * if `self.adj_scan` is even

            ``[0.5, -0.5, 1.5, -1.5, 2.5, -2.5, ...]``

        Returns
        -------
        list(float)
            Ordered list of adjacent scans.
        """

        adj_scan_list = []
        if self.adj_scan % 2:
            adj_scan_list.append(0.0)
            for i in range(1, self.adj_scan // 2 + 1):
                adj_scan_list.extend([i, -i])
        else:
            for i in range(0, self.adj_scan // 2):
                adj_scan_list.extend([i + 0.5, -i - 0.5])
        return adj_scan_list


def coupler(param: dict[str, Any], nasu: bool = False) -> list[Waveguide | NasuWaveguide]:
    """
    Directional coupler.

    Creates the two modes of a Directional Coupler. Waveguides can be standard multi-scan waveguides or Nasu
    Waveguides. The interaction region of the coupler is in the center of the sample.

    Parameters
    ----------
    param : dict
        Set of Waveguide parameters.
    nasu : bool, optional
        Flag value for selecting Nasu Waveguide over standard multi-scan Waveguides. The default value is False.

    Returns
    -------
    list(Waveguide) or list(NasuWaveguide)
        List of the two modes of the Directional Coupler.
    """

    mode1 = NasuWaveguide(**param) if nasu else Waveguide(**param)
    mode2 = NasuWaveguide(**param) if nasu else Waveguide(**param)

    lx = (mode1.samplesize[0] - mode1.dx_coupler) / 2

    mode1.start()
    mode1.linear([lx, None, None], mode='ABS')
    mode1.sin_coupler(mode1.dy_bend)
    mode1.linear([mode1.x_end, None, None], mode='ABS')
    mode1.end()

    mode2.y_init = mode1.y_init + mode2.pitch
    mode2.start()
    mode2.linear([lx, None, None], mode='ABS')
    mode2.sin_coupler(-mode2.dy_bend)
    mode2.linear([mode2.x_end, None, None], mode='ABS')
    mode2.end()

    return [mode1, mode2]


def main() -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Data
    PARAM_WG = dotdict(scan=6, speed=20, radius=15, pitch=0.080, int_dist=0.007, lsafe=3, samplesize=(50, 3))

    increment = [5.0, 0, 0]

    # Calculations
    mzi = []
    for index in range(2):
        wg = Waveguide(**PARAM_WG)
        wg.y_init = -wg.pitch / 2 + index * wg.pitch
        wg.start()
        wg.linear(increment)
        wg.poly_bend((-1) ** index * wg.dy_bend, flat_peaks=0)
        wg.sin_bridge((-1) ** index * 0.08, (-1) ** index * 0.015)
        wg.arc_bend((-1) ** (index + 1) * wg.dy_bend)
        wg.linear(increment)
        wg.end()
        mzi.append(wg)

    # Plot
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]'), ax.set_zlabel('Z [mm]')
    for wg in mzi:
        ax.plot(wg.x[:-1], wg.y[:-1], wg.z[:-1], '-k', linewidth=2.5)
        ax.plot(wg.x[-2:], wg.y[-2:], wg.z[-2:], ':b', linewidth=1.0)
    ax.set_box_aspect(aspect=(3, 1, 0.5))
    plt.show()

    print(f'Expected writing time {sum(wg.fabrication_time for wg in mzi):.3f} seconds')
    print(f'Laser path length {mzi[0].length:.3f} mm')


if __name__ == '__main__':
    main()
