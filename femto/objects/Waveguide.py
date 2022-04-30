import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


class Waveguide:
    def __init__(self, num_scan: int = None, c_max: int = 1200):

        self.num_scan = num_scan
        self.c_max = c_max
        self._M = {}

    @property
    def M(self) -> pd.core.frame.DataFrame:
        """
        COORDINATES MATRIX GETTER.

        The getter returns the coordinates matrix as a pandas.DataFrame object.
        The dataframe is parsed through a unique functions that removes all the
        subsequent identical points in the set.

        Returns
        -------
        pandas.DataFrame
            [X, Y, Z, F, S] unique point matrix.

        """
        return self._unique_points()

    @property
    def x(self) -> np.ndarray:
        """
        X-COORDINATE.

        The getter returns the x-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the x-coordinates.

        """
        df = self._unique_points()
        return df["x"].to_numpy()

    @property
    def y(self) -> np.ndarray:
        """
        Y-COORDINATE.

        The getter returns the y-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the y-coordinates.

        """
        df = self._unique_points()
        return df["y"].to_numpy()

    @property
    def z(self) -> np.ndarray:
        """
        Z-COORDINATE.

        The getter returns the z-coordinate vector as a numpy array. The
        subsequent identical points in the vector are removed.

        Returns
        -------
        numpy.ndarray
            Array of the z-coordinates.

        """
        df = self._unique_points()
        return df["z"].to_numpy()

    @property
    def lastpt(self) -> np.ndarray:
        """
        LAST POINT.

        The function return the last point of the waveguide.

        Returns
        -------
        numpy.ndarray
            Final point [x, y, z].

        """
        if self._M:
            return np.array(
                [self._M["x"][-1], self._M["y"][-1], self._M["z"][-1]]
            )
        else:
            return []

    # Methods
    def start(self, init_pos: List[float]):
        """
        START.

        The function starts a waveguide in the initial position given as
        input.
        The coordinates of the initial position are the first added to the
        matrix that describes the waveguide.

        Parameters
        ----------
        init_pos : List[float]
            Ordered list of coordinate that specifies the waveguide starting
            point [mm].
            init_pos[0] -> X
            init_pos[1] -> Y
            init_pos[2] -> Z

        Returns
        -------
        None.

        """
        assert np.size(init_pos) == 3, (
            "Given initial position is not valid. 3 values are required. "
            f"{np.size(init_pos)} were given."
        )
        assert bool(self._M) is False, (
            "Coordinate matrix is not empty. "
            "Cannot start a new waveguide in this point."
        )

        self._M["x"] = [init_pos[0]]
        self._M["y"] = [init_pos[1]]
        self._M["z"] = [init_pos[2]]
        self._M["f"] = [5]
        self._M["s"] = [0]

    def end(self, speed: float = 75):
        """
        END.

        End a waveguide. The function automatically

        Parameters
        ----------
        speed : float, optional
            DESCRIPTION. The default is 75.

        Returns
        -------
        None.

        """
        self._M["x"].extend([self._M["x"][-1], self._M["x"][0]])
        self._M["y"].extend([self._M["y"][-1], self._M["y"][0]])
        self._M["z"].extend([self._M["z"][-1], self._M["z"][0]])
        self._M["f"].extend([self._M["f"][-1], speed])
        self._M["s"].extend([0, 0])

    def linear(
        self,
        increment: List[float],
        mode: str = "INC",
        speed: float = 0.0,
        shutter: int = 1,
    ):
        """
        LINEAR.

        The function add a linear increment to the last point of the current
        waveguide.


        Parameters
        ----------
        increment : List[float]
            Ordered list of coordinate that specifies the increment if mode
            is INC or new position if mode is ABS. Units are [mm].
            increment[0] -> X-coord
            increment[1] -> Y-coord
            increment[2] -> Z-coord
        mode : str, optional
            Select incremental or absolute mode. The default is 'INC'.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.

        Raises
        ------
        ValueError
            Mode is neither INC nor ABS.

        Returns
        -------
        None.

        """
        if mode.upper() not in ["ABS", "INC"]:
            raise ValueError(
                "Mode should be either ABS or INC.",
                f"{mode.upper()} was given.",
            )
        if mode.upper() == "ABS":
            self._M["x"].append(increment[0])
            self._M["y"].append(increment[1])
            self._M["z"].append(increment[2])
            self._M["f"].append(speed)
            self._M["s"].append(shutter)
        else:
            self._M["x"].append(self._M["x"][-1] + increment[0])
            self._M["y"].append(self._M["y"][-1] + increment[1])
            self._M["z"].append(self._M["z"][-1] + increment[2])
            self._M["f"].append(speed)
            self._M["s"].append(shutter)

    def circ(
        self,
        initial_angle: float,
        final_angle: float,
        radius: float,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 25,
    ):
        """
        CIRC.

        Compute the points in the xy-plane that connects two angles
        (initial_angle and final_angle) with a circular arc of a given radius.
        The user can set the transition speed, the shutter state during
        the movement and the number of points of the arc.

        Parameters
        ----------
        initial_angle : float
            Starting angle of the circular arc [radians].
        final_angle : float
            Ending angle of the circular arc [radians].
        radius : float
            Radius of the circular arc [mm].
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Raises
        ------
        ValueError
            Speed values has the wrong shape.

        Returns
        -------
        None.

        """
        t = np.linspace(initial_angle, final_angle, N)
        new_x = (
            self._M["x"][-1]
            - radius * np.cos(initial_angle)
            + radius * np.cos(t)
        )
        new_y = (
            self._M["y"][-1]
            - radius * np.sin(initial_angle)
            + radius * np.sin(t)
        )
        new_z = self._M["z"][-1] * np.ones(new_x.shape)

        # update coordinates
        self._M["x"].extend(new_x)
        self._M["y"].extend(new_y)
        self._M["z"].extend(new_z)

        # update speed array
        if np.size(speed) == 1:
            self._M["f"].extend(speed * np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._M["f"].extend(speed)
        else:
            raise ValueError(
                "Speed array is neither a single value nor ",
                "array of appropriate size.",
            )

        self._M["s"].extend(shutter * np.ones(new_x.shape))

    def arc_bend(
        self,
        D: float,
        radius: float,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 25,
    ):
        """
        CIRCULAR BEND.

        The function concatenate two circular arc to make a circular S-bend.
        The user can specify the amplitude of the S-bend (height in the y
        direction) and the curvature radius. Starting and ending angle of the
        two arcs are computed automatically.
        The sign of D encodes the direction of the S-bend:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        D : float
            Amplitude of the S-bend along the y direction [mm].
        radius : float
            Curvature radius of the S-bend [mm].
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Returns
        -------
        None.

        """
        (a, _) = self._get_sbend_parameter(D, radius)

        if D > 0:
            self.circ(
                np.pi * (3 / 2),
                np.pi * (3 / 2) + a,
                radius,
                speed=speed,
                shutter=shutter,
                N=np.round(N / 2),
            )
            self.circ(
                np.pi * (1 / 2) + a,
                np.pi * (1 / 2),
                radius,
                speed=speed,
                shutter=shutter,
                N=np.round(N / 2),
            )
        else:
            self.circ(
                np.pi * (1 / 2),
                np.pi * (1 / 2) - a,
                radius,
                speed=speed,
                shutter=shutter,
                N=np.round(N / 2),
            )
            self.circ(
                np.pi * (3 / 2) - a,
                np.pi * (3 / 2),
                radius,
                speed=speed,
                shutter=shutter,
                N=np.round(N / 2),
            )

    def arc_acc(
        self,
        D: float,
        radius: float,
        arm_length: float = 0.0,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 50,
    ):
        """
        CIRCULAR COUPLER.

        The function concatenate two circular S-bend to make a single mode of
        a circular directional coupler.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        D : float
            Amplitude of the coupler along the y-direction [mm].
        radius : float
            Curvature radius of the coupler's bends [mm].
        arm_length : float, optional
            Length of the coupler straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 50.

        Returns
        -------
        None.

        """
        self.arc_bend(D, radius, speed=speed, shutter=shutter, N=N / 2)
        self.linear([arm_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_bend(-D, radius, speed=speed, shutter=shutter, N=N / 2)

    def arc_mzi(
        self,
        D: float,
        radius: float,
        int_length: float = 0.0,
        arm_length: float = 0.0,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 100,
    ):
        """
        CIRCULAR MACH-ZEHNDER INTERFEROMETER (MZI).

        The function concatenate two circular couplers to make a single mode
        of a circular MZI.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        D : float
            Amplitude of the MZI along the y-direction [mm].
        radius : float
            Curvature radius of the MZI's bends [mm].
        int_length : float, optional
            Interaction distance of the MZI [mm]. The default is 0.0.
        arm_length : float, optional
            Length of the MZI straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 100.

        Returns
        -------
        None.

        """
        self.arc_acc(
            D,
            radius,
            arm_length=arm_length,
            speed=speed,
            shutter=shutter,
            N=N / 2,
        )
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.arc_acc(
            D,
            radius,
            arm_length=arm_length,
            speed=speed,
            shutter=shutter,
            N=N / 2,
        )

    def sin_bend(
        self,
        D: float,
        radius: float,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 25,
    ):
        """
        SINUSOIDAL BEND.

        The function compute the points in the xy-plane of a Sin-bend curve.
        The distance between the initial and final point is the same of the
        equivalent (circular) S-bend of given radius.
        The user can specify the amplitude of the Sin-bend (height in the y
        direction) and the curvature radius as well as the transition speed,
        the shutter state during the movement and the number of points of the
        arc.
        The sign of D encodes the direction of the Sin-bend:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        NB: the radius is an effective radius. The radius of curvature of the
            overall curve will be lower (in general) than the specified
            radius.

        Parameters
        ----------
        D : float
            Amplitude of the Sin-bend along the y direction [mm].
        radius : float
            Effective curvature radius of the Sin-bend [mm].
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 25.

        Raises
        ------
        ValueError
            Speed values has the wrong shape.

        Returns
        -------
        None.

        """
        (a, dx) = self._get_sbend_parameter(D, radius)

        new_x = np.arange(self._M["x"][-1], self._M["x"][-1] + dx, dx / (N - 1))
        new_y = self._M["y"][-1] + 0.5 * D * (
            1 - np.cos(np.pi / dx * (new_x - self._M["x"][-1]))
        )
        new_z = self._M["z"][-1] * np.ones(new_x.shape)

        # update coordinates
        self._M["x"].extend(new_x)
        self._M["y"].extend(new_y)
        self._M["z"].extend(new_z)

        # update speed array
        if np.size(speed) == 1:
            self._M["f"].extend(speed * np.ones(new_x.shape))
        elif np.size(speed) == np.size(new_x):
            self._M["f"].extend(speed)
        else:
            raise ValueError(
                "Speed array is neither a single value nor ",
                "array of appropriate size.",
            )

        self._M["s"].extend(shutter * np.ones(new_x.shape))

    def sin_acc(
        self,
        D: float,
        radius: float,
        int_length: float = 0.0,
        speed: float = 0.0,
        shutter: int = 1,
        N: int = 50,
    ):
        """
        SINUSOIDAL COUPLER.

        The function concatenate two Sin-bend to make a single mode of a
        sinusoidal directional coupler.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the effective curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward Sin-bend
            - D < 0, downward Sin-bend

        Parameters
        ----------
        D : float
            Amplitude of the coupler along the y-direction [mm].
        radius : float
            Effective curvature radius of the coupler's bends [mm].
        int_length : float, optional
            Length of the coupler straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 50.

        Returns
        -------
        None.

        """
        self.sin_bend(D, radius, speed=speed, shutter=shutter, N=N / 2)
        self.linear([int_length, 0, 0], speed=speed, shutter=shutter)
        self.sin_bend(-D, radius, speed=speed, shutter=shutter, N=N / 2)

    def sin_mzi(
        self,
        D: float,
        radius: float,
        int_length: float = 0.0,
        arm_length: float = 0.0,
        speed: float = 0.0,
        shutter: int = 1,
        N: float = 100,
    ):
        """
        SINUSOIDAL MACH-ZEHNDER INTERFEROMETER (MZI).

        The function concatenate two sinusoidal couplers to make a single mode
        of a sinusoidal MZI.
        The user can specify the amplitude of the coupler (height in the y
        direction) and the curvature radius.
        The sign of D encodes the direction of the coupler:
            - D > 0, upward S-bend
            - D < 0, downward S-bend

        Parameters
        ----------
        D : float
            Amplitude of the MZI along the y-direction [mm].
        radius : float
            Effective curvature radius of the coupler's bends [mm].
        int_length : float, optional
            Interaction distance of the MZI [mm]. The default is 0.0.
        arm_length : float, optional
            Length of the MZI straight arm [mm]. The default is 0.0.
        speed : float, optional
            Transition speed [mm/s]. The default is 0.0.
        shutter : int, optional
            State of the shutter [0: 'OFF', 1: 'ON']. The default is 1.
        N : int, optional
            Number of points. The default is 100.

        Returns
        -------
        None.

        """
        self.sin_acc(
            D,
            radius,
            int_length=int_length,
            speed=speed,
            shutter=shutter,
            N=N / 2,
        )
        self.linear([arm_length, 0, 0], speed=speed, shutter=shutter)
        self.sin_acc(
            D,
            radius,
            int_length=int_length,
            speed=speed,
            shutter=shutter,
            N=N / 2,
        )

    def curvature(self) -> np.ndarray:
        """
        CURVARURE.

        Compute the 3D point-to-point curvature radius of the waveguide
        shape.

        Returns
        -------
        curvature : numpy.ndarray
            Array of the curvature radii computed at each point of the curve.

        """
        data = self._unique_points()

        x = np.array(data["x"])
        y = np.array(data["y"])
        z = np.array(data["z"])

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        num = (dx_dt**2 + dy_dt**2 + dz_dt**2) ** 1.5
        den = np.sqrt(
            (d2z_dt2 * dy_dt - d2y_dt2 * dz_dt) ** 2
            + (d2x_dt2 * dz_dt - d2z_dt2 * dx_dt) ** 2
            + (d2y_dt2 * dx_dt - d2x_dt2 * dy_dt) ** 2
        )
        default_zero = np.ones(np.size(num)) * np.inf
        # only divide nonzeros else Inf
        curvature = np.divide(num, den, out=default_zero, where=(den != 0))
        return curvature

    def cmd_rate(self) -> np.ndarray:
        """
        COMMAND RATE.

        Compute the point-to-point command rate of the waveguide shape.

        Returns
        -------
        cmd_rate : numpy.ndarray
            Array of the command rates computed at each point of the curve.

        """
        data = self._unique_points()

        # exclude last point, it's there just to close the shutter
        x = np.array(data["x"][:-1])
        y = np.array(data["y"][:-1])
        z = np.array(data["z"][:-1])
        v = np.array(data["f"][:-1])

        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        dz_dt = np.gradient(z)
        dt = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

        default_zero = np.ones(np.size(dt)) * np.inf
        # only divide nonzeros else Inf
        cmd_rate = np.divide(v, dt, out=default_zero, where=(v != 0))
        return cmd_rate

    # Private interface
    def _get_sbend_parameter(self, D: float, radius: float) -> tuple:
        """
        GET S-BEND PARAMETERS.

        The function computes the final angle, and x-displacement for a
        circular S-bend given the y-displacement D and curvature radius.

        Parameters
        ----------
        D : float
            Displacement along y-direction [mm].
        radius : float
            Curvature radius of the S-bend [mm]..

        Returns
        -------
        tuple
            (final angle, x-displacement), ([radians], [mm]).

        """
        dy = np.abs(D / 2)
        a = np.arccos(1 - (dy / radius))
        dx = 2 * radius * np.sin(a)
        return (a, dx)

    def _unique_points(self):
        """
        REMOVE ALL CONSECUTIVE DUPLICATES.

        At least one coordinate (X,Y,Z,F,S) have to change between two
        consecutive lines.

        Duplicates can be selected by crating a boolean index mask as follow:
            - make a row-wise diff operation (data.diff)
            - compute absolute value of all elements in order to work only
                with positive numbers
            - make a column-wise sum (.sum(axis=1))
        In this way consecutive duplicates correspond to a 0.0 value in the
        latter array.
        Converting this array to boolean (all non-zero values are True) the
        index mask can be retrieved.
        The first element is set to True by default since it is lost by the
        diff operation.
        Also indexes are reset to the new dataframe (with less element, in
        principle).

        Returns
        -------
        pandas DataFrame
            Coordinate dataframe (x, y, z, f, s).

        """

        data = pd.DataFrame.from_dict(self._M)
        mask = (
            data.diff()  # row-wise diff
            .abs()  # abs of each element
            .sum(axis=1)  # col-wise sum
            .astype("bool")
        )  # cast to bool
        mask[0] = True
        return data[mask].reset_index()

    def _compute_number_points(self):
        # TODO: write method that compute the optimal number of points given
        #       the max value of cmd rate
        pass


if __name__ == "__main__":

    # Data
    pitch = 0.080
    int_dist = 0.007
    d_bend = 0.5 * (pitch - int_dist)
    increment = [4, 0, 0]

    # Calculations
    mzi = [Waveguide() for _ in range(2)]
    for index, wg in enumerate(mzi):
        [xi, yi, zi] = [-2, -pitch / 2 + index * pitch, 0.035]

        wg.start([xi, yi, zi])
        wg.linear(increment, speed=20)
        wg.sin_mzi((-1) ** index * d_bend, radius=15, speed=20)
        wg.linear(increment, speed=20)
        wg.end()

    print(wg.M)

    # Plot
    fig, ax = plt.subplots()
    for wg in mzi:
        ax.plot(wg.M["x"][:-1], wg.M["y"][:-1], "-k", linewidth=2.5)
        ax.plot(wg.M["x"][-2:], wg.M["y"][-2:], ":b", linewidth=1.0)
    plt.show()
