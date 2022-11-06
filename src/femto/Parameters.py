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


@dataclass(kw_only=True)
class LaserPathParameters:
    """
    Class containing the parameters for generic FLM written structure fabrication.
    """

    scan: int = 1
    speed: float = 1.0
    x_init: float = -2.0
    y_init: float = 0.0
    z_init: float = None
    lsafe: float = 2.0
    speed_closed: float = 5
    speed_pos: float = 0.5
    cmd_rate_max: float = 1200
    acc_max: float = 500
    samplesize: Tuple[float, float] = (None, None)

    def __post_init__(self):
        if not isinstance(self.scan, int):
            raise ValueError('Number of scan must be integer.')

    @property
    def init_point(self):
        z0 = self.z_init if self.z_init else 0.0
        return [self.x_init, self.y_init, z0]

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


@dataclass(kw_only=True)
class WaveguideParameters(LaserPathParameters):
    """
    Class containing the parameters for the waveguide fabrication.
    """

    depth: float = 0.035
    radius: float = 15
    pitch: float = 0.080
    pitch_fa: float = 0.127
    int_dist: float = None
    int_length: float = 0.0
    arm_length: float = 0.0
    ltrench: float = 1.0
    dz_bridge: float = 0.007
    margin: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        # parent method redefinition, to include depth property
        if not isinstance(self.scan, int):
            print(self.scan)
            raise ValueError('Number of scan must be integer.')

        if self.z_init is None:
            self.z_init = self.depth

    @property
    def dy_bend(self):
        if self.pitch is None:
            raise ValueError('Waveguide pitch is set to None.')
        if self.int_dist is None:
            raise ValueError('Interaction distance is set to None.')
        return 0.5 * (self.pitch - self.int_dist)

    @property
    def dx_bend(self) -> float or None:
        if self.radius is None:
            raise ValueError('Curvature radius is set to None.')
        return self.sbend_length(self.dy_bend, self.radius)

    @property
    def dx_acc(self) -> float or None:
        if self.dx_bend is None or self.int_length is None:
            return None
        return 2 * self.dx_bend + self.int_length

    @property
    def dx_mzi(self) -> float or None:
        if self.dx_bend is None or self.int_length is None or self.arm_length is None:
            return None
        return 4 * self.dx_bend + 2 * self.int_length + self.arm_length

    @staticmethod
    def get_sbend_parameter(dy: float, radius: float) -> tuple:
        """
        Computes the final rotation_angle, and x-displacement for a circular S-bend given the y-displacement dy and
        curvature
        radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: (final rotation_angle [radians], x-displacement [mm])
        :rtype: tuple
        """
        if radius <= 0:
            raise ValueError(f'Radius should be a positive value. Given {radius:.3f}.')
        a = np.arccos(1 - (np.abs(dy / 2) / radius))
        dx = 2 * radius * np.sin(a)
        return a, dx

    def sbend_length(self, dy: float, radius: float) -> float:
        """
        Computes the x-displacement for a circular S-bend given the y-displacement dy and curvature radius.

        :param dy: Displacement along y-direction [mm].
        :type dy: float
        :param radius: Curvature radius of the S-bend [mm].
        :type radius: float
        :return: x-displacement [mm]
        :rtype: float
        """
        return self.get_sbend_parameter(dy, radius)[1]

    def get_spline_parameter(self, disp_x: float = None, disp_y: float = None, disp_z: float = None,
                             radius: float = 20) -> tuple:
        """
        Computes the displacements along x-, y- and z-direction and the total lenght of the curve.
        The dy and dz displacements are given by the user. The dx displacement can be known (and thus given as input)
        or unknown and it is computed using the get_sbend_parameter() method for the given radius.

        If disp_x, disp_y, disp_z are given they are returned unchanged and unsed to compute l_curve.
        On the other hand, if disp_x is None, it is computed using the get_sbend_parameters() method using the
        displacement 'disp_yz' along both y- and z-direction and the given radius.
        In this latter case, the l_curve is computed using the formula for the circular arc (radius * angle) which is
        multiply by a factor of 2 in order to retrieve the S-bend shape.

        :param disp_x: Displacement along x-direction [mm]. The default is None.
        :type disp_x: float
        :param disp_y: Displacement along y-direction [mm]. The default is None.
        :type disp_y: float
        :param disp_z: Displacement along z-direction [mm]. The default is None.
        :type disp_z: float
        :param radius: Curvature radius of the spline [mm]. The default is 20 mm.
        :type radius: float
        :return: (deltax [mm], deltay [mm], deltaz [mm], curve length [mm]).
        :rtype: Tuple[float, float, float, float]
        """
        if disp_y is None:
            raise ValueError('y-displacement is None. Give a valid disp_y')
        if disp_z is None:
            raise ValueError('z-displacement is None. Give a valid disp_z')

        if disp_x is None:
            disp_yz = np.sqrt(disp_y ** 2 + disp_z ** 2)
            ang, disp_x = self.get_sbend_parameter(disp_yz, radius)
            l_curve = 2 * ang * radius
        else:
            disp = np.array([disp_x, disp_y, disp_z])
            l_curve = np.sqrt(np.sum(disp ** 2))
        return disp_x, disp_y, disp_z, l_curve


@dataclass(kw_only=True)
class MarkerParameters(LaserPathParameters):
    """
    Class containing the parameters for the surface ablation marker fabrication.
    """
    depth: float = 0.0
    lx: float = 1.0
    ly: float = 0.060

    def __post_init__(self):
        super().__post_init__()

        # parent method redefinition, to include depth property
        if not isinstance(self.scan, int):
            raise ValueError('Number of scan must be integer.')

        if self.z_init is None:
            self.z_init = self.depth


@dataclass(kw_only=True)
class RasterImageParameters(LaserPathParameters):
    """
    Class containing the parameters for generic FLM written structure fabrication.
    """
    px_to_mm: float = 0.01  # pixel to millimeter scale when converting image to laser path
    img_size: Tuple[int, int] = (None, None)

    def __post_init__(self):
        super().__post_init__()

    @property
    def path_size(self):
        if not all(self.img_size):  # check if img_size is None
            raise ValueError("No image size given, unable to compute laserpath dimension")
        else:
            return tuple(self.px_to_mm * elem for elem in self.img_size)


@dataclass(kw_only=True)
class TrenchParameters:
    """
    Class containing the parameters for trench fabrication.
    """

    x_center: float = None
    y_min: float = None
    y_max: float = None
    bridge: float = 0.026
    length: float = 1
    nboxz: int = 4
    z_off: float = 0.020
    h_box: float = 0.075
    base_folder: str = ''
    deltaz: float = 0.0015
    delta_floor: float = 0.001
    beam_waist: float = 0.004
    round_corner: float = 0.005
    u: list = None
    speed: float = 4
    speed_closed: float = 5
    speedpos: float = 0.1
    CWD: str = None

    def __post_init__(self):
        # FARCALL directories
        self.CWD = os.path.dirname(os.path.abspath(__file__))

    @property
    def adj_bridge(self) -> float:
        # adjust bridge size considering the size of the laser focus [mm]
        return self.bridge / 2 + self.beam_waist + self.round_corner

    @property
    def n_repeat(self) -> int:
        return int(ceil((self.h_box + self.z_off) / self.deltaz))

    @property
    def rect(self) -> shapely.geometry.box:
        """
        Getter for the rectangular box for the whole trench column. If the ``x_c``, ``y_min`` and ``y_max`` are set we
        create a rectangular polygon that will be used to create the single trench blocks.

        ::
            +-------+  -> y_max
            |       |
            |       |
            |       |
            +-------+  -> y_min
                x_c

        :return: Rectangular box centered in ``x_c`` and y-borders at ``y_min`` and ``y_max``.
        :rtype: shapely.geometry.box
        """
        if self.x_center is None or self.y_min is None or self.y_max is None:
            return None
        else:
            return box(self.x_center - self.length / 2, self.y_min,
                       self.x_center + self.length / 2, self.y_max)


@dataclass(kw_only=True)
class GcodeParameters:
    """
    Class containing the parameters for the G-Code file compiler.
    """

    filename: str = None
    export_dir: str = ''
    samplesize: Tuple[float, float] = (None, None)
    laser: str = 'CAPABLE'
    home: bool = False
    new_origin: Tuple[float] = (0.0, 0.0)
    warp_flag: bool = False
    n_glass: float = 1.50
    n_environment: float = 1.33
    rotation_angle: float = 0.0
    aerotech_angle: float = None
    long_pause: float = 0.5
    short_pause: float = 0.05
    output_digits: int = 6
    speed_pos: float = 5.0
    flip_x: bool = False
    flip_y: bool = False

    def __post_init__(self):
        if self.filename is None:
            raise ValueError('Filename is None, set GcodeParameters.filename.')
        self.CWD = os.path.dirname(os.path.abspath(__file__))

        self.fwarp = self.antiwarp_management(self.warp_flag)

        if self.rotation_angle != 0:
            print(' BEWARE, ANGLE MUST BE IN DEGREE! '.center(39, "*"))
            print(f' Rotation angle is {self.rotation_angle % 360:.3f} deg. '.center(39, "*"))
        self.rotation_angle = radians(self.rotation_angle % 360)

        if self.aerotech_angle:
            print(' BEWARE, G84 COMMAND WILL BE USED!!! '.center(39, "*"))
            print(' ANGLE MUST BE IN DEGREE! '.center(39, "*"))
            print(f' Rotation angle is {self.aerotech_angle % 360:.3f} deg. '.center(39, "*"))
            print()
            self.aerotech_angle = self.aerotech_angle % 360

    @property
    def xsample(self) -> float:
        return self.samplesize[0]

    @property
    def ysample(self) -> float:
        return self.samplesize[1]

    @property
    def neff(self) -> float:
        return self.n_glass / self.n_environment

    @property
    def tshutter(self) -> float:
        """
        Function that set the shuttering time given the fabrication laboratory.

        :return: shuttering time
        :rtype: float
        """
        if self.laser.lower() not in ['pharos', 'carbide', 'uwe']:
            raise ValueError('Laser can be only PHAROS, CARBIDE or UWE. Given {self.laser.upper()}.')
        if self.laser.lower() == 'pharos':
            return 0.000
        else:
            return 0.005

    def antiwarp_management(self, opt: bool, num: int = 16) -> interp2d:
        """
        It fetches an antiwarp function in the current working direcoty. If it doesn't exist, it lets you create a new
        one. The number of sampling points can be specified.

        :param opt: if True apply antiwarp.
        :type opt: bool
        :param num: number of sampling points
        :type num: int
        :return: warp function, `f(x, y)`
        :rtype: scipy.interpolate.interp2d
        """

        if opt:
            if any(x is None for x in self.samplesize):
                raise ValueError(f'Wrong sample size dimensions. Given ({self.samplesize[0]}, {self.samplesize[1]}).')
            function_pickle = os.path.join(self.CWD, "fwarp.pkl")
            if os.path.exists(function_pickle):
                fwarp = pickle.load(open(function_pickle, "rb"))
            else:
                fwarp = self.antiwarp_generation(self.samplesize, num)
                pickle.dump(fwarp, open(function_pickle, "wb"))
        else:
            def fwarp(x, y):
                return 0
        return fwarp

    @staticmethod
    def antiwarp_generation(samplesize: Tuple[float, float], num_tot: int, *, margin: float = 2) -> interp2d:
        """
        Helper for the generation of antiwarp function.
        The minimum number of data points required is (k+1)**2, with k=1 for linear, k=3 for cubic and k=5 for quintic
        interpolation.

        :param samplesize: glass substrate dimensions, (x-dim, y-dim)
        :type samplesize: Tuple(float, float)
        :param num_tot: number of sampling points
        :type num_tot: int
        :param margin: margin [mm] from the borders of the glass samples
        :type margin: float
        :return: warp function, `f(x, y)`
        :rtype: scipy.interpolate.interp2d
        """

        if num_tot < 4 ** 2:
            raise ValueError('I need more values to compute the interpolation.')

        num_side = int(np.ceil(np.sqrt(num_tot)))
        xpos = np.linspace(margin, samplesize[0] - margin, num_side)
        ypos = np.linspace(margin, samplesize[1] - margin, num_side)
        xlist = []
        ylist = []
        zlist = []

        print('Focus height in µm (!!!) at:')
        for pos in list(product(xpos, ypos)):
            xlist.append(pos[0])
            ylist.append(pos[1])
            zlist.append(float(input('X={:.1f} Y={:.1f}: \t'.format(pos[0], pos[1]))) / 1000)
            if zlist[-1] == '':
                raise ValueError('You have missed the last value.')

        # surface interpolation
        func_antiwarp = interp2d(xlist, ylist, zlist, kind='cubic')

        # plot the surface
        xprobe = np.linspace(-3, samplesize[0] + 3)
        yprobe = np.linspace(-3, samplesize[1] + 3)
        zprobe = func_antiwarp(xprobe, yprobe)
        ax = plt.axes(projection='3d')
        ax.contour3D(xprobe, yprobe, zprobe, 200, cmap='viridis')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()

        return func_antiwarp