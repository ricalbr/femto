import os
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from itertools import zip_longest
from operator import add
from pathlib import Path

import numpy as np

from femto.helpers import dotdict, listcast

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import List

from femto import Trench, TrenchColumn
from femto.Parameters import GcodeParameters


class PGMCompiler(GcodeParameters):
    """
    Class representing a PGM Compiler.
    """

    def __init__(self, param: dict):
        super().__init__(**param)

        self._instructions = deque()
        self._total_dwell_time = 0.0
        self._shutter_on = False
        self._mode_abs = True
        self._loaded_files = []

    @property
    def tdwell(self) -> float:
        return self._total_dwell_time

    def __enter__(self):
        """
        Context manager entry

        :return: Self

        Can be use like:
        ::
            with femto.PGMCompiler(filename, ind_rif) as gc:
                <code block>
        """
        self.header()
        self.dwell(1.0)
        if self.aerotech_angle:
            self._enter_axis_rotation(angle=self.aerotech_angle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit

        :return: None
        """

        if self.aerotech_angle:
            self._exit_axis_rotation()
            self._instructions.append('\n')
        if self.home:
            self.go_init()
        self.close()

    # Methods
    def header(self):
        """
        The function print the header file of the G-Code file. The user can specify the fabrication line to work in
        ``CAPABLE``, ``CARBIDE`` or ``FIRE LINE1`` as parameter when the G-Code Compiler obj is instantiated.

        :return: None
        """
        if self.lab.upper() not in ['CAPABLE', 'FIRE', 'CARBIDE']:
            raise ValueError(f'Fabrication line should be CAPABLE, CARBIDE or FIRE. Given {self.lab.upper()}.')

        if self.lab.upper() == 'CAPABLE':
            with open(os.path.join(self.CWD, 'utils', 'header_capable.txt')) as fd:
                self._instructions.extend(fd.readlines())
        elif self.lab.upper() == 'CARBIDE':
            with open(os.path.join(self.CWD, 'utils', 'header_carbide.txt')) as fd:
                self._instructions.extend(fd.readlines())
        else:
            with open(os.path.join(self.CWD, 'utils', 'header_fire.txt')) as fd:
                self._instructions.extend(fd.readlines())

    def dvar(self, variables: List[str]):
        """
        Adds the declaration of variables in a G-Code file.

        :param variables: List of G-Code variables
        :type variables: List[str]
        :return: None
        """
        args = ' '.join(["${}"] * len(variables)).format(*variables)
        self._instructions.appendleft(f'DVAR {args}\n')

    def mode(self, mode: str = 'ABS'):
        if mode.upper() not in ['ABS', 'INC']:
            raise ValueError(f'Mode should be either ABSOLUTE (ABS) or INCREMENTAL (INC). {mode} was given.')
        if mode.upper() == 'ABS':
            self._instructions.append('ABSOLUTE\n')
            self._mode_abs = True
        else:
            self._instructions.append('INCREMENTAL\n')
            self._mode_abs = False

    def comment(self, comstring: str):
        """
        Adds a comment to a G-Code file.

        :param comstring: Content of the comment (without line-break character).
        :type comstring: str
        :return: None
        """
        if comstring:
            self._instructions.append(f'\n; {comstring}\n')
        else:
            self._instructions.append('\n')

    def shutter(self, state: str):
        """
        Adds the instruction to open (close) the shutter to a G-Code file only when necessary.
        The user specifies the state and the function compare it to the current state of the shutter (which is
        tracked internally during the compilation of the .pgm file). The instruction is printed to file only if the
        new state differs from the current one.

        :param state: New state of the shutter. ``ON`` or ``OFF``
        :type state: str
        :return: None

        :raise ValueError: Shutter state not valid.
        """
        if state.upper() not in ['ON', 'OFF']:
            raise ValueError(f'Shutter state should be ON or OFF. Given {state.upper()}.')

        if state.upper() == 'ON' and self._shutter_on is False:
            self._shutter_on = True
            self._instructions.append('\nPSOCONTROL X ON\n')
        elif state.upper() == 'OFF' and self._shutter_on is True:
            self._shutter_on = False
            self._instructions.append('\nPSOCONTROL X OFF\n')
        else:
            pass

    def dwell(self, pause: float):
        """
        Adds pause instruction to a G-Code file.

        :param pause: Value of the pause time [s].
        :type pause: float
        :return: None
        """
        self._instructions.append(f'DWELL {pause}\n\n')
        self._total_dwell_time += float(pause)

    def set_home(self, home_pos: List[float]):
        """
        Defines a preset position or a software home position to the one specified in the input list.
        To exclude a variable set it to None.

        :param home_pos: Ordered coordinate list that specifies software home position [mm].
        ::
            ``home_pos[0]`` -> X
            ``home_pos[1]`` -> Y
            ``home_pos[2]`` -> Z
        :type home_pos: List[float]
        :return: None

        :raise ValueError: Final position is not valid.
        """
        if np.size(home_pos) != 3:
            raise ValueError(f'Given final position is not valid. 3 values required, given {np.size(home_pos)}.')

        args = self._format_args(*home_pos)
        self._instructions.append(f'G92 {args}\n')

    def homing(self):
        """
        Utility function to return to the origin (0,0,0) with shutter OFF.

        :return: None
        """
        self.comment('HOMING')
        self.move_to([0, 0, 0])

    def go_init(self):
        """
        Utility function to return to the initial point of fabrication (-2,0,0) with shutter OFF.

        :return: None
        """
        self.move_to([-2, 0, 0])

    def move_to(self, position: List[float], speedpos: float = 5):
        """
        Utility function to move to a given position with the shutter OFF. The user can specify the target position
        and the positioning speed.

        :param position: Ordered coordinate list that specifies the target position [mm].
            position[0] -> X
            position[1] -> Y
            position[2] -> Z
        :type position: List[float]
        :param speedpos: Positioning speed [mm/s]. The default is 5 mm/s.
        :type speedpos: float
        :return: None
        """
        if np.size(position) != 3:
            raise ValueError(f'Given final position is not valid. 3 values required, given {np.size(position)}.')

        if self._shutter_on is True:
            self.shutter('OFF')

        args = self._format_args(*position, speedpos)
        self._instructions.append(f'LINEAR {args}\n')
        self.dwell(self.long_pause)

    @contextmanager
    def axis_rotation(self):
        self._enter_axis_rotation()
        try:
            yield
        finally:
            self._exit_axis_rotation()

    @contextmanager
    def for_loop(self, var: str, num: int):
        """
        Context manager that manages a FOR loop in a G-Code file.

        :param var: Name of the variable used for iteration.
        :type var: str
        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError('Number of iterations is None. Set the num_scan attribute in Waveguide obj.')
        if num == 0:
            raise ValueError('Number of iterations is 0. Set num_scan >= 1.')
        self._instructions.append(f'FOR ${var} = 0 TO {num - 1}\n')
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            self._instructions.append(f'NEXT ${var}\n\n')
            _dt_forloop = self._total_dwell_time - _temp_dt
            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += num * _dt_forloop

    @contextmanager
    def repeat(self, num: int):
        """
        Context manager that manages a REPEAT loop in a G-Code file.

        :param num: Number of iterations.
        :type num: int
        :return: None
        """
        if num is None:
            raise ValueError('Number of iterations is None. Set the `scan` attribute in Waveguide obj.')
        if num == 0:
            raise ValueError('Number of iterations is 0. Set "scan" >= 1.')
        if num != 1:
            self._instructions.append(f'REPEAT {num}\n')
        _temp_dt = self._total_dwell_time
        try:
            yield
        finally:
            if num != 1:
                self._instructions.append('ENDREPEAT\n\n')
            _dt_repeat = self._total_dwell_time - _temp_dt
            # pauses should be multiplied by number of cycles as well
            self._total_dwell_time += num * _dt_repeat

    def tic(self):
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used before the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "INIZIO #TS"\n\n')

    def toc(self):
        """
        Print the current time (hh:mm:ss) in message panel. The function is intended to be used after the execution
        of an operation or script to measure its time performances.

        :return: None
        """
        self._instructions.append('MSGDISPLAY 1, "FINE   #TS"\n')
        self._instructions.append('MSGDISPLAY 1, "---------------------"\n')
        self._instructions.append('MSGDISPLAY 1, " "\n\n')

    def load_program(self, filename: str, task_id: int = 0):
        """
        Adds the instruction to LOAD a program in a G-Code file.

        :param filename: Name of the file that have to be loaded.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        file = self._parse_filepath(filename, extension='pgm')
        self._instructions.append(f'PROGRAM {task_id} LOAD "{file}"\n')
        self._loaded_files.append(file.stem)

    def remove_program(self, filename: str, task_id: int = 0):
        """
        Adds the instruction to REMOVE a program from memory buffer in a G-Code file.

        :param filename: Name of the file to remove.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        file = self._parse_filepath(filename, extension='pgm')
        self.programstop(task_id)
        self._instructions.append(f'REMOVEPROGRAM "{file}"\n')
        self._loaded_files.remove(file.stem)

    def farcall(self, filename: str):
        """
        Adds the FARCALL instruction in a G-Code file.

        :param filename: Name of the file to call.
        :type filename: str
        :return: None
        """
        file = self._parse_filepath(filename)
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(f'{file} not loaded. Cannot load it.')
        self.dwell(self.short_pause)
        self._instructions.append(f'FARCALL "{file}"\n')

    def programstop(self, task_id: int = 0):
        self._instructions.append(f'PROGRAM {task_id} STOP\n')
        self._instructions.append(f'WAIT (TASKSTATUS({task_id}, DATAITEM_TaskState) == TASKSTATE_Idle) -1\n')

    def buffercall(self, filename: str, task_id: int = 0):
        """
        Adds the BUFFEREDRUN instruction in a G-Code file.

        :param filename: Name of the file to call.
        :type filename: str
        :param task_id: ID of the task associated to the process.
        :type task_id: int
        :return: None
        """
        file = self._parse_filepath(filename)
        if file.stem not in self._loaded_files:
            raise FileNotFoundError(f'{file} not loaded. Cannot load it.')
        self._instructions.append(f'PROGRAM {task_id} BUFFEREDRUN "{file}"\n')

    def chiamatutto(self, filenames: List[str], task_id: List[int] = 0):
        for (fpath, t_id) in zip_longest(listcast(filenames), listcast(task_id), fillvalue=0):
            _, fn = os.path.split(fpath)
            self.load_program(fpath, t_id)
            self.farcall(fn)
            self.remove_program(fn, t_id)
            self.dwell(self.long_pause)
            self.instruction('\n')

    def write(self, points: np.ndarray):
        """
        The function convert the quintuple (X,Y,Z,F,S) to G-Code instructions. The (X,Y,Z) coordinates are
        transformed using the transformation matrix that takes into account the rotation of a given rotation_angle
        and the homothety to compensate the (effective) refractive index different from 1. Moreover, if the warp_flag
        is True the points are compensated along the z-direction.

        The transformed points are then parsed together with the feed rate and shutter state coordinate to produce
        the LINEAR movements.

        :param points: Numpy matrix containing the values of the tuple [X,Y,Z,F,S] coordinates.
        :type points: numpy.ndarray
        :return: None
        """

        x_c, y_c, z_c, f_c, s_c = self.transform_points(points)
        args = [self._format_args(x, y, z, f) for (x, y, z, f) in zip(x_c, y_c, z_c, f_c)]
        for (arg, s) in zip_longest(args, s_c):
            if s == 0 and self._shutter_on is True:
                self.shutter('OFF')
                self.dwell(self.long_pause)
            elif s == 1 and self._shutter_on is False:
                self.shutter('ON')
                self.dwell(self.long_pause)
            else:
                self._instructions.append(f'LINEAR {arg}\n')
        self.dwell(self.long_pause)

    def transform_points(self, points):
        x, y, z, fc, sc = points.T
        x, y, z = self.flip_path(x, y, z)
        point_matrix = np.stack((x, y, z), axis=-1).astype(np.float32)
        point_matrix -= np.array([self.new_origin[0], self.new_origin[1], 0]).T
        if self.warp_flag:
            point_matrix = np.matmul(point_matrix, self.t_matrix())
            x_c, y_c, z_c = self.compensate(point_matrix).T
        else:
            x_c, y_c, z_c = np.matmul(point_matrix, self.t_matrix()).T
        return x_c, y_c, z_c, fc, sc

    def flip_path(self, xc, yc, zc):
        """
        Flip the laser path along the x-, y- and z-coordinates
        :return: None
        """

        flip_toggle = np.array([1, 1, 1])
        # reverse the coordinates arrays to flip
        if self.flip_x:
            flip_toggle[0] = -1
            xc = np.flip(xc)
        if self.flip_y:
            flip_toggle[1] = -1
            yc = np.flip(yc)

        # create flip matrix (+1 -> no flip, -1 -> flip)
        flip_matrix = np.diag(flip_toggle)

        # create the displacement matrix to map the transformed min/max coordinates to the original min/max coordinates)
        points_matrix = np.array([xc, yc, zc])
        displacements = np.array([self.samplesize[0] - self.new_origin[0],
                                  self.samplesize[1] - self.new_origin[1],
                                  0])
        S = np.multiply((1 - flip_toggle) / 2, displacements)

        # matrix multiplication and sum element-wise, add the displacement only to the flipped coordinates
        flip_x, flip_y, flip_z = map(add, flip_matrix @ points_matrix, S)

        # update coordinates
        if self.flip_x:
            xc = np.flip(flip_x)
        else:
            xc = flip_x
        if self.flip_y:
            yc = np.flip(flip_y)
        else:
            yc = flip_y
        return xc, yc, zc

    def instruction(self, instr: str):
        """
        Adds a G-Code instruction passed as parameter to the PGM file.

        :param instr: Instruction line to be added to the PGM file. The ``\\n`` character is optional.
        :type instr: str
        :return: None
        """
        if instr.endswith('\n'):
            self._instructions.append(instr)
        else:
            self._instructions.append(instr + '\n')

    def close(self, filename: str = None, verbose: bool = False):
        """
        Dumps all the instruction in self._instruction in a .pgm file.
        The filename is specified during the class instatiation. If no extension is present, the proper one is
        automatically added.

        :param filename: Different filename. The default is None, using self.filename.
        :type filename: str
        :param verbose: Print when G-Code export is finished. The default is False.
        :type verbose: bool
        :return: None

        :raise ValueError: Missed filename.
        """

        # filename overrides self.filename. If not present, self.filename must not be None.
        if filename is None and self.filename is None:
            raise ValueError('No filename given.')

        if filename:
            pgm_filename = filename
        else:
            pgm_filename = self.filename

        # if not present in the filename, add the proper file extension
        if not pgm_filename.endswith('.pgm'):
            pgm_filename += '.pgm'

        if self.export_dir:
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)
            pgm_filename = os.path.join(self.export_dir, pgm_filename)

        # write instruction to file
        with open(pgm_filename, 'w') as f:
            f.write(''.join(self._instructions))
        self._instructions.clear()
        if verbose:
            print('G-code compilation completed.')

    def compensate(self, pts: np.ndarray) -> np.ndarray:
        """
        Returns the points compensated along z-direction for the refractive index, the offset and the glass warp.

        :param pts: ``[X,Y,Z]`` matrix or just a single point
        :type pts: np.ndarray
        :return: ``[X,Y,Z]`` matrix of compensated points
        :rtype: np.ndarray
        """
        pts_comp = deepcopy(np.array(pts))

        if pts_comp.size > 3:
            zwarp = [float(self.fwarp(x, y)) for x, y in zip(pts_comp[:, 0], pts_comp[:, 1])]
            zwarp = np.array(zwarp)
            pts_comp[:, 2] = (pts_comp[:, 2] + zwarp / self.neff)
        else:
            pts_comp[2] = (pts_comp[2] + self.fwarp(pts_comp[0], pts_comp[1]) / self.neff)
        return pts_comp

    def t_matrix(self, dim: int = 3) -> np.ndarray:
        """
        Given the rotation rotation_angle and the rifraction index, the function compute the transformation matrix as
        composition of rotatio matrix (RM) and a homothety matrix (SM).

        :param dim: Dimension of the transformation matrix. The default is 3.
        :type dim: int
        :return: Transformation matrix, TM = SM*RM
        :rtype: np.array

        :raise ValueError: Dimension not valid
        """
        RM = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                       [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                       [0, 0, 1]])
        SM = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1 / self.neff]])
        t_mat = SM @ RM
        if dim == 3:
            return t_mat.T
        elif dim == 2:
            # export xy-submatrix
            ixgrid = np.ix_([0, 1], [0, 1])
            return t_mat[ixgrid].T
        else:
            raise ValueError(f'Dimension not valid. dim must be either 2 or 3. Given {dim}.')

    # Private interface
    def _format_args(self, x: float = None, y: float = None, z: float = None, f: float = None) -> str:
        """
        Utility function that creates a string prepending the coordinate name to the given value for all the given
        the coordinates ``[X,Y,Z]`` and feed rate ``F``.
        The decimal precision can be set by the user by setting the output_digits attribute.

        :param x: Value of the x-coordinate [mm]. The default is None.
        :type x: float
        :param y: Value of the y-coordinate [mm]. The default is None.
        :type y: float
        :param z: Value of the z-coordinate [mm]. The default is None.
        :type z: float
        :param f: Value of the f-coordinate [mm]. The default is None.
        :type f: float
        :return: Formatted string of the type: 'X<value> Y<value> Z<value> F<value>'.
        :rtype: str

        :raise ValueError: Try to move null speed.
        """
        args = []
        if x is not None:
            args.append(f'X{x:.{self.output_digits}f}')
        if y is not None:
            args.append(f'Y{y:.{self.output_digits}f}')
        if z is not None:
            args.append(f'Z{z:.{self.output_digits}f}')
        if f is not None:
            if f < 1e-6:
                raise ValueError('Try to move with F = 0.0 mm/s. Check speed parameter.')
            args.append(f'F{f:.{self.output_digits}f}')
        args = ' '.join(args)
        return args

    @staticmethod
    def _parse_filepath(filename: str, filepath: str = None, extension: str = None) -> Path:
        """
        The fuction takes a filename and (optional) filepath. It merges the two and check if the file exists in the
        system.
        An extension parameter can be given as input. In that case the function also checks weather the filename has
        the correct extension.

        :param filename: Name of the file that have to be loaded.
        :type filename: str
        :param filepath: Path of the folder containing the file. The default is None.
        :type filepath: str
        :param extension: File extension. The default is None.
        :type extension: str
        :return: Complete path of the file (filepath + filename).
        :rtype: pathlib.Path
        """
        if extension is not None and not filename.endswith(extension):
            raise ValueError(f'Given filename has wrong extension. Given {filename}, required .{extension}.')

        if filepath is not None:
            file = Path(filepath) / filename
        else:
            file = Path(filename)
        return file

    def _enter_axis_rotation(self, angle: float = None):
        self.comment('ACTIVATE AXIS ROTATION')
        self._instructions.append(f'LINEAR X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n')
        self._instructions.append(f'G84 X Y\n')

        if angle is None:
            return

        angle = float(angle % 360)
        self._instructions.append(f'G84 X Y F{angle}\n\n')

    def _exit_axis_rotation(self):
        self.comment('DEACTIVATE AXIS ROTATION')
        self._instructions.append(f'LINEAR X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{self.speed_pos:.6f}\n')
        self._instructions.append(f'G84 X Y\n')


class PGMTrench(PGMCompiler):
    def __init__(self, param: dict, trench_columns: List[TrenchColumn]):
        super().__init__(param)
        self.trench_columns = listcast(trench_columns)
        self._param = param

    def write(self, dirname: str = 'trench'):
        """
        Helper function for the compilation of trench columns.
        For each trench in the column, the function first compile a PGM file for border (or wall) and for the floor
        inside a directory given by the user (base_folder).
        Secondly, the function produce a FARCALL.pgm program to fabricate all the trenches in the column.

        :param dirname: Name of the directory in which the .pgm file will be written. The path is relative to the
        current file (.\\dirname\\)
        :type dirname: str
        :return: None
        """

        for col_idx, col in enumerate(self.trench_columns):
            # Prepare directories for export .pgm files
            col_dir = os.path.join(os.getcwd(), self.export_dir, dirname, f'trenchCol{col_idx + 1:03}')
            os.makedirs(col_dir, exist_ok=True)

            # Export walls and floors .pgm scripts
            for i, trench in enumerate(col):
                filename = os.path.join(col_dir, f'trench{i + 1:03}')
                self._export_path(filename, trench, speed=col.speed)

            # Export FARCALL for each trench column
            col_param = dotdict(self._param.copy())
            col_param.filename = os.path.join(os.getcwd(), self.export_dir, dirname, f'FARCALL{col_idx + 1:03}.pgm')
            with PGMCompiler(col_param) as G:
                G.dvar(['ZCURR'])

                for nbox in range(col.nboxz):
                    for t_index, trench in enumerate(col):
                        # load filenames (wall/floor)
                        wall_filename = f'trench{t_index + 1:03}_wall.pgm'
                        floor_filename = f'trench{t_index + 1:03}_floor.pgm'
                        wall_path = os.path.join(col.base_folder, f'trenchCol{col_idx + 1:03}', wall_filename)
                        floor_path = os.path.join(col.base_folder, f'trenchCol{col_idx + 1:03}', floor_filename)

                        x0, y0 = trench.block.exterior.coords[0]
                        z0 = (nbox * col.h_box - col.z_off) / super().neff
                        G.comment(f'+--- COLUMN #{col_idx + 1}, TRENCH #{t_index + 1} LEVEL {nbox + 1} ---+')

                        # WALL
                        G.load_program(wall_path)
                        G.instruction(
                                f'MSGDISPLAY 1, "COL {col_idx + 1:03}, TR {t_index + 1:03}, LV {nbox + 1:03}, W"\n')
                        G.shutter('OFF')
                        if col.u:
                            G.instruction(f'LINEAR U{col.u[0]:.6f}')
                        G.move_to([x0 - self.new_origin[0], y0 - self.new_origin[1], z0], speedpos=col.speed_closed)

                        G.instruction(f'$ZCURR = {z0:.6f}')
                        G.shutter('ON')
                        with G.repeat(col.n_repeat):
                            G.farcall(wall_filename)
                            G.instruction(f'$ZCURR = $ZCURR + {col.deltaz / super().neff:.6f}')
                            G.instruction('LINEAR Z$ZCURR')
                        G.remove_program(wall_filename)

                        # FLOOR
                        G.shutter(state='OFF')
                        G.load_program(floor_path)
                        G.instruction(
                                f'MSGDISPLAY 1, "COL {col_idx + 1:03}, TR {t_index + 1:03}, LV {nbox + 1:03}, F"\n')
                        if col.u:
                            G.instruction(f'LINEAR U{col.u[-1]:.6f}')
                        G.shutter(state='ON')
                        G.farcall(floor_filename)
                        G.shutter('OFF')
                        if col.u:
                            G.instruction(f'LINEAR U{col.u[0]:.6f}')
                        G.remove_program(floor_filename)
                G.instruction('MSGCLEAR -1\n')
            del G

        # MAIN FARCALL, calls all columns .pgm scripts
        if self.trench_columns:
            # MAIN FARCALL parameters
            main_param = dotdict(self._param.copy())
            main_param.filename = os.path.join(dirname, 'MAIN.pgm')
            main_param.aerotech_angle = None

            t_list = [os.path.join(col.base_folder, f'FARCALL{idx + 1:03}.pgm') for idx, col in
                      enumerate(self.trench_columns)]
            with PGMCompiler(main_param) as G:
                G.chiamatutto(t_list)

    def _export_path(self, filename: str, trench: Trench, speed: float = 4):
        """
        Helper function for the export of the wall and floor instruction of a Trench obj.

        :param filename: Base filename for the wall.pgm and floor.pgm files. If the filename ends with the '.pgm'
        extension, the latter it is stripped and replaced with '_wall.pgm' and '_floor.pgm' to differentiate the two
        paths.
        :type filename: str
        :param trench: Trench obj to export.
        :type trench: Trench
        :param speed: Traslation speed during fabrication [mm/s]. The default is 4 [mm/s].
        :type speed: float
        :return: None
        """
        if filename is None:
            raise ValueError('No filename given.')
        if filename.endswith('.pgm'):
            filename = filename.split('.')[0]

        # Wall script
        points = np.array(trench.block.exterior.coords.xy).T
        self._write_array(filename + '_wall.pgm', points, speed)
        del points

        # Floor script
        points = []
        [points.extend(np.stack(path, axis=-1)) for path in trench.trench_paths()]
        self._write_array(filename + '_floor.pgm', np.array(points), speed)
        del points

    def _write_array(self, pgm_filename: str, points: np.ndarray, f_val: float):
        """
        Helper function that produces a PGM file for a 3D matrix of points at a given traslation speed,
        without shuttering operations.
        The function parse the points input matrix, applies the rotation and homothety transformations and parse all
        the LINEAR instructions.

        :param pgm_filename: Filename of the file in which the G-Code instructions will be written.
        :type pgm_filename: str
        :param points: 3D points matrix. If the points matrix is 2D it is intended as [x,y] coordinates.
        :type points: np.ndarray
        :param f_val: Traslation speed value.
        :type f_val: float, np.ndarray or list
        :return: None
        """

        points -= np.array([self.new_origin[0], self.new_origin[1]]).T

        if points.shape[-1] == 2:
            x_arr, y_arr = np.matmul(points, self.t_matrix(dim=2)).T
            z_arr = [None]
        else:
            x_arr, y_arr, z_arr = np.matmul(points, self.t_matrix()).T

        instr = [self._format_args(x, y, z, f) for (x, y, z, f) in zip_longest(x_arr, y_arr, z_arr, listcast(f_val))]
        gcode_instr = [f'LINEAR {line}\n' for line in instr]
        with open(pgm_filename, 'w') as f:
            f.write(''.join(gcode_instr))


def _example():
    from femto import Waveguide, Cell

    # Data
    PARAMETERS_WG = dotdict(
            scan=6,
            speed=20,
            radius=15,
            pitch=0.080,
            int_dist=0.007,
            lsafe=3,
            samplesize=(25, 3),
    )
    increment = [PARAMETERS_WG.lsafe, 0, 0]

    PARAMETERS_GC = dotdict(
            filename='testPGMcompiler.pgm',
            lab='CAPABLE',
            samplesize=PARAMETERS_WG.samplesize,
            rotation_angle=2.0,
            flip_x=True,
    )

    # Calculations
    coup = [Waveguide(PARAMETERS_WG) for _ in range(2)]
    for i, wg in enumerate(coup):
        wg.start([-2, -wg.pitch / 2 + i * wg.pitch, 0.035]) \
            .linear(increment) \
            .sin_mzi((-1) ** i * wg.dy_bend, arm_length=1.0) \
            .sin_mzi((-1) ** i * wg.dy_bend, arm_length=1.0) \
            .linear(increment) \
            .sin_mzi((-1) ** i * wg.dy_bend, arm_length=1.0) \
            .linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
        wg.end()

    c = Cell(PARAMETERS_GC)
    c.append(coup)
    c.plot2d()

    # Compilation
    with PGMCompiler(PARAMETERS_GC) as G:
        G.set_home([0, 0, 0])
        with G.repeat(PARAMETERS_WG['scan']):
            for i, wg in enumerate(coup):
                G.comment(f'Modo: {i}')
                G.write(wg.points)
        G.move_to([None, 0, 0.1])
        G.set_home([0, 0, 0])


if __name__ == '__main__':
    _example()
