from __future__ import annotations

import itertools
from typing import Any
from typing import Callable

import numpy as np
import numpy.typing as npt
from femto.curves import sin
from femto.waveguide import Waveguide

# Define array type
nparray = npt.NDArray[np.float64]

sign = itertools.cycle([1, -1])


def _get_upp_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    else:
        return size


def _get_fan_in_out(y0: float, param: dict[str, Any], num_io: int) -> zip[tuple[float, float, float, float, float]]:
    tmp_wg = Waveguide(**param)

    y_fa = y0 + np.arange(0, num_io) * tmp_wg.pitch_fa
    y_wg = y0 + np.arange(0, num_io) * tmp_wg.pitch
    y_wg += np.mean(y_fa) - np.mean(y_wg)

    dy_in = y_wg - y_fa
    dy_out = -dy_in

    dx_fa = [tmp_wg.get_sbend_parameter(elem, 2 * tmp_wg.radius)[1] for elem in np.abs(dy_in)]
    lin = np.max(dx_fa) - np.abs(dx_fa)
    lout = lin
    return zip(y_fa, dy_in, lin, lout, dy_out)


def clements(
    size: int | tuple[int, int],
    param: dict[str, Any],
    f_profile: Callable[..., tuple[nparray, nparray, nparray]] = sin,
    x_init: float = 5,
    y_init: float | None = None,
    disp_marker: float = 0.2,
) -> tuple[list[Waveguide], list[tuple[float, float]]]:
    M, N = _get_upp_size(size)

    y0 = y_init if y_init is not None else param['y_init']
    param_io = _get_fan_in_out(y0=y0, param=param, num_io=N)

    circuit_wgs = []
    mk_coords = []
    for i in range(N):
        yi, fa_in, fa_lin, fa_lout, fa_out = next(param_io)

        wg = Waveguide(**param)
        wg.start([wg.x_init, yi, wg.z_init])

        wg.linear([x_init, None, None], mode='ABS')
        if wg.pitch != wg.pitch_fa:
            wg.bend(dy=fa_in, dz=0, radius=2 * wg.radius, fx=f_profile)
            wg.linear([fa_lin, 0, 0])
        wg.linear([wg.arm_length, 0, 0])
        if i == 0:
            mk_coords.append((wg.lastx - wg.arm_length / 2, wg.lasty - disp_marker))
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        for _ in range(M - 1):
            wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
            if i == 0:
                mk_coords.append((wg.lastx, wg.lasty - disp_marker))
            wg.coupler(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
            wg.linear([wg.arm_length, 0, 0])
            if i == 0:
                mk_coords.append((wg.lastx - wg.arm_length / 2, wg.lasty - disp_marker))
            wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        if i == 0:
            mk_coords.append((wg.lastx, wg.lasty - disp_marker))
        wg.coupler(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        next(sign)
        if wg.pitch != wg.pitch_fa:
            wg.linear([fa_lout, 0, 0])
            wg.bend(dy=fa_out, dz=0, radius=2 * wg.radius, fx=f_profile)
        wg.linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
        wg.end()

        circuit_wgs.append(wg)

    return circuit_wgs, mk_coords


def bell(
    size: int | tuple[int, int],
    param: dict[str, Any],
    f_profile: Callable[..., tuple[nparray, nparray, nparray]] = sin,
    x_init: float = 5,
    y_init: float | None = None,
    disp_marker: float = 0.2,
) -> tuple[list[Waveguide], list[tuple[float, float]]]:
    M, N = _get_upp_size(size)

    y0 = y_init if y_init is not None else param['y_init']
    param_io = _get_fan_in_out(y0=y0, param=param, num_io=N)

    circuit_wgs = []
    mk_coords = []
    for i in range(N):
        yi, fa_in, fa_lin, fa_lout, fa_out = next(param_io)

        wg = Waveguide(**param)
        wg.start([wg.x_init, yi, wg.z_init])

        wg.linear([x_init, None, None], mode='ABS')
        if wg.pitch != wg.pitch_fa:
            wg.bend(dy=fa_in, dz=0, radius=2 * wg.radius, fx=f_profile)
            wg.linear([fa_lin, 0, 0])
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        for _ in range(M):
            wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
            if i == 0:
                mk_coords.append((wg.lastx, wg.lasty - disp_marker))
            wg.double_bend(dy1=next(sign) * wg.dy_bend, dy2=next(sign) * 2 * wg.dy_bend, dz1=0, dz2=0, fx=f_profile)
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        if i == 0:
            mk_coords.append((wg.lastx, wg.lasty - disp_marker))
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        wg.bend(dy=next(sign) * wg.dy_bend, dz=0, fx=f_profile)
        if wg.pitch != wg.pitch_fa:
            wg.linear([fa_lout, 0, 0])
            wg.bend(dy=fa_out, dz=0, radius=2 * wg.radius, fx=f_profile)
        wg.linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
        wg.end()
        next(sign)

        circuit_wgs.append(wg)

    return circuit_wgs, mk_coords
