from __future__ import annotations

from typing import Any
from typing import Tuple

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate
from scipy import special
from scipy.interpolate import BPoly

# Define array types
nparray = npt.NDArray[np.float64]
ptarray = Tuple[nparray, nparray, nparray]


def euler(radius: float, theta: float, dz: float, num_points: int, **kwargs: Any | None) -> ptarray:
    L = 2 * abs(radius) * theta  # total length of the Euler bend
    f = np.sqrt(np.pi * abs(radius) * L)  # Fresnel integral function are defined as function of (pi*t^2/2)

    t = np.linspace(0, L, num_points // 2)
    y, x = special.fresnel(t / f)
    z = np.linspace(0, dz, x.size)
    return f * x, f * y, z


def sin(dx: float, dy: float, dz: float, num_points: int, flat_peaks: float = 0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(0, dx, num_points)
    tmp_cos = np.cos(np.pi / dx * x)
    y = 0.5 * dy * (1 - np.sqrt((1 + flat_peaks**2) / (1 + flat_peaks**2 * tmp_cos**2)) * tmp_cos)
    z = 0.5 * dz * (1 - np.cos(np.pi / dx * x))
    return x, y, z


def double_sin(dy1: float, dy2: float, radius: float, num_points: int = 100, **kwargs: Any | None) -> ptarray:
    # First part of the curve, sinusoidal S-bend with dy1 y-displacement
    dx1 = np.sqrt(4 * np.abs(dy1) * radius - dy1**2)
    x1, y1, _ = sin(dx=dx1, dy=dy1, dz=0, num_points=num_points, flat_peaks=0)

    # Second part of the curve, sinusoidal S-bend with dy2 y-displacement and dx2 x-displacement to match the curvature
    # of the first section.
    # Closed form: dx2 = np.sqrt(4*np.abs(dy2)*radius - np.abs(dy1*dy2))
    dx2 = dx1 * np.sqrt(np.abs(dy2 / dy1))
    x2, y2, _ = sin(dx=dx2, dy=dy2, dz=0, num_points=num_points, flat_peaks=0)

    x = np.concatenate([x1, x2 + x1[-1]])
    y = np.concatenate([y1, y2 + y1[-1]])
    z = np.zeros_like(x)

    return x, y, z


def jack_curve(dx: float, dy: float, dz: float, num_points: int, **kwargs: Any | None) -> ptarray:
    dx1 = dx
    dx2 = kwargs['dx2'] or dx1

    k = np.pi / (2 * dx1 + dx2)
    C = np.sqrt(2 / 3)
    A = dy / (np.cos(np.arcsin(C)) - np.cos(3 * np.arcsin(C)))
    x = np.linspace(0, dx, num_points)
    y = A * (np.cos(k * x) - np.cos(3 * k * x))
    z = np.linspace(0, dz, x.size)
    return x, y, z


def spline(
    dx: float,
    dy: float,
    dz: float,
    num_points: int,
    y_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
    z_derivatives: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
    **kwargs: Any | None,
) -> ptarray:
    x = np.linspace(0, dx, num_points)
    y = BPoly.from_derivatives([0, dx], [[0, *y_derivatives[0]], [dy, *y_derivatives[-1]]])(x)
    z = BPoly.from_derivatives([0, dx], [[0, *z_derivatives[0]], [dz, *z_derivatives[-1]]])(x)
    return x, y, z


def spline_bridge(dx: float, dy: float, dz: float, num_points: int, **kwargs: Any | None) -> ptarray:
    xi, yi, zi = spline(
        dx=dx / 2,
        dy=dy / 2,
        dz=dz,
        num_points=num_points // 2,
        y_derivatives=((0.0, 0.0), (dy / dx, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )
    xf, yf, zf = spline(
        dx=dx / 2,
        dy=dy / 2,
        dz=-dz,
        num_points=num_points // 2,
        y_derivatives=((dy / dx, 0.0), (0.0, 0.0)),
        z_derivatives=((0.0, 0.0), (0.0, 0.0)),
    )
    return np.append(xi, xf + xi[-1]), np.append(yi, yf + yi[-1]), np.append(zi, zf + zi[-1])


def tanh(dx: float, dy: float, dz: float, num_points: int, s: float = 1.0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(-dx / 2, dx / 2, num_points)
    y = dy / 2 * np.tanh(x * s)
    z = np.linspace(0, dz, x.size)
    return x + dx / 2, y + dy / 2, z


def erf(dx: float, dy: float, dz: float, num_points: int, s: float = 1.0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(-dx / 2, dx / 2, num_points)
    y = dy / 2 * (1 + special.erf(x * s))
    z = np.linspace(0, dz, x.size)
    return x + dx / 2, y, z


def arctan(dx: float, dy: float, dz: float, num_points: int, s: float = 1.0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(-dx / 2, dx / 2, num_points)
    y = 2 / np.pi * np.arctan(np.pi * (x * s) / 2)
    z = np.linspace(0, dz, x.size)
    return x + dx / 2, y * dy / (y[-1] - y[0]) + dy / 2, z


def rad(dx: float, dy: float, dz: float, num_points: int, s: float = 1.0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(-dx / 2, dx / 2, num_points)
    y = x * s / (np.sqrt(1 + (x * s) ** 2))
    z = np.linspace(0, dz, x.size)
    return x + dx / 2, y * dy / (y[-1] - y[0]) + dy / 2, z


def abv(dx: float, dy: float, dz: float, num_points: int, s: float = 1.0, **kwargs: Any | None) -> ptarray:
    x = np.linspace(-dx / 2, dx / 2, num_points)
    y = x * s * 1 / (1 + abs(x * s))
    z = np.linspace(0, dz, x.size)
    return x + dx / 2, y * dy / (y[-1] - y[0]) + dy / 2, z


def euler_S2(
    dx: float,
    dy: float,
    dz: float,
    radius: float,
    num_points: int,
    theta: float = np.pi / 24,
    n: int = 1,
    **kwargs: Any | None,
) -> ptarray:
    k = 1 / (theta**n * radius ** (n + 1) * (n + 1) ** n)
    s_f = (theta * (n + 1) / k) ** (1 / (n + 1))
    s_vals = np.linspace(0, s_f, num_points // 2)

    x1 = np.array([integrate.quad(lambda var: np.cos(k * var ** (n + 1) / (n + 1)), 0, s)[0] for s in s_vals])
    y1 = np.array([integrate.quad(lambda var: np.sin(k * var ** (n + 1) / (n + 1)), 0, s)[0] for s in s_vals])

    # first, rotate by the final angle
    x2_data, y2_data = np.dot(-np.eye(2), np.stack([x1, y1], 0))
    x2_data, y2_data = x2_data[::-1] + 2 * x1[-1], y2_data[::-1] + 2 * y1[-1]

    x_data = np.concatenate([x1, x2_data], 0)
    y_data = np.concatenate([y1, y2_data], 0)

    x = x_data * dx / x_data[-1]
    y = y_data * dy / y_data[-1]
    z = np.linspace(0, dz, x.size)
    return x, y, z


def euler_S4(
    dx: float,
    dy: float,
    dz: float,
    radius: float,
    num_points: int,
    theta: float = np.pi / 24,
    n: int = 1,
    **kwargs: Any | None,
) -> ptarray:
    k = 1 / (theta**n * radius ** (n + 1) * (n + 1) ** n)
    s_f = (theta * (n + 1) / k) ** (1 / (n + 1))
    s_vals = np.linspace(0, s_f, num_points // 2)

    x1 = np.array([integrate.quad(lambda var: np.cos(k * var ** (n + 1) / (n + 1)), 0, s)[0] for s in s_vals])
    y1 = np.array([integrate.quad(lambda var: np.sin(k * var ** (n + 1) / (n + 1)), 0, s)[0] for s in s_vals])

    x2, y2 = np.dot(
        np.array([[np.cos(2 * theta), np.sin(2 * theta)], [-np.sin(2 * theta), np.cos(2 * theta)]]),
        np.stack([x1, y1], 0),
    )
    x2, y2 = -x2[::-1] + x2[-1] + x1[-1], y2[::-1] - y2[-1] + y1[-1]
    x21 = np.concatenate([x1, x2], 0)
    y21 = np.concatenate([y1, y2], 0)

    x2, y2 = np.dot(np.array([[-1, 0], [0, -1]]), np.stack([x21, y21], 0))
    x2, y2 = x2[::-1] - x2[-1] + x21[-1], y2[::-1] - y2[-1] + y21[-1]
    x_data, y_data = np.concatenate([x21, x2], 0), np.concatenate([y21, y2], 0)

    x = x_data * dx / x_data[-1]
    y = y_data * dy / y_data[-1]
    z = np.linspace(0, dz, x.size)
    return x, y, z


def arc(dy: float, dz: float, num_points: int, radius: float, theta_offset: float = 0, **kwargs: Any | None) -> ptarray:
    thetaf = np.arccos(1 - np.abs(dy) / (2 * radius))
    theta = theta_offset - np.pi / 2 + np.linspace(0, thetaf, num_points)
    x, y = radius * np.cos(theta), radius * (np.sin(theta) + 1)
    z = np.linspace(0, dz, x.size)
    return x, y, z


def circ(dy: float, dz: float, num_points: int, radius: float, **kwargs: Any | None) -> ptarray:
    x1, y1, _ = arc(dy=dy, dz=dz / 2, num_points=num_points, radius=radius, theta_offset=0)
    x2, y2, _ = arc(dy=dy, dz=dz / 2, num_points=num_points, radius=radius, theta_offset=np.pi)

    # Normalize the x2, y2 arrays, flip them and add the last point of x1, y1
    x2 = np.flip(x2 - x2[-1]) + x1[-1]
    y2 = np.flip(y2 - y2[-1]) + y1[-1]

    x = np.concatenate([x1, x2], 0)
    y = np.concatenate([y1, y2], 0)
    z = np.linspace(0, dz, x.size)

    return x, np.sign(dy) * y, z


def series(curve: ptarray, n: int) -> ptarray:
    x1, y1, z1 = curve
    x, y, z = np.array([]), np.array([]), np.array([])

    for i in range(0, n):
        x2 = x1 + x1[-1] * i
        y2 = ((-1) ** i) * y1 + y1[-1] * (1 - (-1) ** i) / 2
        z2 = ((-1) ** i) * z1 + z1[-1] * (1 - (-1) ** i) / 2
        x = np.append(x, x2)
        y = np.append(y, y2)
        z = np.append(z, z2)
    return x, y, z


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # x, y, z = spline(0.5, 0.3, dz=0.0, num_points=100)
    # x, y, z = sin(dx=0.5, dy=0.6, dz=0.0, num_points=100, flat_peaks=1)
    x, y, z = double_sin(dx=0.5, dy1=0.045, dz=0.0, dy2=-0.077, radius=10, num_points=100, flat_peaks=0)
    # x, y, z = spline_bridge(dx=5, dy=0.06, dz=0.03, num_points=100)
    # x, y, z = tanh(dx=10, dy=0.08, dz=0, num_points=200)
    # x, y, z = tanh(dx=1000, dy=0.04, dz=0, num_points=200, s=1/200)
    # z, y, z = erf(dx=10, dy=0.08, dz=0, num_points=200)
    # z, y, z = arctan(dx=10, dy=0.08, dz=0, num_points=200)
    # z, y, z = rad(dx=10, dy=0.08, dz=0, num_points=200)
    # z, y, z = abv(dx=10, dy=0.08, dz=0, num_points=200)
    # x, y, z = euler_S2(theta=np.pi / 24, radius=15000, dx=5, dy=0.40, dz=0, n=1, num_points=1000)
    # x, y, z = euler_S4(theta=np.pi / 24, radius=15000, dx=5, dy=0.40, dz=0, n=1, num_points=1000)
    # x, y, z = circ(dx=5, dy=-0.40, dz=0, num_points=1000, radius=10)

    # x, y, _ = series(erf(s=1, dy=0.040, dx=5, dz=0, num_points=100), 6)

    plt.figure(1)
    plt.clf()
    plt.plot(x, y)
    plt.show()
