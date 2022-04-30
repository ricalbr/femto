from femto.objects.Waveguide import Waveguide
import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from typing import List
from itertools import chain
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from shapely.geometry import LineString, Polygon, MultiPolygon, box, polygon


class Trench:
    def __init__(
        self,
        trench_block: Polygon,
        delta_floor: float = 0.001,
        bridge_width: float = 0.026,
        beam_size: float = 0.004,
        round_corner: float = 0.005,
    ):

        self.trench_block = trench_block
        self.delta_floor = delta_floor
        self.bridge_width = bridge_width
        self.beam_size = beam_size
        self.round_corner = round_corner
        self.adj_bridge = (
            self.bridge_width + self.beam_size * 2 - self.round_corner
        ) / 2
        self._wall = None
        self._floor = None

    @property
    def center(self):
        return np.asarray([self.trench_block.centroid.x, self.trench_block.centroid.y])

    @property
    def border(self):
        if self._wall is None:
            self.get_paths()
        return self._wall

    @property
    def floor(self):
        if self._floor is None:
            self.get_paths()
        return self._floor

    @property
    def patch(self, fc="k", ec="k", alpha=0.5, zorder=1):
        return PolygonPatch(self.trench_block, fc=fc, ec=ec, alpha=alpha, zorder=zorder)

    def get_paths(self):
        # Initialization
        polygon_list = [self.trench_block]
        insets = []
        while polygon_list:
            current_poly = polygon.orient(polygon_list.pop(0))
            inset_polygon = self._buffer_polygon(current_poly)
            if inset_polygon and inset_polygon.type == "MultiPolygon":
                # add each polygon to the list as a single Polygon object
                polygon_list.extend(list(inset_polygon))
                for poly in list(inset_polygon):
                    insets.append(list(poly.exterior.coords))
            elif inset_polygon and inset_polygon.type == "Polygon":
                polygon_list.append(inset_polygon)
                insets.append(list(inset_polygon.exterior.coords))
            elif inset_polygon:
                raise ValueError("Unhandled geometry type: " f"{inset_polygon.type}")
        self._wall = np.asarray(list(self.trench_block.exterior.coords)).T
        self._floor = np.asarray(list(chain.from_iterable(insets))).T

    # Private interface
    def _buffer_polygon(self, polygon, inset=True):
        if inset:
            new_polygon = polygon.buffer(-self.delta_floor)
        else:
            new_polygon = polygon.buffer(self.delta_floor)
        return new_polygon if new_polygon.is_valid else None


class TrenchColumn:
    def __init__(
        self,
        x_c: float = None,
        y_min: float = None,
        y_max: float = None,
        length: float = 1.0,
        delta_floor: float = 0.001,
        bridge_width: float = 0.026,
        beam_size: float = 0.004,
        round_corner: float = 0.005,
    ):

        self.trench_list = []
        self.length = length
        self._x_c = x_c
        self._y_min = y_min
        self._y_max = y_max

        self._rect = self._make_box()

        self.delta_floor = delta_floor
        self.bridge_width = bridge_width
        self.beam_size = beam_size
        self.round_corner = round_corner
        self.adj_bridge = (
            self.bridge_width + self.beam_size * 2 - self.round_corner
        ) / 2

    # # add gettere and setters for x_c, y_min, y_max, length
    @property
    def x_c(self):
        return self._x_c

    @x_c.setter
    def x_c(self, x_center):
        self._x_c = x_center
        if x_center is not None:
            self._rect = self._make_box()

    @property
    def y_min(self):
        return self._y_min

    @y_min.setter
    def y_min(self, y_min):
        self._y_min = y_min
        if y_min is not None:
            self._rect = self._make_box()

    @property
    def y_max(self):
        return self._y_max

    @y_max.setter
    def y_max(self, y_max):
        self._y_max = y_max
        if y_max is not None:
            self._rect = self._make_box()

    @property
    def patch(self, fc="k", ec="k", alpha=0.5, zorder=1):
        if isinstance(self._rect, MultiPolygon):
            return PolygonPatch(self._rect, fc=fc, ec=ec, alpha=alpha, zorder=zorder)

    def get_trench(self, circuit: List):

        assert all(
            [isinstance(wg, Waveguide) for wg in circuit]
        ), "Objects in circuit list are not of type Waveguide."

        for wg in circuit:
            x, y = np.array(wg.M["x"][:-2]), np.array(wg.M["y"][:-2])
            dilated = LineString(list(zip(x, y))).buffer(self.adj_bridge, cap_style=1)
            self._rect = self._rect.difference(dilated)

        for block in list(self._rect):
            block = polygon.orient(block).buffer(
                self.round_corner, resolution=150, cap_style=1
            )
            trench = Trench(
                block,
                self.delta_floor,
                self.bridge_width,
                self.beam_size,
                self.round_corner,
            )
            self.trench_list.append(trench)

    # Private interface
    def _make_box(self):
        if (
            self._x_c is not None
            and self._y_min is not None
            and self._y_max is not None
        ):
            return box(
                self._x_c - self.length / 2,
                self._y_min,
                self._x_c + self.length / 2,
                self._y_max,
            )
        else:
            return None


if __name__ == "__main__":

    # Data
    pitch = 0.080
    int_dist = 0.007
    d_bend = 0.5 * (pitch - int_dist)
    delta_floor = 0.001
    bridge = 0.026
    beam_size = 0.004
    round_corner = 0.005
    adj_bridge = (bridge + beam_size) / 2 - round_corner

    # Calculations
    coup = [Waveguide(num_scan=6) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * pitch, 0.035])
        wg.sin_acc((-1) ** i * d_bend, radius=15, speed=20, N=250)
        x_mid = wg.x[-1]
        wg.sin_acc((-1) ** i * d_bend, radius=15, speed=20, N=250)
        wg.end()

    TC1 = TrenchColumn(x_c=x_mid, y_min=-0.1, y_max=19 * pitch + 0.1)
    TC1.get_trench(coup)

    fig, ax = plt.subplots()
    for wg in coup:
        ax.plot(wg.x[:-1], wg.y[:-1], "b")
    for t in TC1.trench_list:
        xt, yt = t.floor
        ax.plot(xt, yt, "k", lw=0.5)
    ax.set_aspect("equal")
    # plt.show()
