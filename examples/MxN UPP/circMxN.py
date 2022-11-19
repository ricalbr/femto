import time

from femto import Cell, _Marker, TrenchColumn, _Waveguide
from param import *

t0 = time.perf_counter()

# MxN circuit
circ = Cell(PARAMETERS_GC)

x_trench = []
upp = []
for i in range(MM):
    [xi, yi, zi] = [x0, y0 + (i - 0.5 * (MM - 1)) * PARAMETERS_WG.pitch, PARAMETERS_WG.depth]

    wg = _Waveguide(PARAMETERS_WG)
    wg.start([xi, yi, zi])
    wg.linear(increment)
    wg.sin_bend((-1) ** (i % 2) * d1)
    for j in range(NN - 1):
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d1)
        if i == 0:
            xl, yl, _ = wg.lastpt
            mk = _Marker(PARAMETERS_MK)
            mk.cross([xl, yl - 0.2], lx, ly)
            circ.append(mk)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * d1)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * d2)
    wg.sin_bend((-1) ** (j + i % 2) * d1)
    if i == 0:
        xl, yl, _ = wg.lastpt
        mk = _Marker(PARAMETERS_MK)
        mk.cross([xl, yl - 0.2], lx, ly)
        circ.append(mk)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * d1)
    wg.linear([wg.x_end, wg.lasty, wg.lastz], mode='ABS')
    wg.end()
    upp.append(wg)

wgp1 = _Waveguide(PARAMETERS_WG)
p_init = [x0, y0 - 0.5 * (MM + 1) * wgp1.pitch, wgp1.depth]
wgp1.start(p_init).linear([wgp1.x_end, wgp1.lasty, wgp1.lastz], mode='ABS').end()

wgp2 = _Waveguide(PARAMETERS_WG)
p_init = [x0, y0 + 0.5 * (MM + 1) * wgp1.pitch, wgp1.depth]
wgp2.start(p_init).linear([wgp2.x_end, wgp2.lasty, wgp2.lastz], mode='ABS').end()

# Add waveguides and circuit to the cell
circ.append(wgp1)
circ.append(upp)
circ.append(wgp2)

# Trench
for xt in x_trench:
    col = TrenchColumn(PARAMETERS_TC)
    col.x_center = xt
    col.y_min = y0 - 0.5 * MM * PARAMETERS_WG.pitch
    col.y_max = y0 + 0.5 * MM * PARAMETERS_WG.pitch
    col.get_trench(circ.waveguides)
    circ.append(col)

# # Plot and compilation
circ.plot2d()
# circ.pgm()
