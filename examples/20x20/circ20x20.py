from femto.objects.Waveguide import Waveguide
from femto.objects.Trench import TrenchColumn
from femto.objects.Marker import Marker
from femto.compiler.PGMCompiler import PGMCompiler
import matplotlib.pyplot as plt
import param as p
import time

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('matplotlib', 'qt5')

t0 = time.perf_counter()

# 20x20 circuit
circ = {
    "waveguide": [Waveguide() for _ in range(p.MM)],
    "marker": [Marker() for _ in range(p.MM)],
    "trench": [
        TrenchColumn(
            y_min=p.y0 - 0.5 * (p.MM + 1) * p.pitch - 0.1,
            y_max=p.y0 + (p.MM - 1 - 0.5 * (p.MM + 1)) * p.pitch + 0.1,
        )
        for _ in range(p.NN)
    ],
}

x_trench = []
for i, wg in enumerate(circ["waveguide"]):
    [xi, yi, zi] = [p.x0, p.y0 + (i - 0.5 * (p.MM + 1)) * p.pitch, p.depth]

    wg.start([xi, yi, zi])
    wg.linear(p.increment, speed=p.speed)
    wg.sin_bend((-1) ** (i % 2) * p.d1, p.radius, speed=p.speed, N=200)
    for j in range(p.NN - 1):
        wg.sin_bend((-1) ** (j + i % 2 + 1) * p.d1, p.radius, speed=p.speed, N=200)
        if i == 0:
            xl, yl, _ = wg.lastpt
            circ["marker"][j].cross([xl, yl - 0.2], p.lx, p.ly)
            x_trench.append(xl)
        wg.sin_bend((-1) ** (j + i % 2) * p.d1, p.radius, speed=p.speed, N=200)
        wg.sin_bend((-1) ** (j + i % 2 + 1) * p.d2, p.radius, speed=p.speed, N=200)
    wg.sin_bend((-1) ** (j + i % 2) * p.d1, p.radius, speed=p.speed, N=200)
    if i == 0:
        xl, yl, _ = wg.lastpt
        circ["marker"][j + 1].cross([xl, yl - 0.2], p.lx, p.ly)
        x_trench.append(xl)
    wg.sin_acc((-1) ** (j + i % 2 + 1) * p.d1, p.radius, speed=p.speed, N=200)
    wg.linear(p.increment, speed=p.speed)
    wg.end()

# Trench
for xt, col in zip(x_trench, circ["trench"]):
    col.x_c = xt
    col.get_trench(circ["waveguide"])

# Plot
fig, ax = plt.subplots()
for wg in circ["waveguide"]:
    ax.plot(wg.x[:-1], wg.y[:-1], "-b", linewidth=2.5)
    ax.plot(wg.x[-2:], wg.y[-2:], ":k", linewidth=1)

for c in circ["marker"]:
    ax.plot(c.x[:-1], c.y[:-1], "-k", linewidth=1)

for col in circ["trench"]:
    for t in col.trench_list:
        ax.add_patch(t.patch)
plt.tight_layout(pad=0)

# Compilation
# # OPTICAL CIRCUIT
gc = PGMCompiler("20x20CIRC.pgm", ind_rif=p.ind_env, angle=p.angle)
gc.header()
gc.rpt(wg.num_scan)
for i, wg in enumerate(circ["waveguide"]):
    gc.comment(f" +--- Modo: {i+1} ---+")
    gc.point_to_instruction(wg.M)
gc.endrpt()
gc.compile_pgm()

# # MARKERS
gc = PGMCompiler("20x20MARKERS.pgm", ind_rif=p.ind_env, angle=p.angle)
gc.header()
for i, c in enumerate(circ["marker"]):
    gc.comment(f" +--- Croce: {i+1} ---+")
    gc.rpt(c.num_scan)
    gc.point_to_instruction(c.M)
    gc.endrpt()
gc.compile_pgm()

# # TRENCH
# TODO: PGM per fabbricazione trench

print(f"Elapsed time: {time.perf_counter() - t0:.2f} s.")
plt.show()
