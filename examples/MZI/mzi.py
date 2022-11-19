import numpy as np

from femto import Cell, PGMCompiler, _Waveguide
from param import *

np.set_printoptions(formatter={'float': "\t{: 0.6f}".format})

# Simple script for fabricate a MZI interferometer
mzi = Cell(PARAMETERS_GC)

for i in range(2):
    wg = _Waveguide(PARAMETERS_WG)
    wg.start([wg.x_init, wg.y_init - (0.5 - i) * PARAMETERS_WG.pitch, wg.depth]) \
        .linear(increment) \
        .sin_mzi((-1) ** i * d, arm_length=wg.arm_length) \
        .linear(increment)
    wg.end()
    mzi.append(wg)

# Plot
mzi.plot2d()

# Compilation
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(PARAMETERS_WG.scan):
        for i, wg in enumerate(mzi.waveguides):
            gc.comment(f'Modo: {i + 1}')
            gc.write(wg.points)
