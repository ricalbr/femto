from __future__ import annotations

from femto.curves import sin
from femto.device import Device
from femto.pgmcompiler import PGMCompiler
from femto.waveguide import Waveguide

# WAVEGUIDE PARAMETERS
PARAM_WG = dict(
    scan=6,
    speed=20,
    radius=15,
    depth=0.035,
    pitch=0.080,
    int_dist=0.007,
    lsafe=2,
    arm_length=1.5,
    x_init=-2.0,
    y_init=1,
    samplesize=(25, 2),
)

# G-CODE PARAMETERS
PARAM_GC = dict(
    filename='MZI_multiscan.pgm',
    laser='pharos',
    samplesize=(25, 2),
    rotation_angle=0.0,
)

# Simple script for a MZI interferometer
mzi = []
for i in range(2):
    wg = Waveguide(**PARAM_WG)
    wg.start([wg.x_init, wg.y_init - (0.5 - i) * PARAM_WG['pitch'], wg.depth])
    wg.linear([5, 0, 0])
    wg.mzi(dy=(-1) ** i * wg.dy_bend, dz=0, arm_length=wg.arm_length, fx=sin)
    wg.linear([wg.x_end, None, None], mode='ABS')
    wg.end()
    mzi.append(wg)

# Create a device
dev = Device(**PARAM_GC)
dev.add(mzi)

# Plot and pgm
dev.plot3d()
# dev.pgm() -> Device's built-in .pgm generation method

# Compilation
# Custom .pgm files can be exported using the methods of PGMCompiler
with PGMCompiler(**PARAM_GC) as G:
    G.tic()  # print start time
    with G.repeat(PARAM_WG['scan']):
        for i, wg in enumerate(mzi):
            G.comment(f'Mode: {i + 1}')
            G.write(wg.points)
    G.toc()  # print end time
