from __future__ import annotations

from femto.curves import sin
from femto.device import Device
from femto.pgmcompiler import PGMCompiler
from femto.waveguide import Waveguide

PARAM_WG = dict(
    scan=4,
    speed=7.5,
    depth=0.050,
    radius=25,
)

# WAVEGUIDES
wgs = []

# SWG
wg = Waveguide(**PARAM_WG)
wg.start([-2, 4.5, 0.050])
wg.linear([27, 4.5, 0.050], mode='ABS')
wg.end()
wgs.append(wg)

# MZI
for i in range(6):
    wg = Waveguide(**PARAM_WG)
    wg.start([-2, 5 + i * 0.080, 0.500])
    wg.linear([10, 0, 0], mode='INC')
    wg.mzi(dy=(-1) ** i * 0.037, dz=0, fx=sin)
    wg.linear([27, 5 + i * 0.080, 0.500], mode='ABS')
    wg.end()
    wgs.append(wg)

PARAM_GC = dict(
    filename='MZIs.pgm',
    laser='PHAROS',
    samplesize=(25, 10),
    rotation_angle=0.0,
)

# CIRCUIT
circuit = Device(**PARAM_GC)

# Add waveguides to circuit
circuit.add(wgs)

# Make a plot of the circuit
circuit.plot2d()

# Export G-Code file
with PGMCompiler(**PARAM_GC) as G:
    G.tic()
    with G.repeat(6):
        for i, wg in enumerate(wgs):
            G.comment(f' +--- Mode: {i + 1} ---+')
            G.write(wg.points)
    G.toc()
    G.go_origin()
