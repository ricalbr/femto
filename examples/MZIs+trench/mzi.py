from femto import Cell, _Marker, PGMCompiler, TrenchColumn, _Waveguide
from femto.helpers import dotdict
from femto.PGMCompiler import PGMTrench

# GEOMETRICAL DATA
# Circuit
PARAMETERS_WG = dotdict(
        scan=6,
        speed=20,
        depth=0.035,
        radius=15,
        pitch=0.080,
        pitch_fa=0.127,
        int_dist=0.007,
        int_length=0.0,
        arm_length=0.0,
        lsafe=3
)

x0 = -2.0
y0 = 0.0
z0 = PARAMETERS_WG.depth
increment = [PARAMETERS_WG.lsafe, 0.0, 0.0]

# Markers
PARAMETERS_MK = dotdict(
        scan=1,
        speed=4,
        depth=0.001,
        speed_pos=5,
)
lx = 1
ly = 0.05

# Trench
PARAMETERS_TC = dotdict(
        length=1.0,
        nboxz=4,
        deltaz=0.0015,
        h_box=0.075,
        base_folder=r'C:\Users\Capable\Desktop\RiccardoA',
        y_min=0.08,
        y_max=0.08 + 6 * PARAMETERS_WG.pitch - 0.02
)

# G-CODE DATA
PARAMETERS_GC = dotdict(
        filename='MZI.pgm',
        lab='CAPABLE',
        samplesize=(25, 25),
        rotation_angle=1.0
)

# 20x20 circuit
circ = Cell(PARAMETERS_GC)

# Guida dritta
wg = _Waveguide(PARAMETERS_WG)
wg.start([x0, y0, z0]) \
    .linear([PARAMETERS_GC.samplesize[0] + 4, 0.0, 0.0])
wg.end()
circ.append(wg)

# MZI
delta_x = wg.sbend_length(wg.dy_bend, wg.radius)
l_x = (PARAMETERS_GC.samplesize[0] + 4 - delta_x * 4) / 2
for i in range(6):
    wg = _Waveguide(PARAMETERS_WG) \
        .start([x0, (1 + i) * wg.pitch, z0]) \
        .linear([l_x, 0, 0]) \
        .arc_mzi((-1) ** i * wg.dy_bend) \
        .linear([l_x, 0, 0])
    wg.end()
    circ.append(wg)

# # _Marker

pos = [PARAMETERS_GC.samplesize[0] / 2, y0 - PARAMETERS_WG.pitch]
c = _Marker(PARAMETERS_MK)
c.cross(pos, ly=0.1)
circ.append(c)

# # Trench
PARAMETERS_TC.x_center = PARAMETERS_GC.samplesize[0] / 2
col = TrenchColumn(PARAMETERS_TC)
col.get_trench(circ.waveguides)
circ.append(col)

# Plot
circ.plot2d()
# plt.show()

# _Waveguide G-Code
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(PARAMETERS_WG.scan):
        for i, wg in enumerate(circ.waveguides):
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)

# _Marker G-Code
PARAMETERS_GC.filename = 'Markers.pgm'
with PGMCompiler(PARAMETERS_GC) as gc:
    for mk in circ.markers:
        gc.write(mk.points)

# Trench G-Code

PARAMETERS_GC.filename = 'test'
tc = PGMTrench(PARAMETERS_GC, circ.trench_cols)
tc.write()
# for col_index, col_trench in enumerate(circ.trench_cols):
#     # Generate G-Code for the column
#     col_filename = os.path.join(os.getcwd(), 's-trench', f'FARCALL{col_index + 1:03}')
#     PARAMETERS_GC.filename = col_filename
#     with PGMCompiler(PARAMETERS_GC) as gc:
#         gc.trench(col_trench, col_index, base_folder=PARAMETERS_TC.base_folder)
