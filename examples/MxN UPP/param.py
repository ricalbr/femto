from femto.helpers import dotdict

# GEOMETRICAL DATA
MM = 20
NN = 20

PARAMETERS_WG = dotdict(
        scan=6,
        speed=20,
        radius=15,
        depth=0.035,
        pitch=0.080,
        int_dist=0.007,
        lsafe=3,
        samplesize=(110, 5)
)

x0 = -2.0
y0 = 2.50
increment = [PARAMETERS_WG.lsafe, 0.0, 0.0]

d1 = 0.5 * (PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist)
d2 = PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist

# Markers
PARAMETERS_MK = dotdict(
        scan=1,
        speed=4,
        depth=0.000,
        speedpos=5,
)
lx = 1
ly = 0.05

# Trench
PARAMETERS_TC = dotdict(
        length=1.0,
        nboxz=4,
        deltaz=0.0015,
        h_box=0.075,
        base_folder=r'',
        u=[34.6, 36.4]
)

# G-CODE DATA
PARAMETERS_GC = dotdict(
        filename='UPP',
        lab='CAPABLE',
        samplesize=PARAMETERS_WG.samplesize,
        rotation_angle=0.0
)
