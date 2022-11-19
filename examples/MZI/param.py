from femto.helpers import dotdict

# GEOMETRICAL DATA
PARAMETERS_WG = dotdict(
        scan=6,
        speed=20,
        radius=15,
        depth=0.035,
        pitch=0.080,
        int_dist=0.007,
        lsafe=3,
        arm_length=1.5,
        x_init=-2.0,
        y_init=0.0
)

increment = [PARAMETERS_WG.lsafe, 0.0, 0.0]

d = 0.5 * (PARAMETERS_WG.pitch - PARAMETERS_WG.int_dist)

# G-CODE DATA
PARAMETERS_GC = dotdict(
        filename='MZImultiscan.pgm',
        lab='CAPABLE',
        samplesize=(25, 25),
        rotation_angle=0.0
)
