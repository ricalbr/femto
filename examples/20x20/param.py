import math

# %% GEOMETRICAL DATA
MM = 20
NN = 20

# Circuit
radius = 15
pitch = 0.080
pitch_fa = 0.127
depth = 0.035
int_distance = 0.007
int_length = 0.0
length_arm = 0.0
speed = 20
swg_length = 3
increment = [swg_length, 0.0, 0.0]

x0 = -2.0
y0 = 0.0
z0 = depth

d1 = 0.5 * (pitch - int_distance)
d2 = pitch - int_distance

# Markers
lx = 1
ly = 0.05

# %% G-CODE DATA
n_scan = 6
ind_glass = 1.5
ind_water = 1.33
ind_env = ind_glass / ind_water
angle = 1.35
