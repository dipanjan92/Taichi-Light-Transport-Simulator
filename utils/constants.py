import enum

import taichi as ti

# Constants
inv_pi = 1.0 / ti.math.pi
inv_2_pi = 0.5 * inv_pi
inv_4_pi = 0.25 * inv_pi
pi_over_2 = ti.math.pi / 2.0
pi_over_4 = 0.5 * pi_over_2
EPSILON = 1e-6
INF = 1000000.0
MAX_DEPTH = 64

# Vectors
ZEROS = ti.math.vec3(0.0)
ONES = ti.math.vec3(1.0)
BLUE = ti.Vector([0.25, 0.25, 0.75])