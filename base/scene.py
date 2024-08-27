import taichi as ti
from taichi.math import vec3, normalize, cross, sin, cos

from base.camera import PerspectiveCamera


@ti.dataclass
class Scene:
    integrator: ti.i32
    spp: ti.i32
    max_depth: ti.i32
    sample_lights: ti.i32
    sample_bsdf: ti.i32