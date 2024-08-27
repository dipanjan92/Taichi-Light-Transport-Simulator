import taichi as ti
from taichi.math import vec3, inf



@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t):
        return self.origin + t * self.direction


@ti.func
def offset_ray_origin(p, n, w):
    # Compute the error offset
    epsilon = 1e-4
    offset = n * epsilon
    if w.dot(n) < 0:
        offset = -offset

    po = p + offset

    return po


@ti.func
def spawn_ray(p, n, d):
    origin = offset_ray_origin(p, n, d)
    return Ray(origin, d)
