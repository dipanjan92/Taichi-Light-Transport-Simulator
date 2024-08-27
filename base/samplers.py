import taichi as ti
from taichi.math import pi, length, normalize, dot, vec3, sqrt, vec2, sin, cos

from utils.constants import inv_pi


@ti.dataclass
class ShapeSample:
    p: vec3  # Point of intersection
    n: vec3  # Normal at intersection
    pdf: ti.f32  # Probability density of the sample


@ti.func
def sample_uniform_sphere(u):
    z = 1 - 2 * u[0]
    r = sqrt(max(0.0, 1 - z * z))
    phi = 2 * pi * u[1]
    return vec3(r * cos(phi), r * sin(phi), z)


@ti.func
def uniform_sphere_pdf():
    return 1 / (4 * pi)


@ti.func
def sample_uniform_hemisphere(u2):
    z = u2[0]
    r = sqrt(max(0.0, 1.0 - z * z))
    phi = 2 * pi * u2[1]
    x = r * cos(phi)
    y = r * sin(phi)
    return vec3(x, y, z)


@ti.func
def uniform_hemisphere_pdf():
    return 1.0 / (2.0 * pi)


@ti.func
def sample_uniform_disk_concentric(u):
    r = sqrt(u[0])
    theta = 2 * pi * u[1]
    return vec2(r * cos(theta), r * sin(theta))


@ti.func
def get_shape_pdf(self, intr, wi):
    pdf = intr.primitive.get_pdf()
    # convert to solid angle
    to_center = self.center[None] - intr.intersected_point
    distance = length(to_center)
    cos_theta = dot(normalize(to_center), wi)
    if cos_theta > 0:
        return ((distance * distance) / cos_theta) * pdf
    return 0.0


@ti.func
def sample_uniform_disk_polar(u):
    r = ti.sqrt(u[0])
    theta = 2 * pi * u[1]
    return vec2(r * cos(theta), r * sin(theta))


@ti.func
def concentric_sample_disk(u):
    u_offset = 2.0 * u - vec2(1, 1)
    to_return = vec2(0, 0)
    if u_offset.x != 0 and u_offset.y != 0:
        r, theta = 0.0, 0.0
        if ti.abs(u_offset.x) > ti.abs(u_offset.y):
            r = u_offset.x
            theta = pi / 4 * (u_offset.y / u_offset.x)
        else:
            r = u_offset.y
            theta = pi / 2 - pi / 4 * (u_offset.x / u_offset.y)
        to_return = r * vec2(cos(theta), sin(theta))
    return to_return


@ti.func
def sample_cosine_hemisphere(u):
    d = concentric_sample_disk(u)
    z = ti.sqrt(ti.max(0.0, 1 - d.x * d.x - d.y * d.y))
    return vec3(d.x, d.y, z)


@ti.func
def cosine_hemisphere_pdf(cos_theta):
    return cos_theta * inv_pi