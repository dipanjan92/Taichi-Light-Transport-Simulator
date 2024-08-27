import taichi as ti
from taichi.math import vec3, cross, dot, sqrt, normalize, vec2, isinf, pi, sin, cos, min, max

from base.bsdf import BSDF
from base.materials import Material
from base.samplers import ShapeSample
from utils.constants import INF
from utils.misc import spherical_triangle_area, length_squared, distance_squared


@ti.dataclass
class Triangle:
    vertex_1: vec3
    vertex_2: vec3
    vertex_3: vec3
    centroid: vec3
    normal: vec3
    edge_1: vec3
    edge_2: vec3

    @ti.func
    def intersect(self, ray_o, ray_d, tMax):
        p0 = self.vertex_1
        p1 = self.vertex_2
        p2 = self.vertex_3
        # Check if triangle is degenerate
        cross_product = cross((p2 - p0), (p1 - p0))
        hit = False
        b0 = b1 = b2 = t = 0.0

        norm_sqr = dot(cross_product, cross_product)
        if norm_sqr != 0:
            # Transform triangle vertices to ray coordinate space
            p0t = p0 - ray_o
            p1t = p1 - ray_o
            p2t = p2 - ray_o

            # Permute components of triangle vertices and ray direction
            abs_ray_d = ti.abs(ray_d)
            kz = 0
            if abs_ray_d[1] > abs_ray_d[kz]:
                kz = 1
            if abs_ray_d[2] > abs_ray_d[kz]:
                kz = 2
            kx = (kz + 1) % 3
            ky = (kx + 1) % 3

            d = ti.Vector([ray_d[kx], ray_d[ky], ray_d[kz]])
            p0t = ti.Vector([p0t[kx], p0t[ky], p0t[kz]])
            p1t = ti.Vector([p1t[kx], p1t[ky], p1t[kz]])
            p2t = ti.Vector([p2t[kx], p2t[ky], p2t[kz]])

            # Apply shear transformation to translated vertex positions
            Sx = -d[0] / d[2]
            Sy = -d[1] / d[2]
            Sz = 1 / d[2]

            p0t[0] += Sx * p0t[2]
            p0t[1] += Sy * p0t[2]
            p1t[0] += Sx * p1t[2]
            p1t[1] += Sy * p1t[2]
            p2t[0] += Sx * p2t[2]
            p2t[1] += Sy * p2t[2]

            # Compute edge function coefficients
            e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
            e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
            e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]

            # Handle precision issues and determinant
            if (e0 >= 0 and e1 >= 0 and e2 >= 0) or (e0 <= 0 and e1 <= 0 and e2 <= 0):
                det = e0 + e1 + e2
                if det != 0:
                    # Compute scaled hit distance to triangle and test against ray t range
                    p0t[2] *= Sz
                    p1t[2] *= Sz
                    p2t[2] *= Sz
                    tScaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]

                    if (det < 0 and tScaled < 0 and tScaled >= tMax * det) or (
                            det > 0 and tScaled > 0 and tScaled <= tMax * det):
                        # Compute barycentric coordinates and t value for triangle intersection
                        invDet = 1 / det
                        b0 = e0 * invDet
                        b1 = e1 * invDet
                        b2 = e2 * invDet
                        t = tScaled * invDet

                        # Check for valid intersection
                        if t > 0:
                            hit = True

        return hit, t  # (b0, b1, b2, t)

    @ti.func
    def get_area(self):
        return 0.5 * cross(self.edge_1, self.edge_2).norm()

    @ti.func
    def get_bounds(self):
        min_p = ti.min(ti.min(self.vertex_1, self.vertex_2), self.vertex_3)
        max_p = ti.max(ti.max(self.vertex_1, self.vertex_2), self.vertex_3)
        return min_p, max_p

    @ti.func
    def get_pdf(self):
        return 1 / self.get_area()

    @ti.func
    def sample(self, u):
        # Sample point on triangle uniformly by area
        b = self.sample_uniform_triangle(u)
        p = b[0] * self.vertex_1 + b[1] * self.vertex_2 + b[2] * self.vertex_3
        n = self.normal
        pdf = self.get_pdf()

        return ShapeSample(
            p=p,
            n=n,
            pdf=pdf
        )

    @ti.func
    def sample_p(self, p, u):
        result_ss = ShapeSample()
        ss = self.sample(u)
        pdf = ss.pdf
        wi = ss.p - p

        if length_squared(wi) == 0:
            pdf = 0
        else:
            wi = normalize(wi)
            pdf *= distance_squared(p, ss.p) / ti.abs(dot(ss.n, -wi))
            if isinf(pdf):
                pdf = 0
        result_ss.pdf = pdf
        result_ss.p = ss.p
        result_ss.n = ss.n

        return result_ss

    @ti.func
    def PDF(self, p, wi):
        pdf = 0.0
        ray_origin = p
        ray_direction = wi
        hit, t = self.intersect(ray_origin, ray_direction, INF)
        isec_p = ray_origin + t * ray_direction
        if hit:
            cos_theta = ti.abs(dot(self.normal, -wi))
            if cos_theta > 0:
                pdf = self.get_pdf() / (cos_theta / distance_squared(p, isec_p))
        return pdf

    @ti.func
    def sample_uniform_triangle(self, u):
        b0 = 0.0
        b1 = 0.0
        if u[0] < u[1]:
            b0 = u[0] / 2
            b1 = u[1] - b0
        else:
            b1 = u[1] / 2
            b0 = u[0] - b1
        return vec3(b0, b1, 1 - b0 - b1)

    @ti.func
    def solid_angle(self, p):
        return spherical_triangle_area(self.vertex_1 - p, self.vertex_2 - p, self.vertex_3 - p)


@ti.dataclass
class Sphere():
    center: vec3
    radius: ti.f32

    @ti.func
    def intersect(self, ray_origin, ray_dir, tMax):
        oc = ray_origin - self.center
        a = dot(ray_dir, ray_dir)
        b = 2.0 * dot(oc, ray_dir)
        c = dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        hit = False
        t = 0.0
        # p = ti.Vector([0.0, 0.0, 0.0])
        # n = ti.Vector([0.0, 0.0, 0.0])
        # uv = ti.Vector([0.0, 0.0])

        if discriminant > 0:
            t = (-b - sqrt(discriminant)) / (2.0 * a)
            if t > 0:
                hit = True

        return hit, t  # , p, n, uv

    @ti.func
    def get_bounds(self):
        min_p = self.center - self.radius
        max_p = self.center + self.radius
        return min_p, max_p

    @ti.func
    def get_area(self):
        return 4.0 * ti.math.pi * self.radius ** 2

    @ti.func
    def get_pdf(self):
        return 1.0 / self.get_area()

    @ti.func
    def sample_uniform_sphere(self, u):
        """Sample a point uniformly on the surface of a sphere."""
        z = 1 - 2 * u[0]
        r = ti.sqrt(max(0.0, 1 - z * z))
        phi = 2 * pi * u[1]
        x = r * cos(phi)
        y = r * sin(phi)
        return vec3(x, y, z)

    @ti.func
    def sample(self, u):
        # Sample point on the sphere uniformly by area
        b = self.sample_uniform_sphere(u)
        p = self.center + b * self.radius
        n = b
        pdf = self.get_pdf()

        return ShapeSample(
            p=p,
            n=n,
            pdf=pdf
        )

    @ti.func
    def sample_p(self, p, u):
        result_ss = ShapeSample()
        ss = self.sample(u)
        wi = ss.p - p
        dist_sq = length_squared(wi)

        if dist_sq > 0:
            wi = normalize(wi)
            cos_theta = ti.abs(dot(ss.n, -wi))
            if cos_theta > 0:
                pdf = ss.pdf / (cos_theta / dist_sq)
                result_ss.pdf = pdf
                result_ss.p = ss.p
                result_ss.n = ss.n

        return result_ss

    @ti.func
    def PDF(self, p, wi):
        pdf = 0
        ray_origin = p
        ray_direction = wi
        hit, t = self.intersect(ray_origin, ray_direction, INF)
        isec_p = ray_origin + t * ray_direction
        if hit:
            cos_theta = ti.abs(dot(isec_p - self.center, -wi)) / self.radius
            if cos_theta > 0:
                pdf = self.get_pdf() / (cos_theta / distance_squared(p, isec_p))
        return pdf


@ti.dataclass
class Primitive:
    shape_type: ti.i32
    triangle: Triangle
    sphere: Sphere
    material: Material
    bsdf: BSDF
    is_light: ti.i32
    light_idx: ti.i32

    @ti.func
    def intersect(self, ray_origin, ray_dir, tMax):
        hit, t = 0, 0.0
        if self.shape_type == 0:
            hit, t = self.triangle.intersect(ray_origin, ray_dir, tMax)
        elif self.shape_type == 1:
            hit, t = self.sphere.intersect(ray_origin, ray_dir, tMax)
        return hit, t

    @ti.func
    def get_area(self):
        to_return = 0.0
        if self.shape_type == 0:
            to_return = self.triangle.get_area()
        elif self.shape_type == 1:
            to_return = self.sphere.get_area()
        return to_return

    @ti.func
    def get_bounds(self):
        min_p, max_p = vec3(0.0), vec3(0.0)
        if self.shape_type == 0:
            min_p, max_p = self.triangle.get_bounds()
        elif self.shape_type == 1:
            min_p, max_p = self.sphere.get_bounds()
        return min_p, max_p

    @ti.func
    def get_pdf(self):
        to_return = 0.0
        if self.shape_type == 0:
            to_return = self.triangle.get_pdf()
        elif self.shape_type == 1:
            to_return = self.sphere.get_pdf()
        return to_return
