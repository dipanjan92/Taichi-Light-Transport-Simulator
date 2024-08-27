import taichi as ti
from taichi.math import vec3, normalize, atan2, acos, pi, sqrt, dot

from primitives.primitives import Primitive


@ti.dataclass
class Intersection:
    min_distance: ti.f32
    intersected_point: vec3
    normal: vec3
    shading_normal: vec3
    dpdu: vec3  # Partial derivative of the surface with respect to u
    dpdv: vec3  # Partial derivative of the surface with respect to v
    dndu: vec3  # Partial derivative of the normal with respect to u
    dndv: vec3  # Partial derivative of the normal with respect to v
    nearest_object: Primitive
    intersected: ti.i32

    @ti.func
    def set_intersection(self, ray, prim, tMax):
        self.intersected = 1
        self.min_distance = tMax
        self.intersected_point = ray.origin + self.min_distance * ray.direction
        self.nearest_object = prim
        if prim.shape_type == 0:
            self.normal = prim.triangle.normal
            self.calculate_triangle_interaction(prim.triangle)
        elif prim.shape_type == 1:
            self.normal = normalize(self.intersected_point - prim.sphere.center)
            self.calculate_sphere_interaction(prim.sphere)
        self.shading_normal = self.normal  # flat shading for now


    @ti.func
    def Le(self, d):
        L = vec3(0.0)
        if self.nearest_object.shape_type == 0 and self.nearest_object.is_light:
            # Area Light L
            if dot(self.normal, d) >= 0:
                L += self.nearest_object.material.emission
        return L


    @ti.func
    def calculate_triangle_interaction(self, triangle):
        self.dpdu = triangle.edge_1  # For flat shading, dpdu could be set to an edge
        self.dpdv = triangle.edge_2  # Similarly, dpdv could be set to another edge

        # For flat shading, the normal is constant, so derivatives are zero
        self.dndu = vec3(0.0)
        self.dndv = vec3(0.0)


    @ti.func
    def calculate_sphere_interaction(self, sphere):
        pHit = self.intersected_point - sphere.center
        radius = sphere.radius

        phi = atan2(pHit.y, pHit.x)
        cosTheta = pHit.z / radius
        theta = acos(cosTheta)

        u = phi / (2 * pi)
        v = (theta - (-pi / 2)) / (pi)

        zRadius = sqrt(pHit.x * pHit.x + pHit.y * pHit.y)
        cosPhi = pHit.x / zRadius if zRadius != 0 else 1.0
        sinPhi = pHit.y / zRadius if zRadius != 0 else 0.0
        self.dpdu = vec3(-2 * pi * pHit.y, 2 * pi * pHit.x, 0)

        sinTheta = sqrt(1 - cosTheta * cosTheta)
        self.dpdv = pi * vec3(pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinTheta)

        self.dndu = vec3(0.0)  # For a perfect sphere, these derivatives are often set to zero
        self.dndv = vec3(0.0)

    @ti.func
    def get_bsdf(self):
        bsdf = self.nearest_object.bsdf
        bsdf.init_frame(self.shading_normal, self.dpdu)
        return bsdf


