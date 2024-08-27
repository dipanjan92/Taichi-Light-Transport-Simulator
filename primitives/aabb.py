import taichi as ti
from taichi.math import vec3, isinf

from primitives.primitives import Primitive
from utils.constants import INF


@ti.dataclass
class AABB:
    min_point: vec3
    max_point: vec3
    centroid: vec3

    @ti.func
    def update_centroid(self):
        self.centroid = (self.min_point + self.max_point) * 0.5
        # print(f"Updated centroid: {self.centroid}")

    @ti.func
    def aabb_intersect(self, ray):
        t_min = 0.0
        t_max = INF
        ray_inv_dir = 1 / ray.direction
        for i in range(3):
            t1 = (self.min_point[i] - ray.origin[i]) * ray_inv_dir[i]
            t2 = (self.max_point[i] - ray.origin[i]) * ray_inv_dir[i]
            t_min = ti.min(ti.max(t1, t_min), ti.max(t2, t_min))
            t_max = ti.max(ti.min(t1, t_max), ti.min(t2, t_max))
        return t_min <= t_max

    @ti.func
    def get_largest_dim(self):
        to_return = 0
        dx = abs(self.max_point[0] - self.min_point[0])
        dy = abs(self.max_point[1] - self.min_point[1])
        dz = abs(self.max_point[2] - self.min_point[2])
        if dx > dy and dx > dz:
            to_return = 0
        elif dy > dz:
            to_return = 1
        else:
            to_return = 2
        # print(f"Largest dimension: {to_return} with dx={dx}, dy={dy}, dz={dz}")
        return to_return

    @ti.func
    def offset(self, point):
        o = point - self.min_point
        if self.max_point[0] > self.min_point[0]:
            o[0] /= self.max_point[0] - self.min_point[0]

        if self.max_point[1] > self.min_point[1]:
            o[1] /= self.max_point[1] - self.min_point[1]

        if self.max_point[2] > self.min_point[2]:
            o[2] /= self.max_point[2] - self.min_point[2]

        return o

    @ti.func
    def get_surface_area(self):
        diagonal = self.max_point - self.min_point
        surface_area = 2 * (diagonal[0] * diagonal[1] + diagonal[0] * diagonal[2] + diagonal[1] * diagonal[2])
        # print(f"Surface area: {surface_area}")
        return surface_area

    @ti.func
    def is_empty_box(self):
        return (self.min_point[0]==INF) and (self.max_point[0]==INF) and (self.min_point[0] > self.max_point[0])

    @ti.func
    def union_p(self, p):
        self.min_point = ti.min(self.min_point, p)
        self.max_point = ti.max(self.max_point, p)
        self.update_centroid()
        # print(f"Union with point: {p}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
        return self

    @ti.func
    def union(self, b):
        self.min_point = ti.min(self.min_point, b.min_point)
        self.max_point = ti.max(self.max_point, b.max_point)
        self.update_centroid()
        # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
        return self

    @ti.func
    def contains(self, other):
        return (self.min_point[0] <= other.min_point[0] and
                self.min_point[1] <= other.min_point[1] and
                self.min_point[2] <= other.min_point[2] and
                self.max_point[0] >= other.max_point[0] and
                self.max_point[1] >= other.max_point[1] and
                self.max_point[2] >= other.max_point[2])

    @ti.func
    def equal_bounds(self):
        equal = 1
        for i in range(3):
            if self.max_point[i] != self.min_point[i]:
                equal = 0
                break
        return equal



@ti.dataclass
class BVHPrimitive:
    prim: Primitive
    prim_num: ti.i32
    bounds: AABB


@ti.func
def union(b1, b2):
    b3 = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
    b3.min_point = ti.min(b1.min_point, b2.min_point)
    b3.max_point = ti.max(b1.max_point, b2.max_point)
    b3.update_centroid()
    # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
    return b3


@ti.func
def union_p(b1, p1):
    b2 = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
    b2.min_point = ti.min(b1.min_point, p1)
    b2.max_point = ti.max(b1.max_point, p1)
    b2.update_centroid()
    # print(f"Union with AABB: min_point={b.min_point}, max_point={b.max_point}, resulting min_point: {self.min_point}, max_point: {self.max_point}, centroid: {self.centroid}")
    return b2


# @ti.func
# def intersect_bounds(aabb, ray, inv_dir):
#     t1 = (aabb.min_point - ray.origin) * inv_dir
#     t2 = (aabb.max_point - ray.origin) * inv_dir
#
#     tmin = ti.min(t1, t2)
#     tmax = ti.max(t1, t2)
#
#     tmin_final = ti.max(tmin[0], ti.max(tmin[1], tmin[2]))
#     tmax_final = ti.min(tmax[0], ti.min(tmax[1], tmax[2]))
#
#     epsilon = 1e-6
#     intersected = (tmax_final + epsilon > tmin_final) and (tmin_final < ray.tmax) and (tmax_final > ray.tmin)
#     return intersected


@ti.func
def intersect_bounds(aabb, ray, inv_dir):
    result = 0
    tmin = (aabb.min_point[0] - ray.origin[0]) * inv_dir[0]
    tmax = (aabb.max_point[0] - ray.origin[0]) * inv_dir[0]

    if inv_dir[0] < 0:
        tmin, tmax = tmax, tmin

    tymin = (aabb.min_point[1] - ray.origin[1]) * inv_dir[1]
    tymax = (aabb.max_point[1] - ray.origin[1]) * inv_dir[1]

    if inv_dir[1] < 0:
        tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
        result = 0

    else:
        if tymin > tmin:
            tmin = tymin

        if tymax < tmax:
            tmax = tymax

        tzmin = (aabb.min_point[2] - ray.origin[2]) * inv_dir[2]
        tzmax = (aabb.max_point[2] - ray.origin[2]) * inv_dir[2]

        if inv_dir[2] < 0:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            result = 0

        else:
            result = 1

    return result