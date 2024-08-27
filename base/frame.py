import taichi as ti
from taichi.math import vec3, dot, cross, normalize, length


@ti.dataclass
class Frame:
    x: ti.types.vector(3, ti.f32)
    y: ti.types.vector(3, ti.f32)
    z: ti.types.vector(3, ti.f32)

    @ti.func
    def to_local(self, v):
        return vec3(dot(v, self.x), dot(v, self.y), dot(v, self.z))

    @ti.func
    def from_local(self, v):
        return self.x * v[0] + self.y * v[1] + self.z * v[2]


@ti.func
def create_frame(x, y, z):
    assert ti.abs(length(x) - 1.0) < 1e-4, f"x is not normalized: {length(x)}"
    assert ti.abs(length(y) - 1.0) < 1e-4, f"y is not normalized: {length(y)}"
    assert ti.abs(length(z) - 1.0) < 1e-4, f"z is not normalized: {length(z)}"
    assert ti.abs(dot(x, y)) < 1e-4, "x and y are not orthogonal"
    assert ti.abs(dot(y, z)) < 1e-4, "y and z are not orthogonal"
    assert ti.abs(dot(z, x)) < 1e-4, "z and x are not orthogonal"
    return Frame(x=x, y=y, z=z)


@ti.func
def copysign(x, y):
    return ti.abs(x) if y >= 0 else -ti.abs(x)


@ti.func
def coordinate_system(v):
    sign = copysign(1.0, v.z)
    a = -1.0 / (sign + v.z)
    b = v.x * v.y * a
    v2 = vec3(1.0 + sign * v.x * v.x * a, sign * b, -sign * v.x)
    v3 = vec3(b, sign + v.y * v.y * a, -v.y)
    return v2, v3


@ti.func
def frame_from_xz(x, z):
    x = normalize(x)
    z = normalize(z)
    x = normalize(x - dot(x, z) * z)  # Re-orthogonalize x relative to z
    y = cross(z, x)
    return create_frame(x, y, z)


@ti.func
def frame_from_xy(x, y):
    x = normalize(x)
    y = normalize(y)
    z = cross(x, y)
    return create_frame(x, y, z)


@ti.func
def frame_from_z(z):
    z = normalize(z)
    x, y = coordinate_system(z)
    return create_frame(x, y, z)


@ti.func
def frame_from_x(x):
    x = normalize(x)
    y, z = coordinate_system(x)
    return create_frame(x, y, z)


@ti.func
def frame_from_y(y):
    y = normalize(y)
    z, x = coordinate_system(y)
    return create_frame(x, y, z)
