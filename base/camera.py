import taichi as ti
from taichi.math import vec3, normalize, dot, cross, radians, tan, sqrt, pi, cos, sin, vec2
from base.frame import Frame, frame_from_z
from primitives.ray import Ray


@ti.func
def sample_uniform_disk_concentric(u, v):
    # Map uniform random numbers to [-1,1]^2
    u = 2 * u - 1
    v = 2 * v - 1

    to_return = ti.Vector([0.0, 0.0])

    # Handle degeneracy at the origin
    if not (u == 0 and v == 0):

        # Apply concentric mapping to point
        theta = 0.0
        r = 0.0
        if ti.abs(u) > ti.abs(v):
            r = u
            theta = (pi / 4) * (v / u)
        else:
            r = v
            theta = (pi / 2) - (pi / 4) * (u / v)

        to_return = ti.Vector([r * cos(theta), r * sin(theta)])

    return to_return


@ti.dataclass
class PerspectiveCamera:
    width: ti.i32
    height: ti.i32
    position: vec3
    frame: Frame
    fov: ti.f32
    aspect_ratio: ti.f32
    lens_radius: ti.f32
    focal_distance: ti.f32
    screen_dx: ti.f32
    screen_dy: ti.f32
    dx_camera: vec3
    dy_camera: vec3

    @ti.func
    def camera_from_raster(self, p_film):
        # Convert film coordinates to camera space
        p_camera = vec3(
            (p_film.x - 0.5) * self.screen_dx,
            (p_film.y - 0.5) * self.screen_dy,
            1.0
        )
        return p_camera

    @ti.func
    def generate_ray(self, s: ti.f32, t: ti.f32):
        # Compute raster space coordinates
        p_film = vec3(s, t, 0.0)

        # Convert raster space coordinates to camera space
        p_camera = vec3(
            (p_film.x - 0.5) * self.screen_dx,
            (p_film.y - 0.5) * self.screen_dy,
            1.0
        )

        # Generate the initial ray in camera space
        ray_dir = normalize(p_camera)
        ray_origin = vec3(0.0, 0.0, 0.0)

        # Modify ray for depth of field
        if self.lens_radius > 0:
            lens_u, lens_v = ti.random(), ti.random()
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)

            ft = self.focal_distance / ray_dir.z
            p_focus = ray_origin + ft * ray_dir

            ray_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ray_dir = normalize(p_focus - ray_origin)

        # Transform the ray from camera space to world space
        world_ray_origin = self.position + self.frame.from_local(ray_origin)
        world_ray_dir = self.frame.from_local(ray_dir)

        return world_ray_origin, normalize(world_ray_dir)

    @ti.func
    def generate_ray_differential(self, s: ti.f32, t: ti.f32):
        # Compute raster space coordinates
        p_film = vec3(s, t, 0.0)

        # Convert raster space coordinates to camera space
        p_camera = vec3(
            (p_film.x - 0.5) * self.screen_dx,
            (p_film.y - 0.5) * self.screen_dy,
            1.0
        )

        # Generate the initial ray in camera space
        ray_dir = normalize(p_camera)
        ray_origin = vec3(0.0, 0.0, 0.0)

        # Compute ray differentials
        rx_origin = ray_origin
        ry_origin = ray_origin
        rx_direction = normalize(p_camera + self.dx_camera)
        ry_direction = normalize(p_camera + self.dy_camera)

        # Modify ray for depth of field
        if self.lens_radius > 0:
            lens_u, lens_v = ti.random(), ti.random()
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)

            ft = self.focal_distance / ray_dir.z
            p_focus = ray_origin + ft * ray_dir

            ray_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ray_dir = normalize(p_focus - ray_origin)

            dx = normalize(p_camera + self.dx_camera)
            ft_x = self.focal_distance / dx.z
            p_focus_x = ft_x * dx
            rx_origin = vec3(p_lens.x, p_lens.y, 0.0)
            rx_direction = normalize(p_focus_x - rx_origin)

            dy = normalize(p_camera + self.dy_camera)
            ft_y = self.focal_distance / dy.z
            p_focus_y = ft_y * dy
            ry_origin = vec3(p_lens.x, p_lens.y, 0.0)
            ry_direction = normalize(p_focus_y - ry_origin)

        # Transform the rays from camera space to world space
        world_ray_origin = self.position + self.frame.from_local(ray_origin)
        world_ray_dir = self.frame.from_local(ray_dir)
        world_rx_origin = self.position + self.frame.from_local(rx_origin)
        world_rx_dir = self.frame.from_local(rx_direction)
        world_ry_origin = self.position + self.frame.from_local(ry_origin)
        world_ry_dir = self.frame.from_local(ry_direction)

        return world_ray_origin, normalize(world_ray_dir), \
            world_rx_origin, normalize(world_rx_dir), \
            world_ry_origin, normalize(world_ry_dir)