import taichi as ti
import math

from taichi.math import vec3, min, max, pi, sqrt, normalize, dot, cross, length, vec2, isinf

from accelerators.bvh import unoccluded, intersect_bvh
from base.bsdf import BXDF_SPECULAR
from base.frame import Frame, frame_from_z
from base.samplers import cosine_hemisphere_pdf, sample_cosine_hemisphere, sample_uniform_sphere, uniform_sphere_pdf, \
    sample_uniform_disk_concentric
from primitives.primitives import Triangle, Primitive
from primitives.ray import Ray
from utils.constants import INF
from utils.misc import length_squared, max_component


@ti.dataclass
class LightLiSample:
    L: ti.types.vector(3, ti.f32)
    wi: vec3
    pdf: ti.f32
    intr_p: vec3
    intr_n: vec3


@ti.dataclass
class LightLeSample:
    L: ti.types.vector(3, ti.f32)
    ray_origin: vec3
    ray_dir: vec3
    intr_p: vec3
    intr_n: vec3
    pdf_pos: ti.f32
    pdf_dir: ti.f32


@ti.dataclass
class SampledLight:
    light_idx: ti.i32
    pdf: ti.f32


@ti.dataclass
class DiffuseAreaLight:
    shape_idx: ti.i32
    Le: vec3
    two_sided: ti.i32

    @ti.func
    def L(self, p, n, w, scale=1):
        sampled_light = vec3(0.0)
        if dot(n, w) >= 0:
            sampled_light = self.Le * scale

        return sampled_light

    @ti.func
    def sample_Li(self, p, u, shape):
        ss = shape.sample_p(p, u)
        li_sample = LightLiSample()
        if ss.pdf != 0 and length_squared(ss.p - p) != 0:
            wi = normalize(ss.p - p)
            Le = self.L(ss.p, ss.n, -wi)
            li_sample.L = Le
            li_sample.wi = wi
            li_sample.pdf = ss.pdf
            li_sample.intr_p = ss.p
            li_sample.intr_n = ss.n
        return li_sample

    @ti.func
    def pdf_Li(self, isect, wi, shape):
        pdf = 0.0
        if isect.intersected:
            pdf = shape.PDF(isect.intersected_point, wi)
        return pdf

    @ti.func
    def sample_Le(self, u1, u2, shape):
        ss = shape.sample(u1)
        light_sample = LightLeSample()

        w = vec3([0.0])
        pdf_dir = 0.0
        if self.two_sided:
            if u2[0] < 0.5:
                u2[0] = ti.min(u2[0] * 2, 0.99999)
                w = sample_cosine_hemisphere(u2)
            else:
                u2[0] = ti.min((u2[0] - 0.5) * 2, 0.99999)
                w = sample_cosine_hemisphere(u2)
                w.z *= -1
            pdf_dir = cosine_hemisphere_pdf(abs(w.z)) / 2
        else:
            w = sample_cosine_hemisphere(u2)
            pdf_dir = cosine_hemisphere_pdf(w.z)

        if pdf_dir != 0:
            n_frame = frame_from_z(ss.n)
            w = n_frame.from_local(w)
            Le = self.L(ss.p, ss.n)
            light_sample.L = Le
            light_sample.ray_origin = ss.p
            light_sample.ray_dir = w
            light_sample.intr_p = ss.p
            light_sample.intr_n = ss.n
            light_sample.pdf_pos = ss.pdf
            light_sample.pdf_dir = pdf_dir

        return light_sample

    @ti.func
    def pdf_Le(self, n, w, shape):
        pdf_pos = shape.get_pdf()
        pdf_dir = (cosine_hemisphere_pdf(abs(dot(n, w))) / 2) if self.two_sided else cosine_hemisphere_pdf(
            dot(n, w))
        return pdf_pos, pdf_dir


@ti.dataclass
class UniformLightSampler:
    num_lights: ti.i32

    @ti.func
    def sample(self, u):
        sl = SampledLight()
        if self.num_lights != 0:
            light_idx = ti.min(ti.i32(u * self.num_lights), self.num_lights - 1)
            pdf = 1.0 / self.num_lights if self.num_lights > 0 else 0.0
            sl.light_idx = light_idx
            sl.pdf = pdf
        return sl

    @ti.func
    def pmf(self):
        return 1.0 / self.num_lights if self.num_lights > 0 else 0.0


@ti.func
def uniform_sample_one_light(isect, wo, bsdf, lights, light_sampler, primitives, bvh):
    s_l = light_sampler.sample(ti.random())
    light = lights[s_l.light_idx]
    u_light = vec2(ti.random(), ti.random())
    u_scattering = vec2(ti.random(), ti.random())
    L = estimate_direct(isect, wo, bsdf, u_scattering, lights, light, s_l.light_idx, u_light, primitives, bvh)
    return L / s_l.pdf

@ti.func
def estimate_direct(isect, wo, bsdf, u_scattering, lights, ls, light_idx, u_light, primitives, bvh):
    Ld = vec3(0.0)

    # Sample light source
    li_sample = ls.sample_Li(isect.intersected_point, u_light, primitives[ls.shape_idx].triangle)

    if li_sample.pdf > 0 and not is_black(li_sample.L):
        # Evaluate BSDF for light sampling strategy
        f = bsdf.f(wo, li_sample.wi) * ti.abs(dot(li_sample.wi, isect.normal))

        if not is_black(f):
            # Check visibility
            if unoccluded(isect.intersected_point, isect.normal, li_sample.intr_p, primitives, bvh, 1e-4):
                Ld += f * li_sample.L / li_sample.pdf

    return Ld


@ti.func
def is_black(v):
    return ti.abs(v[0]) < 1e-6 and ti.abs(v[1]) < 1e-6 and ti.abs(v[2]) < 1e-6


@ti.func
def is_delta_light(light):
    # Implement this based on your light types
    return False  # For now, assume no delta lights


@ti.func
def power_heuristic(nf, f_pdf, ng, g_pdf):
    f = nf * f_pdf
    g = ng * g_pdf
    f_sq = f * f
    g_sq = g * g
    to_return = 1.0
    if not isinf(f_sq):
        to_return = f_sq / (f_sq + g_sq)
    return to_return

