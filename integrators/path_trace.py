import taichi as ti
from taichi.math import vec3, vec2, dot, max, isinf

from accelerators.bvh import intersect_bvh, unoccluded
from base.bsdf import BXDF_SPECULAR, BXDF_REFLECTION, BXDF_TRANSMISSION, BXDF_NONE
from base.lights import uniform_sample_one_light, is_black
from base.samplers import sample_uniform_sphere, uniform_sphere_pdf, sample_uniform_hemisphere, uniform_hemisphere_pdf
from primitives.ray import Ray, spawn_ray
from utils.constants import INF


@ti.func
def path_trace(ray, primitives, bvh, lights, light_sampler, sample_lights=1, sample_bsdf=1, max_depth=3):
    L = vec3(0.0)
    beta = vec3(1.0)
    specular_bounce = 1
    depth = 0  # Depth of the recursion
    t_max = INF
    t_min = 0.0

    while 1:
        if depth >= max_depth:
            break

        # Intersect the ray with the scene
        isect = intersect_bvh(ray, primitives, bvh, t_min, t_max)

        if not isect.intersected:
            # TODO: Environment Light
            break

        # Accumulate emission (direct light) from the intersected object
        L += beta * isect.nearest_object.material.emission

        # Check if we've reached the maximum recursion depth
        if depth > 4:
            r_r = isect.nearest_object.material.diffuse.max()
            if ti.random() >= r_r:
                break
            beta = beta/r_r

        depth += 1

        # Get the BSDF of the intersected object
        bsdf = isect.nearest_object.bsdf
        bsdf.init_frame(isect.normal, isect.dpdu)

        wo = -ray.direction

        # direct lighting contribution
        if sample_lights:
            s_l = light_sampler.sample(ti.random())
            sampled_li = lights[s_l.light_idx]
            u_light = vec2(ti.random(), ti.random())
            l_shape = primitives[sampled_li.shape_idx].triangle
            ls = sampled_li.sample_Li(isect.intersected_point, u_light, l_shape)

            if not is_black(ls.L) and ls.pdf > 0:
                wi = ls.wi
                f = bsdf.f(wo, wi) * ti.abs(dot(wi, isect.normal))
                if not is_black(f) and unoccluded(isect.intersected_point, isect.normal, ls.intr_p, primitives, bvh, 1e-4):
                    L += beta * (f * ls.L / ls.pdf) / s_l.pdf

        # BSDF sampling
        u = ti.random()
        u2 = vec2(ti.random(), ti.random())
        bs = bsdf.sample_f(wo, u, u2)

        if is_black(bs.f) or bs.pdf == 0:
            break

        beta *= bs.f * ti.abs(dot(bs.wi, isect.normal)) / bs.pdf
        specular_bounce = (bs.flags & BXDF_SPECULAR != 0)
        wi = bs.wi
        # t_min = 1e-4  # Avoid self-intersection by moving the origin slightly
        # ray = Ray(isect.intersected_point, wi)
        ray = spawn_ray(isect.intersected_point, isect.normal, wi)

    return L


