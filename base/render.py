import sys, time

import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
from taichi.math import vec3

from accelerators.bvh import intersect_bvh
from base.lights import UniformLightSampler
from integrators.mis_pt import trace_mis
from integrators.path_trace import path_trace
from primitives.ray import Ray


@ti.kernel
def render(scene: ti.template(), image: ti.template(), lights: ti.template(), camera: ti.template(), primitives: ti.template(), bvh: ti.template()):
    light_sampler = UniformLightSampler(lights.shape[0])

    height = camera.height
    width = camera.width
    samples_per_pixel = scene.spp

    if scene.integrator == 0:
        ti.loop_config(parallelize=4, block_dim=16)
        for j, i in ti.ndrange(height, width):
            L = vec3(0.0)

            for k in range(samples_per_pixel):
                u = (i + ti.random(ti.f32)) / width
                v = (j + ti.random(ti.f32)) / height

                ray_origin, ray_direction = camera.generate_ray(u, v)
                ray = Ray(ray_origin, ray_direction)
                L += path_trace(ray, primitives, bvh, lights, light_sampler, sample_lights=scene.sample_lights,
                                sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)

            image[j, i] = L / samples_per_pixel
    elif scene.integrator == 1:
        ti.loop_config(parallelize=4, block_dim=16)
        for j, i in ti.ndrange(height, width):
            L = vec3(0.0)

            for k in range(samples_per_pixel):
                u = (i + ti.random(ti.f32)) / width
                v = (j + ti.random(ti.f32)) / height

                ray_origin, ray_direction = camera.generate_ray(u, v)
                ray = Ray(ray_origin, ray_direction)
                L += trace_mis(ray, primitives, bvh, lights, light_sampler, sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)

            image[j, i] = L / samples_per_pixel


