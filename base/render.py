import sys, time

import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
from taichi.math import vec3

from accelerators.bvh import intersect_bvh
from base.lights import UniformLightSampler
from integrators.mis_pt import trace_mis
from integrators.opt_mis_pt import *
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


@ti.kernel
def train_alpha_estimator(scene: ti.template(), lights: ti.template(), camera: ti.template(), primitives: ti.template(), bvh: ti.template(), alpha_estimators: ti.template()):
    light_sampler = UniformLightSampler(lights.shape[0])

    height = camera.height
    width = camera.width
    train_samples_per_pixel = 64  # Set the number of samples for training

    # Loop over each pixel to perform training
    for j, i in ti.ndrange(height, width):
        for k in range(train_samples_per_pixel):
            u = (i + ti.random(ti.f32)) / width
            v = (j + ti.random(ti.f32)) / height

            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(ray_origin, ray_direction)
            L_sample = trace_optimal(ray, primitives, bvh, lights, light_sampler, alpha_estimators, sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)

            # # Update alpha estimator without resetting
            # alpha_estimators.update_estimates()

@ti.kernel
def optimal_render(scene: ti.template(), image: ti.template(), lights: ti.template(), camera: ti.template(), primitives: ti.template(), bvh: ti.template(), alpha_estimator: ti.template()):
    light_sampler = UniformLightSampler(lights.shape[0])

    height = camera.height
    width = camera.width
    samples_per_pixel = scene.spp

    mis_type = MIS_OPTIMAL
    optimal_mode = PROGRESSIVE
    update_type = UPDATE_BOTH


    if scene.integrator == 1:

        ti.loop_config(parallelize=4, block_dim=16)
        for j, i in ti.ndrange(height, width):
            L = vec3(0.0)

            for k in range(samples_per_pixel):
                u = (i + ti.random(ti.f32)) / width
                v = (j + ti.random(ti.f32)) / height

                ray_origin, ray_direction = camera.generate_ray(u, v)
                ray = Ray(ray_origin, ray_direction)
                L_sample = trace_optimal(ray, primitives, bvh, lights, light_sampler, alpha_estimator, sample_lights=scene.sample_lights, sample_bsdf=scene.sample_bsdf, max_depth=scene.max_depth)

                L += L_sample

                if mis_type == 2 and optimal_mode != 3 and optimal_mode != 4:
                    alpha_sum = vec3(0.0)
                    for l in range(alpha_estimator.alphas.shape[0]):
                        alpha_sum += alpha_estimator.compute_alpha_sum(0)

                    L += alpha_sum

            image[j, i] = L / samples_per_pixel
