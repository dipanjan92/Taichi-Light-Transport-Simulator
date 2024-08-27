import taichi as ti
from taichi.math import vec3, vec2, dot, max, isinf

from accelerators.bvh import intersect_bvh, unoccluded
from base.bsdf import BXDF_SPECULAR, BXDF_REFLECTION, BXDF_TRANSMISSION, BXDF_NONE, BXDF_ALL
from base.lights import uniform_sample_one_light, is_black, power_heuristic
from primitives.intersects import Intersection
from primitives.ray import Ray, spawn_ray, offset_ray_origin
from utils.constants import INF

'''
Implementation of the "Optimal multiple importance sampling" paper (Work In Progress).

Kondapaneni, I., Vévoda, P., Grittmann, P., Skřivan, T., Slusallek, P. and Křivánek, J., 2019. Optimal multiple importance sampling. ACM Transactions on Graphics (TOG), 38(4), pp.1-14.
DOI: https://doi.org/10.1145/3306346.3323009.
'''

# OPTIMAL_MODE
ALPHA_SUM = 0
ALPHA_SUM_L2 = 1
ALPHA_SUM_INDB = 2
FULL_WEIGHTS = 3
PROGRESSIVE = 4
ALPHA_SUM_PROG = 5

# MIS_TYPE
MIS_BALANCE = 0
MIS_POWER = 1
MIS_OPTIMAL = 2

# UPDATE_TYPE
UPDATE_MATRIX = 0
UPDATE_RIGHT_SIDE = 1
UPDATE_BOTH = 2


@ti.data_oriented
class AlphaEstimator:
    def __init__(self, techniques=2, update_step=1, skip_tech=-1, training=0, mis_type=2, optimal_mode=0,
                 update_type=0):
        self.techniques = techniques
        self.update_step = update_step
        self.skip_tech = skip_tech

        self.pdfs = ti.field(dtype=ti.f32, shape=(4,))
        self.contribs = ti.Vector.field(3, ti.f32, shape=(2,))
        self.sample_counts = ti.field(dtype=ti.i32, shape=(2,))
        self.weights = Weights.field(shape=(techniques,))

        self.training = training
        self.mis_type = mis_type
        self.optimal_mode = optimal_mode
        self.update_type = update_type

        # Initialize matrices
        self.A = ti.Matrix.field(techniques, techniques, dtype=ti.f32, shape=())
        self.b = ti.Matrix.field(techniques, 3, dtype=ti.f32, shape=())
        self.Aitc = ti.Matrix.field(techniques, 1, dtype=ti.f32, shape=())
        self.bitc = ti.Vector.field(3, dtype=ti.f32, shape=())

        # Initialize other variables
        self.alphas = ti.Vector.field(3, ti.f32, shape=(techniques + 1,))
        self.updates = ti.field(ti.i32, shape=())
        self.updates[None] = 0

        self.aa = ti.Matrix.field(self.techniques + 1, self.techniques + 1, dtype=ti.f32, shape=())
        self.bb = ti.Matrix.field(self.techniques + 1, 3, dtype=ti.f32, shape=())

    @ti.func
    def update_estimates(self):

        early_return = 0

        for i in range(self.techniques):
            if self.pdfs[i * self.techniques + i] == 0:
                early_return = 1

        if not early_return:
            for i in range(self.techniques):
                denom = 0.0
                denom_sq = 0.0
                for j in range(self.techniques):
                    denom += self.sample_counts[j] * self.pdfs[i * self.techniques + j]
                    denom_sq += (self.sample_counts[j] * self.pdfs[i * self.techniques + j]) ** 2

                if self.mis_type == MIS_BALANCE:
                    if self.optimal_mode == ALPHA_SUM_L2:
                        self.weights[i].weight_1 = self.sample_counts[i] / denom
                    else:
                        self.weights[i].weight_1 = self.sample_counts[i] / (denom * denom)
                        self.weights[i].weight_2 = 1 / denom
                else:
                    if self.optimal_mode == ALPHA_SUM_L2:
                        self.weights[i].weight_1 = self.sample_counts[i] ** 2 * self.pdfs[
                            i * self.techniques + i] / denom_sq
                    else:
                        self.weights[i].weight_1 = self.sample_counts[i] ** 2 * self.pdfs[
                            i * self.techniques + i] / (denom_sq * denom)

            for i in range(self.techniques):
                if self.update_type == UPDATE_BOTH or self.update_type == UPDATE_MATRIX:
                    for j in range(self.techniques):
                        for k in range(self.techniques):
                            if self.techniques == 1:
                                self.A[None][0, 0] += self.weights[k].weight_1 * self.pdfs[
                                    k * self.techniques + i] * self.pdfs[
                                                          k * self.techniques + j]
                            else:
                                self.A[None][i, j] += self.weights[k].weight_1 * self.pdfs[
                                    k * self.techniques + i] * self.pdfs[
                                                          k * self.techniques + j]

                if self.update_type == UPDATE_BOTH or self.update_type == UPDATE_RIGHT_SIDE:
                    for c in range(3):
                        for k in range(self.techniques):
                            self.b[None][i, c] += self.weights[k].weight_1 * self.contribs[k][c] * \
                                                  self.pdfs[k * self.techniques + i]

            if self.skip_tech != -1:
                for i in range(self.techniques):
                    if self.update_type == UPDATE_BOTH or self.update_type == UPDATE_MATRIX:
                        for k in range(self.techniques):
                            self.Aitc[None][i, 0] += self.weights[k].weight_2 * self.pdfs[
                                k * self.techniques + i]

                if self.update_type == UPDATE_BOTH or self.update_type == UPDATE_RIGHT_SIDE:
                    for c in range(3):
                        for k in range(self.techniques):
                            self.bitc[None][c] += self.weights[k].weight_2 * self.contribs[k][c]

            self.updates[None] += 1
            if self.updates[None] % self.update_step == 0:
                self.reset_alphas()

    @ti.func
    def compute_alphas(self, debug=0):
        if debug:
            print("Alpha computation")
            print("A:\n", self.A[None])
            print("b:\n", self.b[None])

        if self.updates[None] == 0:
            for i in range(self.techniques + 1):
                self.alphas[i] = vec3(0.0)
        else:
            if self.skip_tech == -1:
                alphasM = self.solve_linear_system(self.A[None], self.b[None])
                for i in range(self.techniques):
                    for c in range(3):
                        self.alphas[i][c] = alphasM[i, c]

                if debug:
                    relative_error = (self.A[None] @ alphasM - self.b[None]).norm() / self.b[None].norm()
                    print("relative error:", relative_error)
            else:
                self.aa[None][:self.techniques, :self.techniques] = self.A[None] * (self.techniques * self.techniques)
                for i in range(self.techniques):
                    self.aa[None][i, self.techniques] = self.Aitc[None][i, 0] * self.techniques  # Corrected
                    self.aa[None][self.techniques, i] = self.Aitc[None][i, 0] * self.techniques  # Corrected
                self.aa[None][self.techniques, self.techniques] = self.updates[None] * self.techniques
                self.bb[None][:self.techniques, :] = self.b[None] * (self.techniques * self.techniques)
                self.bb[None][self.techniques, :] = self.bitc[None] * self.techniques  # Corrected

                # Resize and skip row/column manually
                for i in range(self.skip_tech, self.techniques):
                    for j in range(self.techniques + 1):
                        self.aa[None][i, j] = self.aa[None][i + 1, j]

                for j in range(self.skip_tech, self.techniques):
                    for i in range(self.techniques + 1):
                        self.aa[None][i, j] = self.aa[None][i, j + 1]

                for i in range(self.skip_tech, self.techniques):
                    for j in range(3):
                        self.bb[None][i, j] = self.bb[None][i + 1, j]

                alphasM = self.solve_linear_system(self.aa[None], self.bb[None])

                for i in range(self.techniques + 1):
                    if i == self.skip_tech:
                        self.alphas[i] = ti.Vector([0.0, 0.0, 0.0])
                    else:
                        for c in range(3):
                            self.alphas[i][c] = alphasM[i, c]

        if debug:
            for i in range(self.techniques):
                print(f"alpha[{i}]: ", self.alphas[i])

    @ti.func
    def compute_alpha_sum(self, debug=0):
        if self.alphas.shape[0] == 0:
            self.compute_alphas(debug)

        sum_alpha = vec3(0.0)
        for i in range(self.techniques + 1):
            sum_alpha += self.alphas[i]
        return sum_alpha

    @ti.func
    def get_alpha(self, i, debug=0):
        if self.alphas.shape[0] == 0:
            self.compute_alphas(debug)
        return self.alphas[i]

    @ti.func
    def solve_linear_system(self, A, b):
        # return A.inverse() @ b
        U, S, V = ti.svd(A)

        # Inverting the diagonal matrix S
        S_inv = ti.Matrix.zero(ti.f32, A.n, A.n)
        for i in ti.static(range(A.n)):
            if S[i, i] != 0:
                S_inv[i, i] = 1.0 / S[i, i]

        # Computing the pseudo-inverse of A using SVD components
        A_inv = V @ S_inv @ U.transpose()

        # Solving for x in A * x = b
        x = A_inv @ b

        return x

    @ti.func
    def increment_updates(self):
        self.updates[None] += 1
        if self.updates[None] % self.update_step == 0:
            self.alphas.fill(0.0)

    @ti.func
    def reset_alphas(self):
        for i in range(self.techniques + 1):
            self.alphas[i] = vec3(0.0)

    @ti.func
    def optimal_weight(self, debug=0):
        ts = self.contribs.shape[0]
        L = vec3(0.0)

        if debug:
            print("Weighting")
            for i in range(ts):
                print(f"contribs[{i}]: {self.contribs[i]}")
            for i in range(self.pdfs.shape[0]):
                print(f"pdfs[{i}]: {self.pdfs[i]}")

        if ts == 1:
            if not is_black(self.contribs[0]):
                L += self.contribs[0] / self.pdfs[0]
        else:
            for i in range(ts):
                if self.mis_type == MIS_OPTIMAL:
                    denom = 0.0
                    alpha_comb = vec3(0.0)
                    for j in range(ts):
                        denom += self.sample_counts[j] * self.pdfs[i * ts + j]
                        alpha_comb += self.get_alpha(j, debug) * self.pdfs[i * ts + j]

                    if self.pdfs[i * ts + i] > 0.0:
                        wc = self.sample_counts[i] / denom * (self.contribs[i] - alpha_comb) + self.get_alpha(i, debug)
                        L += wc

                        if debug:
                            print(f"Weighted contrib: {wc}")

                elif not is_black(self.contribs[i]):
                    if self.mis_type == MIS_BALANCE:
                        denom = 0.0
                        for j in range(ts):
                            denom += self.sample_counts[j] * self.pdfs[i * ts + j]

                        L += self.sample_counts[i] / denom * self.contribs[i]
                    else:  # mis_type == MIS_POWER
                        denom = 0.0
                        for j in range(ts):
                            denom += self.sample_counts[j] * self.pdfs[i * ts + j] * self.sample_counts[j] * self.pdfs[
                                i * ts + j]

                        L += self.sample_counts[i] / denom * self.contribs[i] * self.sample_counts[i] * self.pdfs[
                            i * ts + i]

        if debug:
            print(f"Total weighted contrib: {L}")

        return L

    @ti.func
    def estimate_li_or_alpha(self, ray, primitives, bvh, isect, bsdf, light_sampler, lights, ctx_p):
        Ld = vec3(0.0)

        # Uniform Light Sampling
        f_xl = vec3(0.0)

        pl_xl = 0.0
        pb_xl = 0.0

        # Sample light source
        s_l = light_sampler.sample(ti.random())
        sampled_li = lights[s_l.light_idx]
        l_shape = primitives[sampled_li.shape_idx].triangle
        u_light = vec2(ti.random(), ti.random())
        ls = sampled_li.sample_Li(ctx_p, u_light, l_shape)

        pl_xl = ls.pdf

        if pl_xl > 0:
            pb_xl = bsdf.pdf(-ray.direction, ls.wi)
            if not is_black(ls.L):
                # print(ls.L)
                wi = ls.wi
                f = bsdf.f(-ray.direction, wi, ) * ti.abs(dot(wi, isect.normal))
                if not is_black(f) and unoccluded(ctx_p, isect.normal, ls.intr_p, primitives, bvh,
                                                  1e-4):
                    f_xl += f * ls.L

        # BSDF Sampling
        f_xb = vec3(0.0)

        pl_xb = 0.0
        pb_xb = 0.0

        u = ti.random()
        u2 = vec2(ti.random(), ti.random())
        bs = bsdf.sample_f(-ray.direction, u, u2)
        bs.f *= ti.abs(dot(bs.wi, isect.normal))

        pb_xb = bs.pdf

        if pb_xb > 0:
            pl_xb = sampled_li.pdf_Li(isect, bs.wi, l_shape)

        if not is_black(bs.f):
            new_ray = spawn_ray(isect.intersected_point, isect.normal, bs.wi)
            Li = vec3(0.0)
            li_isect = intersect_bvh(new_ray, primitives, bvh, 0, INF)

            if li_isect.intersected:
                if li_isect.nearest_object.is_light:
                    Li += li_isect.Le(-bs.wi)
            # else:
            #     Li += sampled_li.L(isect.intersected_point, isect.normal, new_ray.direction)

            f_xb = bs.f * Li

        # combine
        # first uniform light samples
        self.pdfs[0] = pl_xl
        self.pdfs[1] = pb_xl
        self.contribs[0] = f_xl
        self.sample_counts[0] = 1
        # second bsdf samples
        self.pdfs[2] = pl_xb
        self.pdfs[3] = pb_xb
        self.contribs[1] = f_xb
        self.sample_counts[1] = 1

        for i in range(4):
            self.pdfs[i] *= s_l.pdf  # multiply with light select pdf

        if self.mis_type == MIS_OPTIMAL and self.optimal_mode == PROGRESSIVE:
            Ld += self.optimal_weight()
            self.update_estimates()
        else:
            if self.mis_type != MIS_OPTIMAL or self.optimal_mode == FULL_WEIGHTS and not self.training:
                Ld += self.optimal_weight()
            else:
                self.update_estimates()

        return Ld

    @ti.func
    def sample_Ld(self, ray, primitives, bvh, isect, bsdf, light_sampler, lights):

        L = vec3(0.0)
        f_lu = vec3(0.0)

        # Context setup
        ctx_p = vec3(0.0)
        if bsdf.flags() & BXDF_REFLECTION != 0 and bsdf.flags() & BXDF_TRANSMISSION == 0:
            ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, -ray.direction)
        elif bsdf.flags() & BXDF_TRANSMISSION != 0 and bsdf.flags() & BXDF_REFLECTION == 0:
            ctx_p = offset_ray_origin(isect.intersected_point, isect.normal, ray.direction)

        if bsdf.flags() & BXDF_SPECULAR == 0:
            f_lu += self.estimate_li_or_alpha(ray, primitives, bvh, isect, bsdf, light_sampler, lights, ctx_p)

        # combine
        L += f_lu

        return L


@ti.dataclass
class Weights:
    weight_1: ti.f32
    weight_2: ti.f32


@ti.func
def trace_optimal(ray, primitives, bvh, lights, light_sampler, alpha_estimator, sample_lights=1, sample_bsdf=1,
                  max_depth=3):
    L = vec3(0.0)
    beta = vec3(1.0)  # Path throughput
    depth = 0  # Depth of the recursion
    eta_scale = 1.0  # IoR scale
    specular_bounce = 0
    any_non_specular_bounces = 0
    regularize = 0
    p_b = 1.0  # Probability density of the previous bounce
    prev_intr_ctx = Intersection()  # Previous interaction context for MIS
    t_max = INF
    t_min = 0.0

    while depth < max_depth:
        # Intersect scene
        isect = intersect_bvh(ray, primitives, bvh, t_min, t_max)
        if not isect.intersected:
            break

        # Check if emissive
        Le = vec3(0.0)
        Le += isect.Le(-ray.direction)
        if not is_black(Le):
            if depth == 0 or specular_bounce:
                L += beta * Le
            else:
                # Compute MIS weight for area light
                if isect.nearest_object.is_light:
                    # should be true always
                    area_light = lights[isect.nearest_object.light_idx]
                    p_l = light_sampler.pmf() * area_light.pdf_Li(prev_intr_ctx, ray.direction,
                                                                  isect.nearest_object.triangle)
                    w_l = power_heuristic(1, p_b, 1, p_l)
                    L += beta * w_l * Le

        # Get BSDF
        bsdf = isect.get_bsdf()

        if bsdf.flags() == BXDF_NONE:
            continue

        # End path if maximum depth is reached
        if depth == max_depth:
            break
        depth += 1

        # Sample direct illumination uniformly
        Ld_light = alpha_estimator.sample_Ld(ray, primitives, bvh, isect, bsdf, light_sampler, lights)

        # print(Ld_light)

        L += beta * Ld_light

        # Sample BSDF
        wo = -ray.direction

        u = ti.random()
        u2 = vec2(ti.random(), ti.random())
        bs = bsdf.sample_f(wo, u, u2)

        if is_black(bs.f) or bs.pdf == 0:
            break

        # surface scattering
        beta *= bs.f * ti.abs(dot(bs.wi, isect.normal)) / bs.pdf
        p_b = bs.pdf  # this would work for everything but LayeredBXDF (not implemented)
        specular_bounce = (bs.flags & BXDF_SPECULAR != 0)
        any_non_specular_bounces |= bs.flags & BXDF_SPECULAR == 0
        if bs.flags & BXDF_TRANSMISSION != 0:
            eta_scale *= bs.eta ** 2

        prev_intr_ctx = isect

        # Spawn the next ray
        # t_min = 1e-4
        # ray = Ray(isect.intersected_point, bs.wi)

        ray = spawn_ray(isect.intersected_point, isect.normal, bs.wi)

        # Russian roulette
        rr_beta = beta * eta_scale
        if rr_beta.max() < 1.0 and depth > 2:
            q = max(0.0, 1.0 - rr_beta.max())
            if ti.random() < q:
                break
            beta /= (1.0 - q)

    return L
