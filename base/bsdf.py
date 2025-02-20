import taichi as ti

from base.frame import Frame, frame_from_z, frame_from_xz
from base.samplers import sample_uniform_disk_polar, sample_cosine_hemisphere, cosine_hemisphere_pdf
from utils.constants import inv_pi
from taichi.math import pi, vec3, sqrt, normalize, dot, cross, sin, cos, isinf, clamp, length

from utils.misc import same_hemisphere, abs_cos_theta, cos_theta, face_forward, fr_dielectric, \
    refract, reflect, fr_complex, tan2_theta, cos2_theta, phi, lerp, length_squared, max_component, safe_sqrt, transmit, \
    fresnel

# Modes

RADIANCE = 1
IMPORTANCE = 2  # TODO: MIS

# BxDF Flags
BXDF_NONE = 0
BXDF_REFLECTION = 1 << 0
BXDF_TRANSMISSION = 1 << 1
BXDF_DIFFUSE = 1 << 2
BXDF_GLOSSY = 1 << 3
BXDF_SPECULAR = 1 << 4

# Composite BxDF Flags
BXDF_DIFFUSE_REFLECTION = BXDF_DIFFUSE | BXDF_REFLECTION
BXDF_DIFFUSE_TRANSMISSION = BXDF_DIFFUSE | BXDF_TRANSMISSION
BXDF_GLOSSY_REFLECTION = BXDF_GLOSSY | BXDF_REFLECTION
BXDF_GLOSSY_TRANSMISSION = BXDF_GLOSSY | BXDF_TRANSMISSION
BXDF_SPECULAR_REFLECTION = BXDF_SPECULAR | BXDF_REFLECTION
BXDF_SPECULAR_TRANSMISSION = BXDF_SPECULAR | BXDF_TRANSMISSION
BXDF_ALL = BXDF_DIFFUSE | BXDF_GLOSSY | BXDF_SPECULAR | BXDF_REFLECTION | BXDF_TRANSMISSION


@ti.dataclass
class BSDFSample:
    f: vec3
    wi: vec3
    pdf: ti.f32
    flags: ti.i32
    eta: ti.f32


@ti.dataclass
class TrowbridgeReitzDistribution:
    alpha_x: ti.f32
    alpha_y: ti.f32

    @ti.func
    def initialize(self, ax, ay):
        self.alpha_x = ax
        self.alpha_y = ay
        if not self.effectively_smooth():
            self.alpha_x = max(self.alpha_x, 1e-4)
            self.alpha_y = max(self.alpha_y, 1e-4)

    @ti.func
    def effectively_smooth(self):
        return max(self.alpha_x, self.alpha_y) < 1e-3

    @ti.func
    def d1(self, wm):
        tan2Theta = tan2_theta(wm)
        cos4Theta = cos2_theta(wm) ** 2
        result = 0.0
        if not isinf(tan2Theta) and cos4Theta >= 1e-16:
            e = tan2Theta * ((cos(phi(wm)) / self.alpha_x) ** 2 +
                             (sin(phi(wm)) / self.alpha_y) ** 2)
            result = 1 / (pi * self.alpha_x * self.alpha_y * cos4Theta * (1 + e) ** 2)
        return result

    @ti.func
    def g1(self, w):
        return 1 / (1 + self.Lambda(w))

    @ti.func
    def Lambda(self, w):
        tan2Theta = tan2_theta(w)
        result = 0.0
        if not isinf(tan2Theta):
            alpha2 = (cos(phi(w)) * self.alpha_x) ** 2 + (sin(phi(w)) * self.alpha_y) ** 2
            result = (sqrt(1 + alpha2 * tan2Theta) - 1) / 2
        return result

    @ti.func
    def g2(self, wo, wi):
        return 1 / (1 + self.Lambda(wo) + self.Lambda(wi))

    @ti.func
    def d2(self, w, wm):
        return self.g1(w) / abs_cos_theta(w) * self.d1(wm) * abs(dot(w, wm))

    @ti.func
    def pdf(self, w, wm):
        return self.d2(w, wm)

    @ti.func
    def sample_wm(self, w, u):
        wh = normalize(vec3(self.alpha_x * w[0], self.alpha_y * w[1], w[2]))
        if wh[2] < 0:
            wh = -wh

        t1 = vec3(1, 0, 0) if wh[2] > 0.99999 else normalize(cross(vec3(0, 0, 1), wh))

        t2 = cross(wh, t1)

        p = sample_uniform_disk_polar(u)
        h = sqrt(1 - p[0] ** 2)
        p[1] = lerp((1 + wh[2]) / 2, h, p[1])

        pz = sqrt(max(0.0, 1 - length_squared(p)))
        nh = p[0] * t1 + p[1] * t2 + pz * wh

        return normalize(vec3(self.alpha_x * nh[0],
                              self.alpha_y * nh[1],
                              max(1e-6, nh[2])))

    @ti.func
    def roughness_to_alpha(self, roughness):
        return max(0.0001, roughness * roughness)

    @ti.func
    def regularize(self):
        if self.alpha_x < 0.3:
            self.alpha_x = clamp(2 * self.alpha_x, 0.1, 0.3)
        if self.alpha_y < 0.3:
            self.alpha_y = clamp(2 * self.alpha_y, 0.1, 0.3)


@ti.dataclass
class DiffuseBxDF:
    R: vec3
    type: ti.i32

    @ti.func
    def f(self, wo, wi, mode):
        result = vec3(0.0)
        if same_hemisphere(wo, wi):
            result = self.R * inv_pi
        return result

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()

        if sample_flags & BXDF_REFLECTION != 0:
            wi = sample_cosine_hemisphere(u)
            if wo.z < 0:
                wi.z *= -1
            pdf = cosine_hemisphere_pdf(abs_cos_theta(wi))
            bs.f = self.R * inv_pi
            bs.wi = wi
            bs.pdf = pdf
            bs.flags = BXDF_DIFFUSE_REFLECTION

        return bs

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        pdf_val = 0.0
        if (sample_flags & BXDF_REFLECTION != 0) and same_hemisphere(wo, wi):
            pdf_val = cosine_hemisphere_pdf(abs_cos_theta(wi))
        return pdf_val

    @ti.func
    def flags(self):
        return BXDF_DIFFUSE_REFLECTION if self.R.max() > 0 else BXDF_NONE


@ti.dataclass
class DiffuseTransmissionBxDF:
    R: vec3
    T: vec3
    type: ti.i32

    @ti.func
    def f(self, wo, wi, mode):
        result = vec3(0.0)
        if same_hemisphere(wo, wi):
            result = self.R * inv_pi
        else:
            result = self.T * inv_pi
        return result

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()

        pr = self.R.max()
        pt = self.T.max()

        if not (sample_flags & BXDF_REFLECTION != 0):
            pr = 0.0
        if not (sample_flags & BXDF_TRANSMISSION != 0):
            pt = 0.0

        if pr != 0 or pt != 0:  # TODO: should have been scalar
            if uc < (pr / (pr + pt)):
                wi = sample_cosine_hemisphere(u)
                if wo.z < 0:
                    wi.z *= -1
                pdf = cosine_hemisphere_pdf(abs_cos_theta(wi)) * pr / (pr + pt)
                bs.f = self.f(wo, wi, mode)
                bs.wi = wi
                bs.pdf = pdf
                bs.flags = BXDF_DIFFUSE_REFLECTION
            else:
                wi = sample_cosine_hemisphere(u)
                if wo.z > 0:
                    wi.z *= -1
                pdf = cosine_hemisphere_pdf(abs_cos_theta(wi)) * pt / (pr + pt)
                bs.f = self.f(wo, wi, mode)
                bs.wi = wi
                bs.pdf = pdf
                bs.flags = BXDF_DIFFUSE_TRANSMISSION

        return bs

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        pdf_val = 0.0

        pr = self.R.max()
        pt = self.T.max()

        if not (sample_flags & BXDF_REFLECTION != 0):
            pr = 0
        if not (sample_flags & BXDF_TRANSMISSION != 0):
            pt = 0

        if pr > 0 or pt > 0:
            if same_hemisphere(wo, wi):
                pdf_val = pr / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))
            else:
                pdf_val = pt / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))

        return pdf_val

    @ti.func
    def flags(self):
        result = BXDF_DIFFUSE_REFLECTION if self.R.max() > 0 else BXDF_NONE | BXDF_DIFFUSE_TRANSMISSION if self.T.max() > 0 else BXDF_NONE
        return result


@ti.dataclass
class DielectricBxDF:
    eta: ti.f32
    color: vec3
    mf_distrib: TrowbridgeReitzDistribution
    type: ti.i32

    @ti.func
    def init_tr_distribution(self, roughness):
        alpha = self.mf_distrib.roughness_to_alpha(roughness)
        self.mf_distrib.initialize(alpha, alpha)

    @ti.func
    def f(self, wo, wi, mode):
        result = vec3(0.0)
        if not (self.eta == 1 or self.mf_distrib.effectively_smooth()):
            cosTheta_o = cos_theta(wo)
            cosTheta_i = cos_theta(wi)
            reflect = cosTheta_i * cosTheta_o > 0
            etap = 1.0
            if not reflect:
                etap = self.eta if cosTheta_o > 0.0 else 1.0 / self.eta

            wm = wi * etap + wo
            if cosTheta_i != 0 and cosTheta_o != 0 and length_squared(wm) != 0:
                wm = face_forward(normalize(wm), vec3(0.0, 0.0, 1.0))
                if dot(wm, wi) * cosTheta_i >= 0 and dot(wm, wo) * cosTheta_o >= 0:
                    F = fr_dielectric(dot(wo, wm), self.eta)
                    if reflect:
                        result = self.mf_distrib.d1(wm) * self.mf_distrib.g2(wo, wi) * F / ti.abs(
                            4.0 * cosTheta_i * cosTheta_o)
                    else:
                        denom = pow(dot(wi, wm) + dot(wo, wm) / etap, 2) * cosTheta_i * cosTheta_o
                        ft = self.mf_distrib.d1(wm) * (1 - F) * self.mf_distrib.g2(wo, wi) * ti.abs(
                            dot(wi, wm) * dot(wo, wm) / denom)
                        if mode:
                            ft /= etap ** 2
                        result = ft
        return result

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()

        if self.eta == 1 or self.mf_distrib.effectively_smooth():
            bs = self.sample_f_smooth(wo, uc, u, mode, sample_flags)
        else:
            bs = self.sample_f_rough(wo, uc, u, mode, sample_flags)

        return bs

    @ti.func
    def sample_f_smooth(self, wo, uc, u, mode, sample_flags):
        bs = BSDFSample()

        # R = fr_dielectric(cos_theta(wo), self.eta)
        R, cos_t, eta_it, eta_ti = fresnel(cos_theta(wo), self.eta)

        T = 1.0 - R
        pr, pt = R, T

        pr = R if (sample_flags & BXDF_REFLECTION != 0) else 0.0
        pt = T if (sample_flags & BXDF_TRANSMISSION != 0) else 0.0

        etap = 1.0

        if pr != 0 and pt != 0:
            if uc < pr / (pr + pt):
                wi = vec3(-wo.x, -wo.y, wo.z)

                fr = R / abs_cos_theta(wi)
                bs.f = fr  * self.color
                bs.wi = wi
                bs.eta = etap
                bs.pdf = pr / (pr + pt)
                bs.flags = BXDF_SPECULAR_REFLECTION
            else:
                n = vec3(0.0, 0.0, 1.0) if cos_theta(wo) > 0 else vec3(0.0, 0.0, -1.0)
                # refracted, wi, etap = refract(wo, n, self.eta)

                d = -wo

                entering = d.dot(n) < 0
                eta_t = self.eta if entering else 1.0
                eta_i = 1.0 if entering else self.eta
                n = n if entering else -n  # Flip normal if the ray is exiting the medium
                n_dot_i = d.dot(n)

                eta = eta_i / eta_t

                sin2_theta_i = max(0.0, 1 - n_dot_i ** 2)
                sin2_theta_t = sin2_theta_i * eta ** 2


                refracted = sin2_theta_t <= 1.0

                if refracted:
                    wi = (eta * d + (eta * n_dot_i - cos_t) * n).normalized()
                    ft = T / abs_cos_theta(wi)
                    etap = eta
                    if mode:
                        ft /= etap ** 2
                    bs.f = ft  * self.color
                    bs.wi = wi
                    bs.pdf = pt / (pr + pt)
                    bs.eta = etap
                    bs.flags = BXDF_SPECULAR_TRANSMISSION
                else:
                    # Total internal reflection
                    print("Rare TIR happened!")
                    bs.f = self.color
                    bs.wi = vec3(-wo.x, -wo.y, wo.z)
                    bs.pdf = 1
                    bs.eta = etap
                    bs.flags = BXDF_SPECULAR_REFLECTION

        return bs

    @ti.func
    def sample_f_rough(self, wo, uc, u, mode, sample_flags):
        bs = BSDFSample()

        wm = self.mf_distrib.sample_wm(wo, u)
        R = fr_dielectric(dot(wo, wm), self.eta)

        T = 1.0 - R
        pr, pt = R, T

        pr = R if (sample_flags & BXDF_REFLECTION != 0) else 0.0
        pt = T if (sample_flags & BXDF_TRANSMISSION != 0) else 0.0

        if pr != 0 and pt != 0:
            if uc < pr / (pr + pt):
                wi = reflect(wo, wm)
                if same_hemisphere(wo, wi):
                    pdf = self.mf_distrib.pdf(wo, wm) / (4.0 * ti.abs(dot(wo, wm))) * pr / (pr + pt)
                    f = self.mf_distrib.d1(wm) * self.mf_distrib.g2(wo, wi) * R / (4.0 * cos_theta(wi) * cos_theta(wo))
                    bs.f = f  # * self.color
                    bs.pdf = pdf
                    bs.wi = wi
                    bs.flags = BXDF_GLOSSY_REFLECTION
            else:
                etap = 1.0
                refracted, wi, etap = refract(wo, wm, self.eta)
                if not same_hemisphere(wo, wi) and wi[2] != 0 and refracted:
                    denom = (dot(wi, wm) + dot(wo, wm) / etap) ** 2
                    dwm_dwi = ti.abs(dot(wi, wm)) / denom
                    pdf = self.mf_distrib.pdf(wo, wm) * dwm_dwi * pt / (pr + pt)
                    ft = T * self.mf_distrib.d1(wm) * self.mf_distrib.g2(wo, wi) * ti.abs(
                        dot(wi, wm) * dot(wo, wm) / (cos_theta(wi) * cos_theta(wo) * denom))
                    if mode:
                        ft /= etap ** 2
                    bs.f = ft  # * self.color
                    bs.wi = wi
                    bs.pdf = pdf
                    bs.flags = BXDF_GLOSSY_TRANSMISSION

        return bs

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        pdf_val = 0.0
        if not (self.eta == 1 or self.mf_distrib.effectively_smooth()):
            cosTheta_o = cos_theta(wo)
            cosTheta_i = cos_theta(wi)
            reflect = cosTheta_i * cosTheta_o > 0
            etap = 1.0

            if not reflect:
                etap = self.eta if cosTheta_o > 0.0 else (1.0 / self.eta)

            wm = wi * etap + wo

            if cosTheta_i != 0 and cosTheta_o != 0 and length_squared(wm) != 0:
                wm = face_forward(normalize(wm), vec3(0.0, 0.0, 1.0))
                if dot(wm, wi) * cosTheta_i >= 0 and dot(wm, wo) * cosTheta_o >= 0:
                    R = fr_dielectric(dot(wo, wm), self.eta)
                    T = 1 - R

                    pr = R if (sample_flags & BXDF_REFLECTION != 0) else 0.0
                    pt = T if (sample_flags & BXDF_TRANSMISSION != 0) else 0.0

                    if pr != 0 or pt != 0:
                        if reflect:
                            pdf_val = self.mf_distrib.pdf(wo, wm) / (4.0 * ti.abs(dot(wo, wm))) * pr / (pr + pt)
                        else:
                            denom = pow(dot(wi, wm) + dot(wo, wm) / etap, 2)
                            dwm_dwi = ti.abs(dot(wi, wm)) / denom
                            pdf_val = self.mf_distrib.pdf(wo, wm) * dwm_dwi * pt / (pr + pt)
        return pdf_val

    @ti.func
    def flags(self):
        flags = BXDF_TRANSMISSION if self.eta == 1.0 else (BXDF_REFLECTION | BXDF_TRANSMISSION)
        if self.mf_distrib.effectively_smooth():
            flags |= BXDF_SPECULAR
        else:
            flags |= BXDF_GLOSSY
        return flags

    @ti.func
    def regularize(self):
        self.mf_distrib.regularize()


@ti.dataclass
class ConductorBxDF:
    eta: vec3
    k: vec3
    mf_distrib: TrowbridgeReitzDistribution
    type: ti.i32

    @ti.func
    def init_tr_distribution(self, roughness):
        alpha = self.mf_distrib.roughness_to_alpha(roughness)
        self.mf_distrib.initialize(alpha, alpha)

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()

        if sample_flags & BXDF_REFLECTION != 0:
            if self.mf_distrib.effectively_smooth():
                wi = vec3(-wo.x, -wo.y, wo.z)
                fr = fr_complex(abs_cos_theta(wi), self.eta, self.k) / abs_cos_theta(wi)
                bs.f = fr
                bs.wi = wi
                bs.pdf = 1.0
                bs.flags = BXDF_SPECULAR_REFLECTION
            else:
                if wo.z != 0:
                    wm = self.mf_distrib.sample_wm(wo, u)
                    wi = reflect(wo, wm)
                    if same_hemisphere(wo, wi):
                        cos_theta_o = abs_cos_theta(wo)
                        cos_theta_i = abs_cos_theta(wi)
                        if cos_theta_i != 0 and cos_theta_o != 0:
                            pdf = self.mf_distrib.pdf(wo, wm) / (4 * ti.abs(dot(wo, wm)))
                            F = fr_complex(ti.abs(dot(wo, wm)), self.eta, self.k)
                            f = self.mf_distrib.d1(wm) * F * self.mf_distrib.g2(wo, wi) / (
                                    4 * cos_theta_i * cos_theta_o)
                            bs.f = f
                            bs.wi = wi
                            bs.pdf = pdf
                            bs.flags = BXDF_GLOSSY_REFLECTION

        return bs

    @ti.func
    def f(self, wo, wi, mode):
        result = vec3(0.0)
        if same_hemisphere(wo, wi) and not self.mf_distrib.effectively_smooth():
            cos_theta_o = abs_cos_theta(wo)
            cos_theta_i = abs_cos_theta(wi)
            if cos_theta_i != 0 and cos_theta_o != 0:
                wm = wi + wo
                if length_squared(wm) != 0:
                    wm = normalize(wm)
                    F = fr_complex(ti.abs(dot(wo, wm)), self.eta, self.k)
                    result = self.mf_distrib.d1(wm) * F * self.mf_distrib.g2(wo, wi) / (4 * cos_theta_i * cos_theta_o)
        return result

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        pdf_val = 0.0
        if (sample_flags & BXDF_REFLECTION != 0) and same_hemisphere(wo,
                                                                     wi) and not self.mf_distrib.effectively_smooth():
            wm = wo + wi
            wm = face_forward(normalize(wm), vec3(0.0, 0.0, 1.0))
            pdf_val = self.mf_distrib.pdf(wo, wm) / (4 * ti.abs(dot(wo, wm)))
        return pdf_val

    @ti.func
    def flags(self):
        return BXDF_SPECULAR_REFLECTION if self.mf_distrib.effectively_smooth() else BXDF_GLOSSY_REFLECTION

    @ti.func
    def regularize(self):
        self.mf_distrib.regularize()


@ti.dataclass
class MirrorBxDF:
    R: vec3  # Reflectance of the mirror
    type: ti.i32

    @ti.func
    def f(self, wo, wi, mode):
        # The BRDF of a perfect mirror is a delta function, hence it returns 0 for all directions except the perfect reflection direction
        return vec3(0.0)  # Delta function cannot be represented in continuous functions

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()

        if sample_flags & BXDF_REFLECTION != 0:
            # Compute the perfect reflection direction
            wi = vec3(-wo.x, -wo.y, wo.z)  # Reflect wo around the surface normal

            # Since this is a perfect mirror, the pdf is always 1 for the reflection direction
            pdf = 1.0

            # The f value is R / cos(theta) for the reflection direction
            fr = self.R / abs_cos_theta(wi)
            bs.f = fr
            bs.wi = wi
            bs.pdf = pdf
            bs.flags = BXDF_SPECULAR_REFLECTION

        return bs

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        # The pdf of sampling the perfect reflection direction is 1.0, and 0 for all other directions
        pdf_val = 0.0
        if (sample_flags & BXDF_REFLECTION != 0) and wi.x == -wo.x and wi.y == -wo.y and wi.z == wo.z:
            pdf_val = 1.0
        return pdf_val

    @ti.func
    def flags(self):
        return BXDF_SPECULAR_REFLECTION if self.R.max() > 0 else BXDF_NONE


@ti.dataclass
class IdealSpecularBxDF:
    color: vec3  # Uniform color for both reflection and transmission
    eta: ti.f32  # Index of refraction of the medium (assume the other medium is air)
    type: ti.i32

    @ti.func
    def ideal_specular_transmit(self, d, n, u, mode, sample_flags):
        result = TransmitResult(vec3(0.0), 0.0, 0.0, BXDF_SPECULAR_REFLECTION, 1.0)

        wo = d
        d = -d

        entering = d.dot(n) < 0
        eta_t = self.eta if entering else 1.0
        eta_i = 1.0 if entering else self.eta
        n = n if entering else -n  # Flip normal if the ray is exiting the medium
        n_dot_i = d.dot(n)

        eta = eta_i / eta_t
        sin_t_2 = eta ** 2 * (1.0 - n_dot_i ** 2)

        etap = 1.0

        if sin_t_2 <= 1.0:  # Not total internal reflection

            R, cos_t, eta_it, eta_ti = fresnel(cos_theta(wo), self.eta)

            T = 1.0 - R
            pr, pt = R, T

            pr = R if (sample_flags & BXDF_REFLECTION != 0) else 0.0
            pt = T if (sample_flags & BXDF_TRANSMISSION != 0) else 0.0

            if pr != 0 and pt != 0:
                if u < pr / (pr + pt):  # Reflect
                    result.direction = vec3(-wo.x, -wo.y, wo.z) #self.ideal_specular_reflect(d, n)
                    result.factor = R / abs_cos_theta(result.direction)
                    result.pdf = pr / (pr + pt)
                    result.bsdf_type = BXDF_SPECULAR_REFLECTION
                    result.eta = etap
                else:  # Transmit
                    result.direction = (eta * d + (eta * n_dot_i - cos_t) * n).normalized()
                    # result.direction = (d / eta + (n_dot_i / eta - cos_t) * n).normalized()
                    result.factor = T / abs_cos_theta(result.direction)
                    etap = eta
                    if mode:
                        result.factor /= etap**2
                    result.pdf = pt / (pr + pt)
                    result.bsdf_type = BXDF_SPECULAR_TRANSMISSION
                    result.eta = etap
        else:  # Total internal reflection
            print("TIR")
            result.direction = vec3(-wo.x, -wo.y, wo.z) #self.ideal_specular_reflect(d, n)
            result.factor = 1.0
            result.pdf = 1.0
            result.bsdf_type = BXDF_SPECULAR_REFLECTION
            result.eta = etap

        return result

    @ti.func
    def f(self, wo, wi, mode):
        return vec3(0.0)

    @ti.func
    def sample_f(self, wo, uc, u, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()
        n = vec3(0.0, 0.0, 1.0) if cos_theta(wo) > 0 else vec3(0.0, 0.0, -1.0)
        transmit_result = self.ideal_specular_transmit(wo, n, uc, mode, sample_flags)

        bs.f = transmit_result.factor * self.color
        bs.wi = transmit_result.direction
        bs.pdf = transmit_result.pdf
        bs.flags = transmit_result.bsdf_type
        bs.eta = transmit_result.eta

        return bs

    @ti.func
    def pdf(self, wo, wi, mode, sample_flags=BXDF_ALL):
        return 0.0

    @ti.func
    def flags(self):
        flags = BXDF_TRANSMISSION if self.eta == 1.0 else (BXDF_REFLECTION | BXDF_TRANSMISSION)
        flags |= BXDF_SPECULAR
        return flags


@ti.dataclass
class TransmitResult:
    direction: vec3
    factor: ti.f32
    pdf: ti.f32
    bsdf_type: ti.i32
    eta: ti.f32


@ti.dataclass
class BSDF:
    type: ti.i32
    frame: Frame
    diffuse: DiffuseBxDF
    transmit: DiffuseTransmissionBxDF
    dielectric: DielectricBxDF
    conductor: ConductorBxDF
    mirror: MirrorBxDF
    specular: IdealSpecularBxDF

    @ti.func
    def init_frame(self, normal, dpdu):
        # self.frame = frame_from_z(normal)
        self.frame = frame_from_xz(normalize(dpdu), normal)

    @ti.func
    def to_local(self, v):
        return self.frame.to_local(v)

    @ti.func
    def from_local(self, v):
        return self.frame.from_local(v)

    @ti.func
    def add_diffuse(self, R):
        self.diffuse.R = R
        self.type = 0

    @ti.func
    def add_transmission(self, R, T):
        self.transmit.R = R
        self.transmit.T = T
        self.type = 1

    @ti.func
    def add_dielectric(self, eta, color, uroughness, vroughness):
        self.dielectric.eta = eta
        self.dielectric.color = color
        alpha_x = self.dielectric.mf_distrib.roughness_to_alpha(uroughness)
        alpha_y = self.dielectric.mf_distrib.roughness_to_alpha(vroughness)
        self.dielectric.mf_distrib.initialize(alpha_x, alpha_y)
        self.type = 2

    @ti.func
    def add_conductor(self, eta, k, uroughness, vroughness):
        self.conductor.eta = eta
        self.conductor.k = k
        alpha_x = self.conductor.mf_distrib.roughness_to_alpha(uroughness)
        alpha_y = self.conductor.mf_distrib.roughness_to_alpha(vroughness)
        self.conductor.mf_distrib.initialize(alpha_x, alpha_y)
        self.type = 3

    @ti.func
    def add_mirror(self, R):
        self.mirror.R = R
        self.type = 4

    @ti.func
    def add_specular(self, color, eta):
        self.specular.color = color
        self.specular.eta = eta
        self.type = 5

    @ti.func
    def f(self, wo_world, wi_world, mode=1):
        wo = self.to_local(wo_world)
        wi = self.to_local(wi_world)
        result = vec3(0.0)
        if wo.z != 0:
            if self.type == 0:
                result = self.diffuse.f(wo, wi, mode)
            elif self.type == 1:
                result = self.transmit.f(wo, wi, mode)
            elif self.type == 2:
                result = self.dielectric.f(wo, wi, mode)
            elif self.type == 3:
                result = self.conductor.f(wo, wi, mode)
            elif self.type == 4:
                result = self.mirror.f(wo, wi, mode)
            elif self.type == 5:
                result = self.specular.f(wo, wi, mode)
        return result

    @ti.func
    def sample_f(self, wo_world, u, u2, mode=1, sample_flags=BXDF_ALL):
        bs = BSDFSample()
        bxdf_sample = bs

        wo = self.to_local(wo_world)

        if self.type == self.diffuse.type:
            if wo.z != 0 and (self.diffuse.flags() & sample_flags != 0):
                bxdf_sample = self.diffuse.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == self.transmit.type:
            if wo.z != 0 and (self.transmit.flags() & sample_flags != 0):
                bxdf_sample = self.transmit.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == self.dielectric.type:
            if wo.z != 0 and (self.dielectric.flags() & sample_flags != 0):
                bxdf_sample = self.dielectric.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == self.conductor.type:
            if wo.z != 0 and (self.conductor.flags() & sample_flags != 0):
                bxdf_sample = self.conductor.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == self.mirror.type:
            if wo.z != 0 and (self.mirror.flags() & sample_flags != 0):
                bxdf_sample = self.mirror.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)
        elif self.type == 5:
            if wo.z != 0 and (self.specular.flags() & sample_flags != 0):
                bxdf_sample = self.specular.sample_f(wo, u, u2, mode, sample_flags)
                bxdf_sample.wi = self.from_local(bxdf_sample.wi)

        if not bs.f.max() > 0 and bxdf_sample.pdf != 0 and bxdf_sample.wi.z != 0:
            bs.f = bxdf_sample.f
            bs.wi = bxdf_sample.wi
            bs.pdf = bxdf_sample.pdf
            bs.flags = bxdf_sample.flags

        return bs

    @ti.func
    def pdf(self, wo_world, wi_world, mode=1, sample_flags=BXDF_ALL):
        wo = self.to_local(wo_world)
        wi = self.to_local(wi_world)
        result = 0.0
        if wo.z != 0:
            if self.type == 0:
                result = self.diffuse.pdf(wo, wi, mode, sample_flags)
            elif self.type == 1:
                result = self.transmit.pdf(wo, wi, mode, sample_flags)
            elif self.type == 2:
                result = self.dielectric.pdf(wo, wi, mode, sample_flags)
            elif self.type == 3:
                result = self.conductor.pdf(wo, wi, mode, sample_flags)
            elif self.type == 4:
                result = self.mirror.pdf(wo, wi, mode, sample_flags)
            elif self.type == 5:
                result = self.specular.pdf(wo, wi, mode, sample_flags)
        return result

    @ti.func
    def flags(self):
        flag = BXDF_NONE
        if self.type == self.diffuse.type:
            flag = self.diffuse.flags()
        elif self.type == self.transmit.type:
            flag = self.transmit.flags()
        elif self.type == self.dielectric.type:
            flag = self.dielectric.flags()
        elif self.type == self.conductor.type:
            flag = self.conductor.flags()
        elif self.type == self.mirror.type:
            flag = self.mirror.flags()
        elif self.type == self.specular.type:
            flag = self.specular.flags()

        return flag
