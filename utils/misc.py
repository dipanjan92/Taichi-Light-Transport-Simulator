import taichi as ti
from taichi.math import vec2, sign, pi, vec3, cos, atan2, dot, sin, cos, length, cross, clamp, sqrt

from utils.constants import EPSILON, inv_pi, INF


@ti.func
def safe_inverse(x):
    return 1.0 / x if abs(x) > EPSILON else 1.0 / EPSILON


@ti.func
def same_hemisphere(w, wp):
    return w[2] * wp[2] > 0


@ti.func
def cos_theta(w):
    return w[2]


@ti.func
def cos2_theta(w):
    return w[2] * w[2]


@ti.func
def abs_cos_theta(w):
    return ti.abs(w[2])


@ti.func
def sin_theta(w):
    return sqrt(sin2_theta(w))


@ti.func
def sin2_theta(w):
    return max(0.0, 1 - cos2_theta(w))


@ti.func
def tan2_theta(w):
    return sin2_theta(w) / cos2_theta(w)


@ti.func
def tan_theta(w):
    return sin_theta(w) / cos_theta(w)


@ti.func
def cos_phi(w):
    _sin_theta = sin_theta(w)
    return 1 if _sin_theta == 0 else clamp(w[0] / _sin_theta, -1, 1)


@ti.func
def sin_phi(w):
    _sin_theta = sin_theta(w)
    return 0 if _sin_theta == 0 else clamp(w[1] / _sin_theta, -1, 1)


@ti.func
def phi(w):
    return atan2(sin_phi(w), cos_phi(w))


@ti.func
def face_forward(v, n):
    return -v if dot(v, n) < 0 else v


@ti.func
def safe_sqrt(x):
    return sqrt(max(0.0, x))


@ti.func
def fr_dielectric(cos_theta_i, eta):
    cos_theta_i = clamp(cos_theta_i, -1.0, 1.0)
    to_return = 0.0
    # Potentially flip interface orientation for Fresnel equations
    if cos_theta_i < 0:
        eta = 1.0 / eta
        cos_theta_i = -cos_theta_i

    # Compute cos_theta_t for Fresnel equations using Snell's law
    sin2_theta_i = 1.0 - cos_theta_i ** 2
    sin2_theta_t = sin2_theta_i / eta ** 2
    if sin2_theta_t >= 1.0:
        to_return = 1.0  # Total internal reflection
    else:
        cos_theta_t = safe_sqrt(1.0 - sin2_theta_t)
        r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t)
        r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)
        to_return = (r_parl * r_parl + r_perp * r_perp) / 2.0
    return to_return


@ti.func
def fresnel(cos_theta_i, eta):
    # Initialize output variables
    r = 0.0
    cos_theta_t = 0.0
    eta_it = 0.0
    eta_ti = 0.0

    # Check if the ray is entering or exiting the surface
    outside_mask = cos_theta_i >= 0.0

    rcp_eta = 1.0 / eta
    eta_it = eta if outside_mask else rcp_eta
    eta_ti = rcp_eta if outside_mask else eta

    # Calculate the squared sine of the transmitted angle using Snell's law
    sin2_theta_t = eta_ti * eta_ti * (1.0 - cos_theta_i * cos_theta_i)
    cos_theta_t_sqr = 1.0 - sin2_theta_t

    # Absolute cosines of the incident and transmitted rays
    cos_theta_i_abs = ti.abs(cos_theta_i)
    cos_theta_t_abs = ti.sqrt(ti.max(0.0, cos_theta_t_sqr))

    # Handle special cases where the index is matched or cos_theta_i is zero
    index_matched = eta == 1.0
    special_case = index_matched or (cos_theta_i_abs == 0.0)

    r_sc = 0.0 if index_matched else 1.0

    # Calculate the reflection coefficients
    a_s = (eta_it * cos_theta_t_abs - cos_theta_i_abs) / (eta_it * cos_theta_t_abs + cos_theta_i_abs)
    a_p = (eta_it * cos_theta_i_abs - cos_theta_t_abs) / (eta_it * cos_theta_i_abs + cos_theta_t_abs)

    r = 0.5 * (a_s * a_s + a_p * a_p)

    # Apply the special case handling
    if special_case:
        r = r_sc

    # Adjust the sign of the transmitted direction
    if cos_theta_i < 0:
        cos_theta_t = -cos_theta_t_abs
    else:
        cos_theta_t = cos_theta_t_abs

    return r, cos_theta_t, eta_it, eta_ti


@ti.func
def reflect(wo, n):
    return -wo + 2 * dot(wo, n) * n


@ti.func
def refract(wi, n, eta):
    cos_theta_i = dot(n, wi)
    refracted = 0
    wt = vec3(0.0)
    etap = 1.0

    if cos_theta_i < 0:
        print(("low prob"))
        eta = 1 / eta
        cos_theta_i = -cos_theta_i
        n = -n

    # Compute sin²θᵢ using Snell's law
    sin2_theta_i = max(0.0, 1 - cos_theta_i ** 2)
    sin2_theta_t = sin2_theta_i / eta ** 2

    # Handle total internal reflection case
    if sin2_theta_t < 1:
        cos_theta_t = ti.sqrt(1 - sin2_theta_t)
        wt = -wi * eta + (cos_theta_i / eta - cos_theta_t) * n
        refracted = 1
        etap = eta
    else:
        # Total internal reflection
        print("Rare TIR happened!")
        refracted = 0
        wt = vec3(0.0)
        etap = 0.0

    return refracted, wt, etap


@ti.func
def lerp(a, b, t):
    return a + t * (b - a)


@ti.func
def fr_complex(cosTheta_i, eta_re, eta_im):
    # Clamp cosTheta_i to the range [0, 1]
    cosTheta_i = min(max(cosTheta_i, 0.0), 1.0)

    # Compute sin²θᵢ
    sin2Theta_i = 1.0 - cosTheta_i * cosTheta_i

    # Complex eta = eta_re + i * eta_im
    # Complex sin²θₜ = sin²θᵢ / eta²
    eta_re2 = eta_re * eta_re - eta_im * eta_im
    eta_im2 = 2.0 * eta_re * eta_im
    sin2Theta_t_re = sin2Theta_i * eta_re2
    sin2Theta_t_im = sin2Theta_i * eta_im2

    # Compute complex sqrt(1 - sin²θₜ)
    a_re = 1.0 - sin2Theta_t_re
    a_im = -sin2Theta_t_im
    cosTheta_t_re = sqrt((sqrt(a_re * a_re + a_im * a_im) + a_re) / 2.0)
    cosTheta_t_im = a_im / (2.0 * cosTheta_t_re)

    # Compute r_parallel
    eta_cosTheta_i_re = eta_re * cosTheta_i
    eta_cosTheta_i_im = eta_im * cosTheta_i
    num_re_parl = eta_cosTheta_i_re - cosTheta_t_re
    num_im_parl = eta_cosTheta_i_im - cosTheta_t_im
    denom_re_parl = eta_cosTheta_i_re + cosTheta_t_re
    denom_im_parl = eta_cosTheta_i_im + cosTheta_t_im
    r_parl_re = (num_re_parl * denom_re_parl + num_im_parl * denom_im_parl) / (
            denom_re_parl * denom_re_parl + denom_im_parl * denom_im_parl)
    r_parl_im = (num_im_parl * denom_re_parl - num_re_parl * denom_im_parl) / (
            denom_re_parl * denom_re_parl + denom_im_parl * denom_im_parl)

    # Compute r_perpendicular
    num_re_perp = cosTheta_i - eta_re * cosTheta_t_re + eta_im * cosTheta_t_im
    num_im_perp = -eta_re * cosTheta_t_im - eta_im * cosTheta_t_re
    denom_re_perp = cosTheta_i + eta_re * cosTheta_t_re - eta_im * cosTheta_t_im
    denom_im_perp = eta_re * cosTheta_t_im + eta_im * cosTheta_t_re
    r_perp_re = (num_re_perp * denom_re_perp + num_im_perp * denom_im_perp) / (
            denom_re_perp * denom_re_perp + denom_im_perp * denom_im_perp)
    r_perp_im = (num_im_perp * denom_re_perp - num_re_perp * denom_im_perp) / (
            denom_re_perp * denom_re_perp + denom_im_perp * denom_im_perp)

    # Compute the final Fresnel reflection coefficient
    r_parl_norm = r_parl_re * r_parl_re + r_parl_im * r_parl_im
    r_perp_norm = r_perp_re * r_perp_re + r_perp_im * r_perp_im

    return 0.5 * (r_parl_norm + r_perp_norm)


@ti.func
def transmit(wi, cos_theta_t, eta_ti):
    # Calculate the direction of the transmitted vector
    # wi is the incident vector, eta_ti is the ratio of refractive indices (eta_i / eta_t)

    # Refracted vector calculation using Snell's law:
    wt = vec3([-eta_ti * wi.x, -eta_ti * wi.y, cos_theta_t])

    # Correct the Z component to ensure Snell's law is followed:
    wt.z = ti.sqrt(ti.max(0.0, 1.0 - wt.x * wt.x - wt.y * wt.y)) * (1.0 if cos_theta_t > 0 else -1.0)

    return wt


@ti.func
def length_squared(v):
    return length(v) ** 2


@ti.func
def distance_squared(u, v):
    return length_squared(u - v)


@ti.func
def spherical_triangle_area(a, b, c):
    return ti.abs(2 * atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)))


@ti.func
def max_component(vec):
    max_val = vec[0]
    for i in range(1, vec.n):
        max_val = max(max_val, vec[i])
    return max_val


@ti.func
def gamma(n):
    machine_epsilon = ti.f32(1e-7)
    return (n * machine_epsilon) / (1 - n * machine_epsilon)
