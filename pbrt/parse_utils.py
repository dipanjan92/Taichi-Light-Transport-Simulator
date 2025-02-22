# parse_utils.py
import re
import taichi as ti
import numpy as np
from taichi.math import vec3

def extract_numbers(s):
    number_pattern = r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?'
    return re.findall(number_pattern, s)

def clean_brackets(s):
    return s.strip().strip("[]").strip('"')

def parse_value(param_type, raw_value):
    cleaned = clean_brackets(raw_value)
    if param_type in ["float", "integer", "point3", "point2", "vector", "vector3", "rgb", "spectrum", "point", "normal"]:
        tokens = extract_numbers(cleaned)
        if param_type == "normal":
            tokens = [tok for tok in tokens if tok != "."]
        if not tokens:
            return None
    else:
        tokens = cleaned.split()
    try:
        if param_type == "float":
            return float(tokens[0])
        elif param_type == "integer":
            if len(tokens) > 1:
                return [int(float(tok)) for tok in tokens]
            else:
                return int(float(tokens[0]))
        elif param_type in ["point3", "point2", "point"]:
            return tuple(float(tok) for tok in tokens)
        elif param_type in ["vector", "vector3", "rgb", "spectrum"]:
            if len(tokens) >= 3:
                return tuple(float(tok) for tok in tokens[:3])
            else:
                return None
        elif param_type == "normal":
            return tuple(float(tok) for tok in tokens)
        elif param_type == "bool":
            return tokens[0].lower() == "true"
        elif param_type in ["string", "texture"]:
            return tokens[0]
        else:
            return cleaned
    except Exception:
        return None

def to_vec3(val):
    try:
        return ti.Vector([float(x) for x in val])
    except Exception:
        return ti.Vector([0.0, 0.0, 0.0])

def py_cross(a, b):
    return np.cross(np.array(a), np.array(b))

def py_normalize(v):
    arr = np.array(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return vec3(arr[0], arr[1], arr[2])

def set_matrix(m):
    IDENTITY_4x4 = (1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0)
    return tuple(m) if m and len(m) == 16 else IDENTITY_4x4

def multiply_matrix4(A, B):
    C = [0.0]*16
    for r in range(4):
        for c in range(4):
            val = 0.0
            for k in range(4):
                val += A[r*4+k]*B[k*4+c]
            C[r*4+c] = val
    return tuple(C)

def lookat_matrix(eye, target, up):
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    M = np.array([
        [right[0], true_up[0], -forward[0], eye[0]],
        [right[1], true_up[1], -forward[1], eye[1]],
        [right[2], true_up[2], -forward[2], eye[2]],
        [0.0,      0.0,        0.0,        1.0]
    ], dtype=float)
    return tuple(M.flatten())
