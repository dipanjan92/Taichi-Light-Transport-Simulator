# =============================================================================
# pbrt_parser.py
# =============================================================================

import re
import numpy as np
import taichi as ti
from taichi.math import vec3
from base.bsdf import BSDF
from base.materials import Material
from primitives.primitives import Primitive, Triangle, Sphere

# =============================================================================
# Helper Function to Extract Valid Numbers
# =============================================================================

def extract_numbers(s):
    """
    Extracts all valid numbers (including integers, decimals, and scientific notation)
    from the string s and returns them as a list of strings.
    """
    number_pattern = r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?'
    return re.findall(number_pattern, s)

# =============================================================================
# Shared Helper Functions
# =============================================================================

def clean_brackets(s):
    """
    Remove leading and trailing square brackets and quotes from a string.
    For example, '[ 0.63 0.065 0.05 ]' becomes '0.63 0.065 0.05'
    """
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

def accumulate_property_line(lines, start_index):
    acc_line = lines[start_index].rstrip()
    i = start_index + 1
    while ']' not in acc_line and i < len(lines):
        acc_line += " " + lines[i].strip()
        i += 1
    return acc_line, i

def parse_property_line(line):
    m_decl = re.search(r'"([^"]+)"', line)
    if not m_decl:
        return None, None, None
    declaration = m_decl.group(1)
    parts = declaration.split()
    if len(parts) < 2:
        return None, None, None
    ptype, key = parts[0], parts[1]
    m_val = re.search(r'\[([\s\S]+?)\]', line)
    if m_val:
        raw_value = f"[{m_val.group(1).strip()}]"
        value = parse_value(ptype, raw_value)
        return ptype, key, value
    else:
        all_quotes = re.findall(r'"([^"]+)"', line)
        if len(all_quotes) >= 2:
            return ptype, key, all_quotes[1]
    return ptype, key, None

def parse_transform_line(line):
    m = re.search(r'\[([^\]]+)\]', line)
    if not m:
        return None
    numbers = extract_numbers(m.group(1))
    if len(numbers) < 16:
        return None
    try:
        return tuple(float(n) for n in numbers[:16])
    except Exception:
        return None

# =============================================================================
# Camera Parsing (unchanged; your camera creation remains separate)
# =============================================================================

def lookat_matrix(eye, target, up):
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    M = np.array([
        [right[0], true_up[0], -forward[0], eye[0]],
        [right[1], true_up[1], -forward[1], eye[1]],
        [right[2], true_up[2], -forward[2], eye[2]],
        [0.0,      0.0,        0.0,        1.0]
    ], dtype=float)
    return tuple(M.flatten())

def multiply_matrix4(A, B):
    C = [0.0] * 16
    for r in range(4):
        for c in range(4):
            val = 0.0
            for k in range(4):
                val += A[r * 4 + k] * B[k * 4 + c]
            C[r * 4 + c] = val
    return tuple(C)

def set_matrix(m):
    if len(m) == 16:
        return tuple(m)
    return IDENTITY_4x4

IDENTITY_4x4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
)

def parse_camera_from_file(filename):
    camera_data = {
        "type": None,
        "shutteropen": 0.0,
        "shutterclose": 1.0,
        "frameaspectratio": None,
        "screenwindow": None,
        "lensradius": 0.0,
        "focaldistance": 1e30,
        "fov": 90.0,
        "mapping": "equalarea",
        "lensfile": "",
        "aperturediameter": 1.0,
        "focusdistance": 10.0,
        "aperture": None,
        "near": 0.1,
        "far": 1000.0,
        "position": (0.0, 0.0, 0.0),
        "target": (0.0, 0.0, -1.0),
        "up": (0.0, 1.0, 0.0),
        "transform": IDENTITY_4x4,
    }
    current_transform = IDENTITY_4x4
    with open(filename, "r") as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        stripped_line = lines[i].strip()
        if not stripped_line:
            i += 1
            continue
        if stripped_line.startswith("WorldBegin"):
            break
        if stripped_line.startswith("Camera"):
            m = re.search(r'Camera\s+"([^"]+)"', stripped_line)
            if m:
                camera_data["type"] = m.group(1)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.startswith(" "):
                    break
                next_str = next_line.strip()
                m_decl = re.search(r'"([^"]+)"', next_str)
                if m_decl:
                    param_decl = m_decl.group(1)
                    parts = param_decl.split()
                    if len(parts) >= 2:
                        ptype, key = parts[0], parts[1]
                        m_val = re.search(r'\[(.*?)\]', next_str)
                        if m_val:
                            try:
                                val = float(m_val.group(1).split()[0])
                            except Exception:
                                val = m_val.group(1).split()[0]
                            camera_data[key] = val
                i += 1
            continue
        if stripped_line.startswith("LookAt"):
            tokens = stripped_line.split()
            if len(tokens) == 10:
                try:
                    ex, ey, ez = map(float, tokens[1:4])
                    lx, ly, lz = map(float, tokens[4:7])
                    ux, uy, uz = map(float, tokens[7:10])
                    camera_data["position"] = (ex, ey, ez)
                    camera_data["target"] = (lx, ly, lz)
                    camera_data["up"] = (ux, uy, uz)
                    L = lookat_matrix((ex, ey, ez), (lx, ly, lz), (ux, uy, uz))
                    current_transform = multiply_matrix4(L, current_transform)
                except Exception as e:
                    print("WARNING: could not parse LookAt.", e)
            i += 1
            continue
        if stripped_line.startswith("Transform"):
            mat = parse_transform_line(stripped_line)
            if mat is not None:
                current_transform = set_matrix(mat)
            i += 1
            continue
        if stripped_line.startswith("ConcatTransform"):
            mat_line = re.search(r'\[([^\]]+)\]', stripped_line)
            if mat_line:
                floats_str = mat_line.group(1)
                floats_nums = extract_numbers(floats_str)
                if len(floats_nums) == 16:
                    cat_mat = tuple(float(x) for x in floats_nums)
                    current_transform = multiply_matrix4(set_matrix(cat_mat), current_transform)
            i += 1
            continue
        i += 1
    camera_data["transform"] = current_transform
    return camera_data

# =============================================================================
# Light Parsing (unchanged)
# =============================================================================

def parse_light_property_line(line):
    m_decl = re.search(r'"([^"]+)"', line)
    if not m_decl:
        return None, None, None
    declaration = m_decl.group(1)
    parts = declaration.split()
    if len(parts) < 2:
        return None, None, None
    ptype, key = parts[0], parts[1]
    m_val = re.search(r'\[([^\]]+)\]', line)
    if not m_val:
        return ptype, key, None
    raw_value = f"[{m_val.group(1)}]"
    value = parse_value(ptype, raw_value)
    return ptype, key, value

def parse_pbrt_lights_and_counts(filename):
    lights = []
    light_counts = {
        "total_lights": 0,
        "point": 0,
        "spot": 0,
        "distant": 0,
        "infinite": 0,
        "goniometric": 0,
        "projection": 0,
        "area": 0,
    }
    is_area_light = False
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        if not stripped_line:
            i += 1
            continue
        if stripped_line.startswith("AttributeBegin"):
            is_area_light = True
            i += 1
            continue
        if stripped_line.startswith("AttributeEnd"):
            is_area_light = False
            i += 1
            continue
        if (stripped_line.startswith("LightSource") or
            stripped_line.startswith("AreaLightSource")):
            tokens = stripped_line.split()
            if len(tokens) < 2:
                i += 1
                continue
            light_type = tokens[1].strip('"')
            if is_area_light:
                light_counts["area"] += 1
            elif light_type in light_counts:
                light_counts[light_type] += 1
            light_counts["total_lights"] += 1
            light_data = {
                "type": light_type,
                "is_area": is_area_light,
                "from": [0.0, 0.0, 0.0],
                "to": [0.0, 0.0, 1.0],
                "I": [1.0, 1.0, 1.0],
                "L": [1.0, 1.0, 1.0],
                "scale": 1.0,
                "power": 0.0,
                "illuminance": 0.0,
                "filename": "",
                "coneangle": 30.0,
                "conedeltaangle": 5.0,
                "twosided": False,
                "extra": {}
            }
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('"'):
                ptype, key, value = parse_light_property_line(lines[i])
                if key is None:
                    i += 1
                    continue
                if key in light_data:
                    light_data[key] = value
                else:
                    light_data["extra"][key] = value
                i += 1
            lights.append(light_data)
        else:
            i += 1
    return lights, light_counts

# =============================================================================
# Material & Texture Parsing (updated)
# =============================================================================

def parse_pbrt_materials(filename):
    materials = {}
    textures = {}
    current_medium = ""
    medium_stack = []
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        stripped = line.strip()
        tokens = stripped.split()
        if not tokens:
            i += 1
            continue
        if tokens[0] == "AttributeBegin":
            medium_stack.append(current_medium)
            i += 1
            continue
        if tokens[0] == "AttributeEnd":
            if medium_stack:
                current_medium = medium_stack.pop()
            i += 1
            continue
        if tokens[0] == "MediumInterface":
            args = [t.strip('"') for t in tokens[1:]]
            if len(args) == 1:
                current_medium = args[0]
            elif len(args) >= 2:
                current_medium = args[0] if args[0] != "" else args[1]
            i += 1
            continue
        if tokens[0] == "Texture":
            tex_name = tokens[1].strip('"') if len(tokens) > 1 else ""
            tex_data = {"name": tex_name}
            i += 1
            while i < len(lines) and lines[i].startswith(" "):
                param_line, i = accumulate_property_line(lines, i)
                ptype, key, value = parse_property_line(param_line)
                if key is not None:
                    tex_data[key] = value
            textures[tex_name] = tex_data
            continue
        if tokens[0] in ["Material", "MakeNamedMaterial"]:
            mat_name = tokens[1].strip('"')
            # Initialize material with an extra "emission" key.
            material = {
                "name": mat_name,
                "type": None,
                "reflectance": None,
                "transmittance": None,
                "eta": None,
                "k": None,
                "roughness": None,
                "uroughness": None,
                "vroughness": None,
                "remaproughness": None,
                "displacement": None,
                "normalmap": None,
                "filename": None,
                "scale": None,
                "medium": current_medium if current_medium != "" else "",
                "tex_reflectance": "",
                "tex_filename": "",
                "emission": None
            }
            i += 1
            while i < len(lines) and lines[i] and lines[i][0].isspace():
                param_line, i = accumulate_property_line(lines, i)
                ptype, key, value = parse_property_line(param_line)
                if key is not None:
                    material[key] = value
            materials[mat_name] = material
            continue
        if tokens[0] == "NamedMaterial":
            mat_name = tokens[1].strip('"')
            if mat_name in materials:
                if current_medium != "":
                    materials[mat_name]["medium"] = current_medium
            else:
                materials[mat_name] = {
                    "name": mat_name,
                    "type": None,
                    "reflectance": None,
                    "transmittance": None,
                    "eta": None,
                    "k": None,
                    "roughness": None,
                    "uroughness": None,
                    "vroughness": None,
                    "remaproughness": None,
                    "displacement": None,
                    "normalmap": None,
                    "filename": None,
                    "scale": None,
                    "medium": current_medium if current_medium != "" else "",
                    "tex_reflectance": "",
                    "tex_filename": "",
                    "emission": None
                }
            i += 1
            continue
        i += 1
    for mat in materials.values():
        tex_ref = mat.get("tex_reflectance", "")
        if tex_ref and tex_ref in textures:
            tex_def = textures[tex_ref]
            mat["tex_filename"] = tex_def.get("filename", "")
    return list(materials.values())

def _extract_emission(mat):
    # Look for a key that is "emission" (case-insensitive)
    for key, value in mat.items():
        if key.lower() == "emission":
            return value
    return None

# -----------------------------------------------------------------------------
# Update Light Materials
# -----------------------------------------------------------------------------
def update_light_materials(materials_list, lights):
    # We assume that area light blocks reference a material by name.
    # When multiple light blocks reference the same material name, we want to create
    # extra copies of that material with each light's emission value.
    new_materials = []
    for light in lights:
        if "L" in light:
            # Use the "material" key if present; otherwise assume "Null"
            material_name = light.get("material", "Null")
            L_raw = light["L"]
            if isinstance(L_raw, (list, tuple)):
                L_val = tuple(float(x) for x in L_raw)
            elif isinstance(L_raw, str):
                nums = extract_numbers(L_raw)
                if len(nums) >= 3:
                    L_val = (float(nums[0]), float(nums[1]), float(nums[2]))
                else:
                    L_val = (0.0, 0.0, 0.0)
            else:
                L_val = (0.0, 0.0, 0.0)
            # Find all materials with this name in the base list.
            base_mats = [mat for mat in materials_list if mat.get("name", "") == material_name]
            if base_mats:
                # For the first occurrence, update its emission if not already set.
                first = base_mats[0]
                if first.get("emission") in (None, (0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)):
                    first["emission"] = L_val
                # For each subsequent light (if any), create a copy of the base material.
                # (If there are more lights than base materials, create new copies.)
                count = len([mat for mat in materials_list if mat.get("name", "").startswith(material_name + "_light_")])
                # Here we create one new copy per light occurrence beyond the first.
                new_mat = first.copy()
                new_mat["name"] = material_name + f"_light_{count+1}"
                new_mat["emission"] = L_val
                new_materials.append(new_mat)
            else:
                # If no base material found, create a new material with default values and the emission.
                new_mat = {
                    "name": material_name,
                    "type": "diffuse",
                    "reflectance": (0.0, 0.0, 0.0),
                    "transmittance": None,
                    "eta": None,
                    "k": None,
                    "roughness": None,
                    "uroughness": None,
                    "vroughness": None,
                    "remaproughness": None,
                    "displacement": None,
                    "normalmap": None,
                    "filename": None,
                    "scale": None,
                    "medium": "",
                    "tex_reflectance": "",
                    "tex_filename": "",
                    "emission": L_val
                }
                new_materials.append(new_mat)
    # Append any new materials created to the base list.
    materials_list.extend(new_materials)
    return materials_list

# -----------------------------------------------------------------------------
# Material Population Functions
# -----------------------------------------------------------------------------

def populate_materials_field(materials_list, materials_field):
    type_mapping = {
        "diffuse": 0,
        "dielectric": 1,
        "conductor": 2,
    }
    n = len(materials_list)
    for i in range(n):
        mat = materials_list[i]
        mat_type_str = (mat.get("type") or "").lower()
        material_type = type_mapping.get(mat_type_str, -1)
        refl = mat.get("reflectance")
        if refl is not None:
            try:
                color = vec3(*refl)
            except Exception:
                color = vec3(0.0, 0.0, 0.0)
        else:
            color = vec3(0.0, 0.0, 0.0)
        eta_list = mat.get("eta")
        try:
            eta_val = float(eta_list) if eta_list is not None else 1.0
        except Exception:
            eta_val = 1.0
        k_list = mat.get("k")
        if k_list is not None:
            try:
                k_val = vec3(*k_list)
            except Exception:
                k_val = vec3(0.0, 0.0, 0.0)
        else:
            k_val = vec3(0.0, 0.0, 0.0)
        try:
            uroughness = float(mat.get("uroughness", mat.get("roughness", 0.0)))
        except Exception:
            uroughness = 0.0
        try:
            vroughness = float(mat.get("vroughness", mat.get("roughness", 0.0)))
        except Exception:
            vroughness = 0.0
        remap_flag = mat.get("remaproughness")
        if remap_flag is not None:
            if (isinstance(remap_flag, bool) and remap_flag) or (isinstance(remap_flag, str) and remap_flag.lower() == "true"):
                uroughness = max(0.001, uroughness * uroughness)
                vroughness = max(0.001, vroughness * vroughness)
        # Extract emission using helper _extract_emission
        emission_val = _extract_emission(mat)
        if emission_val is not None:
            if isinstance(emission_val, (list, tuple)):
                try:
                    emission = vec3(*emission_val)
                except Exception:
                    emission = vec3(0.0, 0.0, 0.0)
            elif isinstance(emission_val, str):
                nums = extract_numbers(emission_val)
                if len(nums) >= 3:
                    try:
                        emission = vec3(float(nums[0]), float(nums[1]), float(nums[2]))
                    except Exception:
                        emission = vec3(0.0, 0.0, 0.0)
                else:
                    emission = vec3(0.0, 0.0, 0.0)
            else:
                emission = vec3(0.0, 0.0, 0.0)
        else:
            emission = vec3(0.0, 0.0, 0.0)
        materials_field[i] = Material(material_type, color, uroughness, vroughness, eta_val, k_val, emission)
    return materials_field

def build_materials_dict(materials_data):
    type_mapping = {"diffuse": 0, "dielectric": 1, "conductor": 2}
    mat_dict = {}
    for mat in materials_data:
        mat_type_str = (mat.get("type") or "").lower()
        material_type = type_mapping.get(mat_type_str, -1)
        refl = mat.get("reflectance")
        if refl is None:
            color = vec3(1.0, 1.0, 1.0)
        else:
            try:
                color = vec3(*refl)
            except Exception:
                color = vec3(1.0, 1.0, 1.0)
        eta_val = 1.0
        eta = mat.get("eta")
        if eta is not None:
            try:
                eta_val = float(eta[0])
            except Exception:
                try:
                    eta_val = float(eta)
                except Exception:
                    eta_val = 1.0
        k_val = vec3(0.0, 0.0, 0.0)
        k = mat.get("k")
        if k is not None:
            try:
                k_val = vec3(*k)
            except Exception:
                k_val = vec3(0.0, 0.0, 0.0)
        try:
            uroughness = float(mat.get("uroughness", mat.get("roughness", 0.0)))
        except Exception:
            uroughness = 0.0
        try:
            vroughness = float(mat.get("vroughness", mat.get("roughness", 0.0)))
        except Exception:
            vroughness = 0.0
        remap_flag = mat.get("remaproughness")
        if remap_flag is not None:
            if (isinstance(remap_flag, bool) and remap_flag) or (isinstance(remap_flag, str) and remap_flag.lower() == "true"):
                uroughness = max(0.001, uroughness * uroughness)
                vroughness = max(0.001, vroughness * vroughness)
        emission_val = _extract_emission(mat)
        if emission_val is not None:
            if isinstance(emission_val, (list, tuple)):
                try:
                    emission = vec3(*emission_val)
                except Exception:
                    emission = vec3(0.0, 0.0, 0.0)
            elif isinstance(emission_val, str):
                nums = extract_numbers(emission_val)
                if len(nums) >= 3:
                    try:
                        emission = vec3(float(nums[0]), float(nums[1]), float(nums[2]))
                    except Exception:
                        emission = vec3(0.0, 0.0, 0.0)
                else:
                    emission = vec3(0.0, 0.0, 0.0)
            else:
                emission = vec3(0.0, 0.0, 0.0)
        else:
            emission = vec3(0.0, 0.0, 0.0)
        mat_obj = Material(material_type, color, uroughness, vroughness, eta_val, k_val, emission)
        mat_dict[mat.get("name", "")] = mat_obj
    return mat_dict

# =============================================================================
# PBRT Parser for Shapes, Textures, and Mediums
# =============================================================================

def parse_pbrt_file(filename):
    shapes = []
    shape_counts_raw = {}
    textures = {}
    mediums = {}
    current_transform = None
    current_material = None
    global_medium = ""
    attr_depth = 0
    local_medium = ""
    local_medium_stack = []
    light_counter = 0
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        tokens = stripped.split()
        if not tokens:
            i += 1
            continue
        if tokens[0] == "AttributeBegin":
            local_medium_stack.append(local_medium)
            attr_depth += 1
            i += 1
            continue
        if tokens[0] == "AttributeEnd":
            attr_depth -= 1
            if local_medium_stack:
                local_medium = local_medium_stack.pop()
            else:
                local_medium = ""
            i += 1
            continue
        if stripped.startswith("Transform"):
            current_transform = parse_transform_line(stripped)
            i += 1
            continue
        if stripped.startswith("NamedMaterial"):
            if len(tokens) >= 2:
                current_material = tokens[1].strip('"')
            i += 1
            continue
        if stripped.startswith("MediumInterface"):
            if len(tokens) >= 2:
                medium_candidate = tokens[1].strip('"')
                if medium_candidate == "" and len(tokens) >= 3:
                    medium_candidate = tokens[2].strip('"')
                if attr_depth > 0:
                    local_medium = medium_candidate
                else:
                    global_medium = medium_candidate
            i += 1
            continue
        if stripped.startswith("MakeNamedMedium"):
            if len(tokens) < 2:
                i += 1
                continue
            medium_name = tokens[1].strip('"')
            medium_data = {"name": medium_name}
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('"'):
                acc_line, i = accumulate_property_line(lines, i)
                ptype, key, value = parse_property_line(acc_line)
                if key is not None:
                    medium_data[key] = value
            mediums[medium_name] = medium_data
            continue
        if stripped.startswith("Texture"):
            if len(tokens) < 2:
                i += 1
                continue
            tex_name = tokens[1].strip('"')
            tex_data = {"name": tex_name}
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('"'):
                acc_line, i = accumulate_property_line(lines, i)
                ptype, key, value = parse_property_line(acc_line)
                if key is not None:
                    tex_data[key] = value
            textures[tex_name] = tex_data
            continue
        if stripped.startswith("Shape"):
            if len(tokens) < 2:
                i += 1
                continue
            shape_type = tokens[1].strip('"')
            shape_counts_raw[shape_type] = shape_counts_raw.get(shape_type, 0) + 1
            medium_for_shape = local_medium if attr_depth > 0 else global_medium
            shape_data = {
                "shape": shape_type,
                "transform": current_transform,
                "material": current_material,
                "medium": medium_for_shape,
                "is_light": 0,
                "light_idx": -1
            }
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('"'):
                acc_line, i = accumulate_property_line(lines, i)
                ptype, key, value = parse_property_line(acc_line)
                if key is not None:
                    shape_data[key] = value
            if shape_data.get("material", "").lower() == "light":
                shape_data["is_light"] = 1
                shape_data["light_idx"] = light_counter
                light_counter += 1
            shapes.append(shape_data)
            continue
        i += 1
    actual_counts = {"triangles": 0, "spheres": 0}
    for shape in shapes:
        if shape["shape"] == "trianglemesh":
            indices = shape.get("indices")
            if isinstance(indices, list):
                ntri = len(indices) // 3
            else:
                ntri = 1
            actual_counts["triangles"] += ntri
        elif shape["shape"] == "sphere":
            actual_counts["spheres"] += 1
    return shapes, actual_counts, textures, mediums

# =============================================================================
# Helper Functions for Converting Values to 3-Tuples
# =============================================================================

def ensure_tuple3(val):
    if val is None:
        return (float('nan'), float('nan'), float('nan'))
    if isinstance(val, (tuple, list)):
        try:
            if len(val) >= 3:
                return tuple(float(x) for x in val[:3])
        except Exception:
            return (float('nan'), float('nan'), float('nan'))
    if isinstance(val, str):
        tokens = val.strip().split()
        if len(tokens) >= 3:
            try:
                return tuple(float(tok) for tok in tokens[:3])
            except Exception:
                return (float('nan'), float('nan'), float('nan'))
    try:
        f = float(val)
        return (f, f, f)
    except Exception:
        return (float('nan'), float('nan'), float('nan'))

def to_tuple3(val):
    if val is None:
        return (float('nan'), float('nan'), float('nan'))
    if isinstance(val, (tuple, list)):
        try:
            if len(val) >= 3:
                return tuple(float(x) for x in val[:3])
        except Exception:
            return (float('nan'), float('nan'), float('nan'))
    if isinstance(val, str):
        tokens = val.strip().split()
        if len(tokens) >= 3:
            try:
                return tuple(float(tok) for tok in tokens[:3])
            except Exception:
                return (float('nan'), float('nan'), float('nan'))
        else:
            try:
                f = float(val)
                return (f, f, f)
            except Exception:
                return (float('nan'), float('nan'), float('nan'))
    try:
        f = float(val)
        return (f, f, f)
    except Exception:
        return (float('nan'), float('nan'), float('nan'))

# =============================================================================
# NumPy Helpers (used during primitive construction)
# =============================================================================

def py_cross(a, b):
    return np.cross(np.array(a), np.array(b))

def py_normalize(v):
    arr = np.array(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return vec3(arr[0], arr[1], arr[2])

# =============================================================================
# Primitive Creation (using provided normals when available)
# =============================================================================

def create_primitives_field(shapes_list, materials_dict, prim_field):
    TRIANGLE_TYPE = 0
    SPHERE_TYPE = 1
    prims = []
    for shape in shapes_list:
        mat_name = shape.get("material", "")
        taichi_mat = materials_dict.get(mat_name,
                                        Material(-1, vec3(1.0, 1.0, 1.0), 0.0, 0.0,
                                                 1.0, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)))
        bsdf = BSDF()
        is_light = shape.get("is_light", 0)
        light_idx = shape.get("light_idx", -1)
        if shape["shape"] == "trianglemesh":
            P_val = shape.get("P")
            indices = shape.get("indices")
            if P_val is None or indices is None:
                continue
            try:
                vertices = []
                num_vertices = len(P_val) // 3
                for j in range(num_vertices):
                    vertices.append(vec3(P_val[3*j],
                                         P_val[3*j+1],
                                         P_val[3*j+2]))
            except Exception:
                continue
            normals_val = shape.get("N")
            normal_list = None
            if normals_val is not None and len(normals_val) >= 3:
                normal_list = []
                num_normals = len(normals_val) // 3
                for j in range(num_normals):
                    try:
                        n_raw = vec3(
                            normals_val[3*j],
                            normals_val[3*j+1],
                            normals_val[3*j+2]
                        )
                        n = py_normalize(n_raw)
                        normal_list.append(n)
                    except Exception as e:
                        print("Error parsing normal tokens:", normals_val[3*j:3*j+3], e)
                if len(normal_list) == 0:
                    normal_list = None
            T = shape.get("transform")
            if T is not None:
                def apply_transform(v):
                    x, y, z = v[0], v[1], v[2]
                    new_x = T[0] * x + T[4] * y + T[8] * z + T[12]
                    new_y = T[1] * x + T[5] * y + T[9] * z + T[13]
                    new_z = T[2] * x + T[6] * y + T[10] * z + T[14]
                    return vec3(new_x, new_y, new_z)
                vertices = [apply_transform(v) for v in vertices]
                if normal_list is not None:
                    def transform_normal(n):
                        new_x = T[0] * n[0] + T[4] * n[1] + T[8] * n[2]
                        new_y = T[1] * n[0] + T[5] * n[1] + T[9] * n[2]
                        new_z = T[2] * n[0] + T[6] * n[1] + T[10] * n[2]
                        return py_normalize(vec3(new_x, new_y, new_z))
                    normal_list = [transform_normal(n) for n in normal_list]
            if isinstance(indices, (list, tuple)):
                num_triangles = len(indices) // 3
            else:
                num_triangles = 1
            for tri_idx in range(num_triangles):
                if isinstance(indices, (list, tuple)):
                    idx0 = indices[3 * tri_idx]
                    idx1 = indices[3 * tri_idx + 1]
                    idx2 = indices[3 * tri_idx + 2]
                else:
                    idx0 = idx1 = idx2 = 0
                if idx0 >= len(vertices) or idx1 >= len(vertices) or idx2 >= len(vertices):
                    continue
                v1 = vertices[idx0]
                v2 = vertices[idx1]
                v3 = vertices[idx2]
                centroid = (v1 + v2 + v3) / 3.0
                edge_1 = v2 - v1
                edge_2 = v3 - v1
                if normal_list is not None:
                    try:
                        n1 = normal_list[idx0]
                        n2 = normal_list[idx1]
                        n3 = normal_list[idx2]
                        avg = vec3((n1[0] + n2[0] + n3[0]) / 3.0,
                                   (n1[1] + n2[1] + n3[1]) / 3.0,
                                   (n1[2] + n2[2] + n3[2]) / 3.0)
                        normal = py_normalize(avg)
                    except Exception as e:
                        print("Exception in parsing normals:", e)
                        normal = py_normalize(vec3(*py_cross([edge_2[0], edge_2[1], edge_2[2]],
                                                             [edge_1[0], edge_1[1], edge_1[2]])))
                else:
                    normal = py_normalize(vec3(*py_cross([edge_2[0], edge_2[1], edge_2[2]],
                                                         [edge_1[0], edge_1[1], edge_1[2]])))
                tri = Triangle(
                    vertex_1=v1, vertex_2=v2, vertex_3=v3,
                    centroid=centroid,
                    normal=normal,
                    edge_1=edge_1,
                    edge_2=edge_2
                )
                prim = Primitive(
                    shape_type=TRIANGLE_TYPE,
                    triangle=tri,
                    sphere=Sphere(vec3(0.0, 0.0, 0.0), 0.0),
                    material=taichi_mat,
                    bsdf=bsdf,
                    is_light=is_light,
                    light_idx=light_idx
                )
                prims.append(prim)
        elif shape["shape"] == "sphere":
            radius = shape.get("radius", 1.0)
            center = vec3(0.0, 0.0, 0.0)
            T = shape.get("transform")
            if T is not None:
                center = vec3(T[12], T[13], T[14])
            sph = Sphere(center=center, radius=radius)
            prim = Primitive(
                shape_type=SPHERE_TYPE,
                triangle=Triangle(vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0),
                                  vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)),
                sphere=sph,
                material=taichi_mat,
                bsdf=bsdf,
                is_light=is_light,
                light_idx=light_idx
            )
            prims.append(prim)
    for i, prim in enumerate(prims):
        prim_field[i] = prim
    return prim_field

# =============================================================================
# End of pbrt_parser.py
# =============================================================================
