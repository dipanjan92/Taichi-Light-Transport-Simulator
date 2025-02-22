# material_parser.py
import taichi as ti
from taichi.math import vec3

from base.materials import Material
from pbrt.parse_utils import to_vec3

# Define material type constants (you can expand these as needed)
DIFFUSE = 0
DIFFUSE_TRANSMISSION = 1
DIELECTRIC = 2
CONDUCTOR = 3


# Create functions for each material type.
def create_diffuse(mat_dict, emission=vec3(0.0)):
    # For a diffuse material, we typically use the "reflectance" parameter.
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    return Material(material_type=DIFFUSE,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=vec3(1.0),
                    k=vec3(0.0),
                    emission=emission)

def create_diffuse_transmission(mat_dict, emission=vec3(0.0)):
    # Diffuse transmission might combine a diffuse response with a transmission component.
    # For this example, we use the "reflectance" parameter for the transmitted color.
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    transmittance = to_vec3(mat_dict.get("transmittance", [1.0, 1.0, 1.0]))
    return Material(material_type=DIFFUSE_TRANSMISSION,
                    reflectance=reflectance,
                    transmittance=transmittance,
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=1.0,
                    k=to_vec3([0.0, 0.0, 0.0]),
                    emission=emission)

def create_dielectric(mat_dict, emission=vec3(0.0)):
    # For a dielectric material, we might use "reflectance" as color
    # and an "eta" parameter (index of refraction).
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    eta = to_vec3(mat_dict.get("eta", [1.0, 1.0, 1.0]))
    # Roughness is usually zero for an ideal dielectric.
    return Material(material_type=DIELECTRIC,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=0.0,
                    vroughness=0.0,
                    eta=vec3(eta[0]),
                    k=vec3(0.0),
                    emission=emission)

def create_conductor(mat_dict, emission=vec3(0.0)):
    # For conductor materials, you typically need roughness,
    # and complex refractive index components "eta" and "k".
    reflectance = to_vec3(mat_dict.get("reflectance", [1.0, 1.0, 1.0]))
    urough = float(mat_dict.get("uroughness", [0.0])[0])
    vrough = float(mat_dict.get("vroughness", [0.0])[0])
    eta = to_vec3(mat_dict.get("eta", [1.0, 1.0, 1.0]))
    k = to_vec3(mat_dict.get("k", [0.0, 0.0, 0.0]))
    return Material(material_type=CONDUCTOR,
                    reflectance=reflectance,
                    transmittance=vec3(0.0),
                    uroughness=urough,
                    vroughness=vrough,
                    eta=eta,
                    k=k,
                    emission=emission)

# A mapping from material type strings to create functions.
TYPE_MAP = {
    "diffuse": create_diffuse,
    "diffusetransmission": create_diffuse_transmission,
    "dielectric": create_dielectric,
    "conductor": create_conductor
}

# Main parser function: choose create function based on material type.
def parse_materials(material_list):
    # material_list is a list of material dictionaries (as shown in your example).
    parsed_materials = []
    for m in material_list:
        mat_props = m.get("properties", {})
        # Assume the type is provided as a list with one element, e.g., ['"diffuse"'].
        raw_type = mat_props.get("type", ["diffuse"])
        # Remove quotes if necessary.
        mat_type = raw_type[0].strip('"').lower() if isinstance(raw_type, list) and raw_type else "diffuse"
        create_fn = TYPE_MAP.get(mat_type, create_diffuse)
        parsed_materials.append(create_fn(mat_props))
    return parsed_materials

# Finally, create a Taichi field populated with these materials.
def create_material_field(material_list, material_field):
    parsed_materials = parse_materials(material_list)
    for i, mat in enumerate(parsed_materials):
        material_field[i] = mat
    return material_field

def create_material_by_name(material_list, target_name):
    """
    Given a list of material dictionaries and a material name, search for the
    material with that name, create it using the appropriate create function, and return it.
    Raises a ValueError if no material with the specified name is found.
    """
    for m in material_list:
        if m.get("name") == target_name:
            mat_props = m.get("properties", {})
            raw_type = mat_props.get("type", ["diffuse"])
            mat_type = raw_type[0].strip('"').lower() if isinstance(raw_type, list) and raw_type else "diffuse"
            create_fn = TYPE_MAP.get(mat_type, create_diffuse)
            return create_fn(mat_props)
    raise ValueError(f"Material with name '{target_name}' not found.")

