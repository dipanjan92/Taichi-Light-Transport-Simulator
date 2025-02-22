import taichi as ti
from taichi.math import vec3
from pbrt.parse_utils import py_cross, py_normalize, to_vec3, set_matrix, multiply_matrix4
from pbrt.parser import pbrt_to_dict, IDENTITY_4x4
from primitives.primitives import Triangle, Primitive, Sphere
from pbrt.transformations import parse_transform  # Expects a list of transformation blocks.
from pbrt.material_parser import create_material_by_name, create_diffuse
from base.materials import Material
from base.bsdf import BSDF

TRIANGLE_TYPE = 0
SPHERE_TYPE = 1


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def apply_transform(vertices, T):
    """
    Apply a 4x4 transform T (as a flat tuple of 16 numbers) to a list of vec3 vertices.
    """

    def transform(v):
        x, y, z = v[0], v[1], v[2]
        new_x = T[0] * x + T[4] * y + T[8] * z + T[12]
        new_y = T[1] * x + T[5] * y + T[9] * z + T[13]
        new_z = T[2] * x + T[6] * y + T[10] * z + T[14]
        return vec3(new_x, new_y, new_z)

    return [transform(v) for v in vertices]


def build_triangle(v1, v2, v3, normal_list, idx0, idx1, idx2):
    """
    Build a Triangle object from three vertices.
    If per-vertex normals are available, average them to compute the triangle normal;
    otherwise, compute the cross product of the edges.
    """
    centroid = (v1 + v2 + v3) / 3.0
    edge_1 = v2 - v1
    edge_2 = v3 - v1
    if normal_list is not None and len(normal_list) > max(idx0, idx1, idx2):
        n1 = normal_list[idx0]
        n2 = normal_list[idx1]
        n3 = normal_list[idx2]
        avg = vec3((n1[0] + n2[0] + n3[0]) / 3.0,
                   (n1[1] + n2[1] + n3[1]) / 3.0,
                   (n1[2] + n2[2] + n3[2]) / 3.0)
        normal = py_normalize(avg)
    else:
        normal = py_normalize(vec3(*py_cross([edge_2[0], edge_2[1], edge_2[2]],
                                             [edge_1[0], edge_1[1], edge_1[2]])))
    return Triangle(
        vertex_1=v1,
        vertex_2=v2,
        vertex_3=v3,
        centroid=centroid,
        normal=normal,
        edge_1=edge_1,
        edge_2=edge_2
    )


# -----------------------------------------------------------------------------
# Create Functions for Specific Shapes
# -----------------------------------------------------------------------------
def create_triangle_primitives(shape_data, material, bsdf, is_light, light_idx):
    """
    Parse a 'trianglemesh' shape and return a list of Primitive objects (one per triangle).
    """
    primitives_list = []
    P_val = shape_data["properties"].get("P")
    if P_val is None:
        return primitives_list

    # Build vertex list from the flattened position array.
    vertices = []
    num_vertices = len(P_val) // 3
    for i in range(num_vertices):
        vertices.append(vec3(P_val[3 * i], P_val[3 * i + 1], P_val[3 * i + 2]))

    # Apply transform if provided.
    T = shape_data.get("transform")
    if T is not None:
        vertices = apply_transform(vertices, T)

    # Process normals if available.
    normals_val = shape_data["properties"].get("N")
    normal_list = None
    if normals_val is not None and len(normals_val) >= 3:
        normal_list = []
        num_normals = len(normals_val) // 3
        for i in range(num_normals):
            n = vec3(normals_val[3 * i], normals_val[3 * i + 1], normals_val[3 * i + 2])
            normal_list.append(py_normalize(n))
        if T is not None:
            # Transform normals (ignoring translation).
            def transform_normal(n):
                new_x = T[0] * n[0] + T[4] * n[1] + T[8] * n[2]
                new_y = T[1] * n[0] + T[5] * n[1] + T[9] * n[2]
                new_z = T[2] * n[0] + T[6] * n[1] + T[10] * n[2]
                return py_normalize(vec3(new_x, new_y, new_z))

            normal_list = [transform_normal(n) for n in normal_list]

    indices = shape_data["properties"].get("indices")
    if indices is not None:
        # Convert indices to integers.
        indices = [int(i) for i in indices]

    if indices is None:
        if len(vertices) >= 3:
            tri = build_triangle(vertices[0], vertices[1], vertices[2], normal_list, 0, 1, 2)
            if is_light:
                l_idx = light_idx[0]
                light_idx[0] += 1
            else:
                l_idx = -1
            prim = Primitive(
                shape_type=TRIANGLE_TYPE,
                triangle=tri,
                sphere=Sphere(vec3(0.0, 0.0, 0.0), 0.0),
                material=material,
                bsdf=bsdf,
                is_light=is_light,
                light_idx=l_idx
            )
            primitives_list.append(prim)
    else:
        num_triangles = len(indices) // 3 if isinstance(indices, list) else 1
        for tri_idx in range(num_triangles):
            if isinstance(indices, list):
                idx0 = indices[3 * tri_idx]
                idx1 = indices[3 * tri_idx + 1]
                idx2 = indices[3 * tri_idx + 2]
            else:
                idx0 = idx1 = idx2 = 0
            if idx0 >= len(vertices) or idx1 >= len(vertices) or idx2 >= len(vertices):
                continue
            tri = build_triangle(vertices[idx0], vertices[idx1], vertices[idx2],
                                 normal_list, idx0, idx1, idx2)
            if is_light:
                l_idx = light_idx[0]
                light_idx[0] += 1
            else:
                l_idx = -1
            prim = Primitive(
                shape_type=TRIANGLE_TYPE,
                triangle=tri,
                sphere=Sphere(vec3(0.0, 0.0, 0.0), 0.0),
                material=material,
                bsdf=bsdf,
                is_light=is_light,
                light_idx=l_idx
            )
            primitives_list.append(prim)
    return primitives_list


def create_sphere_primitive(shape_data, material, bsdf, is_light, light_idx, global_transform):
    """
    Parse a sphere shape and return a Primitive, properly handling transformations.
    """
    # Get base radius
    radius = shape_data["properties"].get("radius", [1.0])[0]

    # Start with center at the origin
    center = vec3(0.0, 0.0, 0.0)

    # Get transformation matrix
    local_transform = shape_data.get("transform", IDENTITY_4x4)

    # Combine global and local transforms
    T = multiply_matrix4(global_transform, local_transform)

    # Apply full transformation to center
    center = vec3(
        T[0] * 0.0 + T[4] * 0.0 + T[8] * 0.0 + T[12],
        T[1] * 0.0 + T[5] * 0.0 + T[9] * 0.0 + T[13],
        T[2] * 0.0 + T[6] * 0.0 + T[10] * 0.0 + T[14]
    )

    # Compute scale factor from transformation matrix
    scale_x = vec3(T[0], T[1], T[2]).norm()
    scale_y = vec3(T[4], T[5], T[6]).norm()
    scale_z = vec3(T[8], T[9], T[10]).norm()

    # Adjust radius based on maximum scale factor
    radius *= max(scale_x, scale_y, scale_z)

    # Create sphere primitive
    sphere = Sphere(center=center, radius=radius)

    # Handle light indexing
    l_idx = light_idx[0] if is_light else -1
    if is_light:
        light_idx[0] += 1

    # Create and return primitive
    return Primitive(
        shape_type=SPHERE_TYPE,
        triangle=Triangle(
            vertex_1=vec3(0, 0, 0),
            vertex_2=vec3(0, 0, 0),
            vertex_3=vec3(0, 0, 0),
            centroid=vec3(0, 0, 0),
            normal=vec3(0, 0, 0),
            edge_1=vec3(0, 0, 0),
            edge_2=vec3(0, 0, 0)
        ),
        sphere=sphere,
        material=material,
        bsdf=bsdf,
        is_light=is_light,
        light_idx=l_idx
    )



def parse_shapes(shapes_list, materials_list, primitives, global_transform):
    is_light = 0
    light_count = [0]
    global_index = 0  # A running counter for primitives

    for i in range(len(shapes_list)):
        shape = shapes_list[i]
        mat_name = shape["material"]
        material = create_material_by_name(materials_list, mat_name)
        emission = shape["emission"]
        if emission is not None:
            try:
                if isinstance(emission, (list, tuple)):
                    if sum(abs(float(x)) for x in emission) > 0:
                        is_light = 1
                        material.emission = to_vec3(emission)
                else:
                    if abs(float(emission)) > 0:
                        is_light = 1
                        material.emission = to_vec3([emission, emission, emission])
            except Exception:
                pass
        bsdf = BSDF()
        shape_type = shape["shape_type"]
        if shape_type == "trianglemesh":
            prims = create_triangle_primitives(shape, material, bsdf, is_light, light_count)
            for j in range(len(prims)):
                primitives[global_index] = prims[j]
                # print(mat_name, material.reflectance, primitives[global_index].material.reflectance)
                global_index += 1
        elif shape_type == "sphere":
            prim = create_sphere_primitive(shape, material, bsdf, is_light, light_count, global_transform)
            primitives[global_index] = prim
            # print(mat_name, material.reflectance, primitives[global_index].material.reflectance)
            global_index += 1

    # print("*" * 30)
    # for i in range(global_index):
    #     print(primitives[i].material.reflectance)

    return global_index, light_count[0]



def gather_shapes(blocks, inherited_transform=IDENTITY_4x4, inherited_material=None, inherited_emission=None):
    """
    Recursively traverse the list of blocks to extract Shape blocks.
    Each shape gets:
      - shape_type: from the block’s "name" (e.g. "sphere" or "trianglemesh")
      - material: inherited from an enclosing NamedMaterial (or None)
      - transform: the most recent transform in scope (inherited or set locally)
      - emission: inherited from an enclosing AreaLightSource (if any)
      - properties: the shape’s own properties
    """
    shapes = []
    current_transform = inherited_transform
    current_material = inherited_material
    current_emission = inherited_emission

    for block in blocks:
        btype = block.get("type")
        if btype == "Transform":
            # Update the transform context.
            current_transform = block["properties"].get("matrix", IDENTITY_4x4)
        elif btype == "AreaLightSource":
            # Update the emission context. (Assuming emission is given in property "L".)
            # print(block["properties"])
            current_emission = block["properties"].get("L", inherited_emission)
            # Note: We don't add an AreaLightSource as a shape.
        elif btype == "NamedMaterial":
            # Update material context.
            current_material = block.get("name")
            # Process any children (e.g. Shape blocks) under this material.
            if "children" in block:
                shapes.extend(gather_shapes(block["children"], current_transform, current_material, current_emission))
        elif btype == "Attribute":
            # Recurse into an Attribute block.
            if "children" in block:
                shapes.extend(gather_shapes(block["children"], current_transform, current_material, current_emission))
        elif btype == "Shape":
            # Found a shape block; record its info.
            shape_info = {
                "shape_type": block.get("name"),  # e.g. "sphere" or "trianglemesh"
                "material": current_material,
                "transform": current_transform,
                "emission": current_emission,
                "properties": block.get("properties", {})
            }
            shapes.append(shape_info)
        else:
            # For any other block that contains children, process them.
            if "children" in block:
                shapes.extend(gather_shapes(block["children"], current_transform, current_material, current_emission))
    return shapes


def extract_all_shapes(filename):
    scene_dict = pbrt_to_dict(filename)

    # Determine the global transform from a top-level Transform directive if it exists.
    global_transform = IDENTITY_4x4
    if "Transform" in scene_dict and scene_dict["Transform"]:
        global_transform = scene_dict["Transform"][0]["properties"].get("matrix", IDENTITY_4x4)

    # Look for a default material in MakeNamedMaterial (use "Null" if available).
    default_material = None
    if "MakeNamedMaterial" in scene_dict:
        for mat in scene_dict["MakeNamedMaterial"]:
            if mat.get("name") == "Null":
                default_material = "Null"
                break

    shapes = []
    # Process top-level blocks that may contain shapes.
    for key in ["NamedMaterial", "Attribute", "Shape"]:
        if key in scene_dict:
            for block in scene_dict[key]:
                shapes.extend(gather_shapes([block], inherited_transform=global_transform))

    # If a shape did not inherit a material, assign the default material (if defined).
    for shape in shapes:
        if shape.get("material") is None and default_material is not None:
            shape["material"] = default_material

    return shapes
