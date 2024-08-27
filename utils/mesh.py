import numba
import numpy as np
import pywavefront
from numba import njit, prange
from collections import defaultdict

# Define the dtype for the material data
material_dtype = np.dtype([
    ('face_idx', np.int32),
    ('diffuse', np.float32, (3,)),
    ('ambient', np.float32, (3,)),
    ('specular', np.float32, (3,)),
    ('emission', np.float32, (3,)),
    ('shininess', np.float32),
    ('ior', np.float32),
    ('opacity', np.float32),
    ('illum', np.int32),
    ('is_light', np.int32)
])


@njit
def normalize_color(color):
    """Normalize color values to [0, 1] range."""
    return np.clip(np.array(color) / 1.0, 0, 1)


def parse_obj_file(obj_file_path):
    vertices = []
    faces_by_material = defaultdict(list)
    current_material = None
    face_counter = defaultdict(int)

    with open(obj_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                if len(face) == 4:  # Convert quad to two triangles
                    faces_by_material[current_material].append(
                        [face_counter[current_material], face[0], face[1], face[2]])
                    face_counter[current_material] += 1
                    faces_by_material[current_material].append(
                        [face_counter[current_material], face[0], face[2], face[3]])
                else:
                    faces_by_material[current_material].append(
                        [face_counter[current_material], face[0], face[1], face[2]])
                face_counter[current_material] += 1
            elif parts[0] == 'usemtl':
                current_material = parts[1]

    vertices = np.array(vertices)
    return vertices, faces_by_material


@njit(parallel=True, fastmath=True)
def process_faces_numba(material_data, faces, face_start_idx, diffuse, ambient, specular, emission, shininess, ior,
                        opacity, illum):
    """Process faces and assign material properties using Numba-compatible types."""
    for i in prange(len(faces)):
        face_idx = face_start_idx + i
        data = material_data[face_idx]
        data['face_idx'] = faces[i, 0]  # Index of the face
        data['diffuse'] = diffuse
        data['ambient'] = ambient
        data['specular'] = specular
        data['emission'] = emission
        data['shininess'] = shininess
        data['ior'] = ior
        data['opacity'] = opacity
        data['illum'] = illum
        data['is_light'] = 1 if np.any(emission > 0) else 0


def create_material_data_numba(faces_by_material, materials):
    total_faces = sum(len(faces) for faces in faces_by_material.values())
    material_data = np.zeros(total_faces, dtype=material_dtype)

    index = 0
    for material, faces in faces_by_material.items():
        faces = np.array(faces)  # Convert to NumPy array
        mat_props = materials.get(material, None)

        if mat_props:
            process_faces_numba(
                material_data, faces, index,
                np.array(mat_props.diffuse[:3], dtype=np.float32),
                np.array(mat_props.ambient[:3], dtype=np.float32),
                np.array(mat_props.specular[:3], dtype=np.float32),
                np.array(mat_props.emissive[:3], dtype=np.float32),
                mat_props.shininess,
                mat_props.optical_density,
                mat_props.transparency,
                mat_props.illumination_model
            )
        else:
            process_faces_numba(
                material_data, faces, index,
                np.array([1, 1, 1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                0.0, 1.0, 1.0, 1
            )
        index += len(faces)

    return material_data


def parse_scene(obj_file_path):
    # Load materials from MTL file
    scene = pywavefront.Wavefront(obj_file_path, collect_faces=True)
    materials = {name: material for name, material in scene.materials.items()}

    # Parse OBJ file to get vertices and faces
    vertices, faces_by_material = parse_obj_file(obj_file_path)

    # Create material data array using the optimized function
    material_data = create_material_data_numba(faces_by_material, materials)

    # Prepare faces array for further processing or visualization
    total_faces = sum(len(faces) for faces in faces_by_material.values())
    faces = np.zeros(total_faces * 4, dtype=np.int32)  # 4 elements per face: 1 for count, 3 for vertices

    index = 0
    for face_group in faces_by_material.values():
        for face_info in face_group:
            faces[index] = 3  # Number of vertices in the face (triangle)
            faces[index + 1:index + 4] = face_info[1:]  # The vertices
            index += 4

    # Now 'vertices', 'faces', and 'material_data' are ready for use
    return vertices, faces, material_data


def extract_material_data_from_mesh(triangulated_mesh):
    # Number of faces in the triangulated mesh
    num_faces = triangulated_mesh.n_cells

    # Create an empty structured array with the appropriate dtype
    extracted_material_data = np.zeros(num_faces, dtype=material_dtype)

    # Fill the structured array with the material data from the mesh's cell_data
    extracted_material_data['face_idx'] = np.arange(num_faces)
    extracted_material_data['diffuse'] = triangulated_mesh.cell_data['diffuse']
    extracted_material_data['ambient'] = triangulated_mesh.cell_data['ambient']
    extracted_material_data['specular'] = triangulated_mesh.cell_data['specular']
    extracted_material_data['emission'] = triangulated_mesh.cell_data['emission']
    extracted_material_data['shininess'] = triangulated_mesh.cell_data['shininess']
    extracted_material_data['ior'] = triangulated_mesh.cell_data['ior']
    extracted_material_data['opacity'] = triangulated_mesh.cell_data['opacity']
    extracted_material_data['illum'] = triangulated_mesh.cell_data['illum']
    extracted_material_data['is_light'] = triangulated_mesh.cell_data['is_light']

    return extracted_material_data







# import numpy as np
# import pywavefront
# from collections import defaultdict
#
# # Define the dtype for the material data
# material_dtype = np.dtype([
#     ('face_idx', np.int32),
#     ('diffuse', np.float32, (3,)),
#     ('ambient', np.float32, (3,)),
#     ('specular', np.float32, (3,)),
#     ('emission', np.float32, (3,)),
#     ('shininess', np.float32),
#     ('ior', np.float32),
#     ('opacity', np.float32),
#     ('illum', np.int32),
#     ('is_light', np.int32)
# ])
#
#
# def normalize_color(color):
#     """Normalize color values to [0, 1] range."""
#     return np.clip(np.array(color) / 1.0, 0, 1)
#
#
# def parse_obj_file(obj_file_path):
#     vertices = []
#     faces_by_material = defaultdict(list)
#     current_material = None
#     face_counter = defaultdict(int)  # To keep track of face indices per material
#
#     with open(obj_file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#
#             if parts[0] == 'v':
#                 vertices.append([float(x) for x in parts[1:4]])
#             elif parts[0] == 'f':
#                 face = []
#                 for part in parts[1:]:
#                     vertex_index = int(part.split('/')[0])
#                     if vertex_index < 0:
#                         vertex_index += len(vertices)
#                     else:
#                         vertex_index -= 1
#                     face.append(vertex_index)
#
#                 # Check if the face is a quad and split into triangles
#                 if len(face) == 4:
#                     faces_by_material[current_material].append({
#                         'face_idx': face_counter[current_material],
#                         'vertices': [face[0], face[1], face[2]]
#                     })
#                     face_counter[current_material] += 1
#                     faces_by_material[current_material].append({
#                         'face_idx': face_counter[current_material],
#                         'vertices': [face[0], face[2], face[3]]
#                     })
#                 else:
#                     faces_by_material[current_material].append({
#                         'face_idx': face_counter[current_material],
#                         'vertices': face
#                     })
#                 face_counter[current_material] += 1
#
#             elif parts[0] == 'usemtl':
#                 current_material = parts[1]
#
#     vertices = np.array(vertices)
#     return vertices, faces_by_material
#
#
# def create_material_data(faces_by_material, materials):
#     # Calculate the total number of faces
#     total_faces = sum(len(faces) for faces in faces_by_material.values())
#
#     # Initialize the material_data array
#     material_data = np.zeros(total_faces, dtype=material_dtype)
#
#     index = 0
#     for material, faces in faces_by_material.items():
#         mat_props = materials.get(material, None)
#
#         for face_info in faces:
#             face_idx = face_info['face_idx']
#
#             material_data[index]['face_idx'] = face_idx
#
#             # Set material properties
#             if mat_props:
#                 material_data[index]['diffuse'] = normalize_color(mat_props.diffuse[:3])
#                 material_data[index]['ambient'] = normalize_color(mat_props.ambient[:3])
#                 material_data[index]['specular'] = normalize_color(mat_props.specular[:3])
#
#                 # Ensure the emissive color is handled correctly
#                 emissive_color = np.array(mat_props.emissive[:3])
#                 material_data[index]['emission'] = normalize_color(emissive_color)
#                 material_data[index]['shininess'] = mat_props.shininess
#                 material_data[index]['ior'] = mat_props.optical_density
#                 material_data[index]['opacity'] = mat_props.transparency
#                 material_data[index]['illum'] = mat_props.illumination_model
#                 material_data[index]['is_light'] = 1 if np.any(emissive_color > 0) else 0
#             else:
#                 # Default values if no material found
#                 material_data[index]['diffuse'] = np.array([1, 1, 1], dtype=np.float32)
#                 material_data[index]['ambient'] = np.array([1, 1, 1], dtype=np.float32)
#                 material_data[index]['specular'] = np.array([0, 0, 0], dtype=np.float32)
#                 material_data[index]['emission'] = np.array([0, 0, 0], dtype=np.float32)
#                 material_data[index]['shininess'] = 0.0
#                 material_data[index]['ior'] = 1.0
#                 material_data[index]['opacity'] = 1.0
#                 material_data[index]['illum'] = 1
#                 material_data[index]['is_light'] = 0
#
#             index += 1
#
#     return material_data
#
#
# def parse_scene(obj_file_path):
#     # Load materials from MTL file
#     scene = pywavefront.Wavefront(obj_file_path, collect_faces=True)
#     materials = {name: material for name, material in scene.materials.items()}
#
#     # Parse OBJ file
#     vertices, faces_by_material = parse_obj_file(obj_file_path)
#
#     # Create material data array
#     material_data = create_material_data(faces_by_material, materials)
#
#     # Prepare faces array for PyVista
#     total_faces = sum(len(faces) for faces in faces_by_material.values())
#     faces = np.zeros(total_faces * 4, dtype=np.int32)  # 4 elements per face: 1 for count, 3 for vertices
#
#     index = 0
#     for face in faces_by_material.values():
#         for face_info in face:
#             faces[index] = 3  # Number of vertices in the face (triangle)
#             faces[index + 1:index + 4] = face_info['vertices']
#             index += 4
#
#     return vertices, faces, material_data
#
#
# def extract_material_data_from_mesh(triangulated_mesh):
#     # Number of faces in the triangulated mesh
#     num_faces = triangulated_mesh.n_cells
#
#     # Create an empty structured array with the appropriate dtype
#     extracted_material_data = np.zeros(num_faces, dtype=material_dtype)
#
#     # Fill the structured array with the material data from the mesh's cell_data
#     extracted_material_data['face_idx'] = np.arange(num_faces)
#     extracted_material_data['diffuse'] = triangulated_mesh.cell_data['diffuse']
#     extracted_material_data['ambient'] = triangulated_mesh.cell_data['ambient']
#     extracted_material_data['specular'] = triangulated_mesh.cell_data['specular']
#     extracted_material_data['emission'] = triangulated_mesh.cell_data['emission']
#     extracted_material_data['shininess'] = triangulated_mesh.cell_data['shininess']
#     extracted_material_data['ior'] = triangulated_mesh.cell_data['ior']
#     extracted_material_data['opacity'] = triangulated_mesh.cell_data['opacity']
#     extracted_material_data['illum'] = triangulated_mesh.cell_data['illum']
#     extracted_material_data['is_light'] = triangulated_mesh.cell_data['is_light']
#
#     return extracted_material_data
