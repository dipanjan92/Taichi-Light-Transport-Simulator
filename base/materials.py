import taichi as ti
from taichi.math import pi, vec3, length

from base.bsdf import BSDF


@ti.dataclass
class Material:
    face_idx: ti.i32
    diffuse: vec3
    ambient: vec3
    specular: vec3
    emission: vec3
    shininess: ti.f32
    ior: ti.f32
    opacity: ti.f32
    illum: ti.i32
    is_light: ti.i32

    @ti.func
    def create_bsdf_from_material(self):
        bsdf = BSDF()
        bsdf.n_bxdfs = 0

        # Add diffuse component if the material has a non-black diffuse color
        if self.diffuse[0] != 0 or self.diffuse[1] != 0 or self.diffuse[2] != 0 and self.illum == 1:
            bsdf.add_diffuse(self.diffuse)

        # Add specular component if the material has a non-black specular color
        if self.specular[0] != 0 or self.specular[1] != 0 or self.specular[2] != 0 and self.illum == 5:
            # silver is hardcoded
            eta = vec3(0.14, 0.16, 0.13)
            k = vec3(4.1, 3.8, 3.6)
            roughness = ti.max(0.0, 1.0 - self.shininess / 100.0)  # Convert shininess to roughness
            bsdf.add_conductor(eta, k, roughness)

        # Add transmission component if the material is partially transparent
        if self.opacity < 1.0 and self.illum == 6:
            transmission = vec3(
                (1.0 - self.opacity) * self.diffuse[0],
                (1.0 - self.opacity) * self.diffuse[1],
                (1.0 - self.opacity) * self.diffuse[2]
            )
            print(self.diffuse, transmission)
            bsdf.add_transmission(self.diffuse, transmission)

        # Add dielectric component if the material has a different IOR
        if self.ior != 1.0 and self.illum == 7:
            eta = self.ior
            roughness = ti.max(0.0, 1.0 - self.shininess / 100.0)
            bsdf.add_dielectric(eta, roughness)  # Assuming smooth surface

        # Handle light
        # if self.emissive[0] != 0 or self.emissive[1] != 0 or self.emissive[2] != 0:
        #     # Emissive materials typically don't scatter light, but emit it
        #     # You might want to handle this differently depending on your renderer
        #     pass

        return bsdf
