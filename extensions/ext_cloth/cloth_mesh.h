#pragma once

#include "ext_cloth/cloth_components.h"
#include "ext_cloth/cloth_types.h"
#include <vector>

namespace ext_cloth {

struct ClothMeshData {
    std::vector<ClothPosition> positions;
    std::vector<ClothVelocity> velocities;
    std::vector<ClothMass> masses;
    std::vector<ClothEdge> edges;
    std::vector<ClothFace> faces;
    uint32 width = 0;   // columns
    uint32 height = 0;  // rows
};

// Generate a grid cloth mesh on the XZ plane at Y=height_offset.
// Pins top-left and top-right corners (inv_mass=0).
ClothMeshData GenerateGrid(uint32 width, uint32 height, float32 spacing,
                            float32 stiffness = 500.0f,
                            float32 height_offset = 3.0f);

}  // namespace ext_cloth
