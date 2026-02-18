#include "ext_cloth/cloth_mesh.h"
#include <cmath>

using namespace mps;

namespace ext_cloth {

ClothMeshData GenerateGrid(uint32 width, uint32 height, float32 spacing,
                            float32 stiffness, float32 height_offset) {
    ClothMeshData mesh;
    mesh.width = width;
    mesh.height = height;

    uint32 node_count = width * height;
    mesh.positions.resize(node_count);
    mesh.velocities.resize(node_count);
    mesh.masses.resize(node_count);

    // Center the grid on the XZ plane
    float32 offset_x = -static_cast<float32>(width - 1) * spacing * 0.5f;
    float32 offset_z = -static_cast<float32>(height - 1) * spacing * 0.5f;

    // Create nodes
    for (uint32 row = 0; row < height; ++row) {
        for (uint32 col = 0; col < width; ++col) {
            uint32 idx = row * width + col;

            mesh.positions[idx].x = offset_x + static_cast<float32>(col) * spacing;
            mesh.positions[idx].y = height_offset;
            mesh.positions[idx].z = offset_z + static_cast<float32>(row) * spacing;
            mesh.positions[idx].w = 0.0f;

            mesh.velocities[idx] = {};

            mesh.masses[idx].mass = 1.0f;
            mesh.masses[idx].inv_mass = 1.0f;
        }
    }

    // Pin top-left and top-right corners (first row)
    mesh.masses[0].inv_mass = 0.0f;
    mesh.masses[width - 1].inv_mass = 0.0f;

    // Create structural edges (horizontal + vertical)
    // Horizontal edges
    for (uint32 row = 0; row < height; ++row) {
        for (uint32 col = 0; col < width - 1; ++col) {
            uint32 n0 = row * width + col;
            uint32 n1 = row * width + col + 1;
            ClothEdge edge;
            edge.n0 = n0;
            edge.n1 = n1;
            edge.rest_length = spacing;
            edge.stiffness = stiffness;
            mesh.edges.push_back(edge);
        }
    }
    // Vertical edges
    for (uint32 row = 0; row < height - 1; ++row) {
        for (uint32 col = 0; col < width; ++col) {
            uint32 n0 = row * width + col;
            uint32 n1 = (row + 1) * width + col;
            ClothEdge edge;
            edge.n0 = n0;
            edge.n1 = n1;
            edge.rest_length = spacing;
            edge.stiffness = stiffness;
            mesh.edges.push_back(edge);
        }
    }

    // Create faces (2 triangles per cell, CCW winding when viewed from +Y)
    for (uint32 row = 0; row < height - 1; ++row) {
        for (uint32 col = 0; col < width - 1; ++col) {
            uint32 tl = row * width + col;
            uint32 tr = row * width + col + 1;
            uint32 bl = (row + 1) * width + col;
            uint32 br = (row + 1) * width + col + 1;

            // Triangle 1: tl -> bl -> tr
            ClothFace f1;
            f1.n0 = tl;
            f1.n1 = bl;
            f1.n2 = tr;
            mesh.faces.push_back(f1);

            // Triangle 2: tr -> bl -> br
            ClothFace f2;
            f2.n0 = tr;
            f2.n1 = bl;
            f2.n2 = br;
            mesh.faces.push_back(f2);
        }
    }

    return mesh;
}

}  // namespace ext_cloth
