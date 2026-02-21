#include "ext_mesh/mesh_generator.h"
#include "ext_mesh/mesh_types.h"
#include "ext_mesh/mesh_component.h"
#include "core_gpu/asset_path.h"
#include "core_database/database.h"
#include "core_simulate/sim_components.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace mps;
using namespace mps::simulate;
using namespace mps::database;

namespace ext_mesh {

MeshResult CreateGrid(Database& db,
                      uint32 width, uint32 height, float32 spacing,
                      float32 height_offset) {
    uint32 node_count = width * height;

    std::vector<SimPosition> positions(node_count);
    std::vector<SimVelocity> velocities(node_count);
    std::vector<SimMass> masses(node_count);

    // Center the grid on the XZ plane
    float32 offset_x = -static_cast<float32>(width - 1) * spacing * 0.5f;
    float32 offset_z = -static_cast<float32>(height - 1) * spacing * 0.5f;

    constexpr float32 surface_density = 100.0f;
    float32 node_mass = surface_density * spacing * spacing;

    for (uint32 row = 0; row < height; ++row) {
        for (uint32 col = 0; col < width; ++col) {
            uint32 idx = row * width + col;

            positions[idx].x = offset_x + static_cast<float32>(col) * spacing;
            positions[idx].y = height_offset;
            positions[idx].z = offset_z + static_cast<float32>(row) * spacing;
            positions[idx].w = 0.0f;

            velocities[idx] = {};

            masses[idx].mass = node_mass;
            masses[idx].inv_mass = 1.0f / node_mass;
        }
    }

    // Generate faces: 2 triangles per cell (CCW winding from +Y)
    uint32 face_count = (width - 1) * (height - 1) * 2;
    std::vector<MeshFace> faces;
    faces.reserve(face_count);

    for (uint32 row = 0; row < height - 1; ++row) {
        for (uint32 col = 0; col < width - 1; ++col) {
            uint32 tl = row * width + col;
            uint32 tr = row * width + col + 1;
            uint32 bl = (row + 1) * width + col;
            uint32 br = (row + 1) * width + col + 1;

            MeshFace f1;
            f1.n0 = tl;
            f1.n1 = bl;
            f1.n2 = tr;
            faces.push_back(f1);

            MeshFace f2;
            f2.n0 = tr;
            f2.n1 = bl;
            f2.n2 = br;
            faces.push_back(f2);
        }
    }

    // Create entity with mesh data
    Entity mesh_e = db.CreateEntity();

    MeshComponent mesh_comp{};
    mesh_comp.vertex_count = node_count;
    mesh_comp.face_count = face_count;
    mesh_comp.edge_count = 0;
    db.AddComponent<MeshComponent>(mesh_e, mesh_comp);

    db.SetArray<SimPosition>(mesh_e, std::move(positions));
    db.SetArray<SimVelocity>(mesh_e, std::move(velocities));
    db.SetArray<SimMass>(mesh_e, std::move(masses));
    db.SetArray<MeshFace>(mesh_e, std::move(faces));

    MeshResult result;
    result.mesh_entity = mesh_e;
    result.node_count = node_count;
    result.face_count = face_count;
    return result;
}

// Parse a vertex index from an OBJ face token ("v", "v/vt", "v/vt/vn", "v//vn")
static uint32 ParseVertexIndex(const std::string& token, uint32 vertex_count) {
    auto slash_pos = token.find('/');
    std::string v_str = (slash_pos != std::string::npos) ? token.substr(0, slash_pos) : token;
    int idx = std::stoi(v_str);
    if (idx < 0) {
        return static_cast<uint32>(static_cast<int>(vertex_count) + idx);
    }
    return static_cast<uint32>(idx - 1);
}

MeshResult ImportOBJ(Database& db, const std::string& filepath, float32 scale) {
    MeshResult result;

    auto full_path = gpu::ResolveAssetPath("objs/" + filepath);
    std::ifstream file(full_path);
    if (!file.is_open()) {
        return result;
    }

    std::vector<SimPosition> positions;
    std::vector<MeshFace> faces;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float32 x = 0.0f, y = 0.0f, z = 0.0f;
            iss >> x >> y >> z;
            SimPosition pos;
            pos.x = x * scale;
            pos.y = y * scale;
            pos.z = z * scale;
            pos.w = 0.0f;
            positions.push_back(pos);
        } else if (prefix == "f") {
            std::vector<uint32> indices;
            std::string token;
            while (iss >> token) {
                indices.push_back(ParseVertexIndex(token, static_cast<uint32>(positions.size())));
            }

            // Triangulate: first triangle + fan for quads/polygons
            for (uint32 i = 1; i + 1 < static_cast<uint32>(indices.size()); ++i) {
                MeshFace face;
                face.n0 = indices[0];
                face.n1 = indices[i];
                face.n2 = indices[i + 1];
                faces.push_back(face);
            }
        }
        // Skip vn, vt, mtllib, usemtl, g, o, s, and other lines
    }

    if (positions.empty() || faces.empty()) {
        return result;
    }

    uint32 node_count = static_cast<uint32>(positions.size());
    uint32 face_count = static_cast<uint32>(faces.size());

    // Compute area-weighted mass per vertex
    constexpr float32 surface_density = 100.0f;
    std::vector<SimMass> masses(node_count);
    std::vector<float32> vertex_area(node_count, 0.0f);

    for (const auto& face : faces) {
        const auto& p0 = positions[face.n0];
        const auto& p1 = positions[face.n1];
        const auto& p2 = positions[face.n2];

        float32 e1x = p1.x - p0.x, e1y = p1.y - p0.y, e1z = p1.z - p0.z;
        float32 e2x = p2.x - p0.x, e2y = p2.y - p0.y, e2z = p2.z - p0.z;

        float32 cx = e1y * e2z - e1z * e2y;
        float32 cy = e1z * e2x - e1x * e2z;
        float32 cz = e1x * e2y - e1y * e2x;
        float32 tri_area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);

        float32 share = tri_area / 3.0f;
        vertex_area[face.n0] += share;
        vertex_area[face.n1] += share;
        vertex_area[face.n2] += share;
    }

    for (uint32 i = 0; i < node_count; ++i) {
        float32 m = surface_density * vertex_area[i];
        if (m < 1e-12f) m = 1e-6f;  // prevent zero mass
        masses[i].mass = m;
        masses[i].inv_mass = 1.0f / m;
    }

    // Zero velocities
    std::vector<SimVelocity> velocities(node_count);

    // Create entity with mesh data
    Entity mesh_e = db.CreateEntity();

    MeshComponent mesh_comp{};
    mesh_comp.vertex_count = node_count;
    mesh_comp.face_count = face_count;
    mesh_comp.edge_count = 0;
    db.AddComponent<MeshComponent>(mesh_e, mesh_comp);

    db.SetArray<SimPosition>(mesh_e, std::move(positions));
    db.SetArray<SimVelocity>(mesh_e, std::move(velocities));
    db.SetArray<SimMass>(mesh_e, std::move(masses));
    db.SetArray<MeshFace>(mesh_e, std::move(faces));

    result.mesh_entity = mesh_e;
    result.node_count = node_count;
    result.face_count = face_count;
    return result;
}

void PinVertices(Database& db, Entity mesh_entity,
                 const std::vector<uint32>& vertex_indices) {
    if (vertex_indices.empty()) return;

    const auto* masses_ptr = db.GetArray<SimMass>(mesh_entity);
    if (!masses_ptr) return;

    uint32 node_count = static_cast<uint32>(masses_ptr->size());
    auto masses = *masses_ptr;

    // Load existing fixed vertices (may already have some pinned)
    std::vector<FixedVertex> fixed;
    const auto* existing = db.GetArray<FixedVertex>(mesh_entity);
    if (existing) fixed = *existing;

    for (uint32 idx : vertex_indices) {
        if (idx >= node_count) continue;
        // Skip if already pinned
        bool already = std::any_of(fixed.begin(), fixed.end(),
            [idx](const FixedVertex& fv) { return fv.vertex_index == idx; });
        if (already) continue;

        FixedVertex fv;
        fv.vertex_index = idx;
        fv.original_mass = masses[idx].mass;
        fv.original_inv_mass = masses[idx].inv_mass;
        fixed.push_back(fv);

        masses[idx].mass = 9999999.0f;
        masses[idx].inv_mass = 0.0f;
    }

    db.SetArray<SimMass>(mesh_entity, std::move(masses));
    db.SetArray<FixedVertex>(mesh_entity, std::move(fixed));
}

void UnpinVertices(Database& db, Entity mesh_entity,
                   const std::vector<uint32>& vertex_indices) {
    if (vertex_indices.empty()) return;

    const auto* masses_ptr = db.GetArray<SimMass>(mesh_entity);
    const auto* fixed_ptr = db.GetArray<FixedVertex>(mesh_entity);
    if (!masses_ptr || !fixed_ptr) return;

    auto masses = *masses_ptr;
    auto fixed = *fixed_ptr;

    for (uint32 idx : vertex_indices) {
        auto it = std::find_if(fixed.begin(), fixed.end(),
            [idx](const FixedVertex& fv) { return fv.vertex_index == idx; });
        if (it == fixed.end()) continue;

        // Restore original mass
        masses[idx].mass = it->original_mass;
        masses[idx].inv_mass = it->original_inv_mass;

        // Remove from fixed list (swap-and-pop)
        *it = fixed.back();
        fixed.pop_back();
    }

    db.SetArray<SimMass>(mesh_entity, std::move(masses));
    db.SetArray<FixedVertex>(mesh_entity, std::move(fixed));
}

}  // namespace ext_mesh
