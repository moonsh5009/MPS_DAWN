#include "ext_dynamics/constraint_builder.h"
#include "ext_dynamics/spring_types.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_types.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_mesh/mesh_types.h"
#include "ext_mesh/mesh_component.h"
#include "core_database/database.h"
#include "core_simulate/sim_components.h"
#include <cmath>

using namespace mps;
using namespace mps::simulate;
using namespace mps::database;

namespace ext_dynamics {

uint32 BuildSpringConstraints(Database& db, Entity mesh_entity, float32 stiffness) {
    const auto* positions = db.GetArray<SimPosition>(mesh_entity);
    const auto* mesh_edges = db.GetArray<ext_mesh::MeshEdge>(mesh_entity);
    if (!positions || !mesh_edges) return 0;

    // Build SpringEdge array from MeshEdge topology (just compute rest_length)
    std::vector<SpringEdge> edges;
    edges.reserve(mesh_edges->size());
    for (const auto& me : *mesh_edges) {
        const auto& pa = (*positions)[me.n0];
        const auto& pb = (*positions)[me.n1];
        float32 dx = pb.x - pa.x;
        float32 dy = pb.y - pa.y;
        float32 dz = pb.z - pa.z;

        SpringEdge edge;
        edge.n0 = me.n0;
        edge.n1 = me.n1;
        edge.rest_length = std::sqrt(dx * dx + dy * dy + dz * dz);
        edges.push_back(edge);
    }

    uint32 count = static_cast<uint32>(edges.size());

    db.SetArray<SpringEdge>(mesh_entity, std::move(edges));
    db.AddComponent<SpringConstraintData>(mesh_entity, {stiffness});
    return count;
}

uint32 BuildAreaConstraints(Database& db, Entity mesh_entity, float32 stretch_stiffness, float32 shear_stiffness) {
    const auto* positions = db.GetArray<SimPosition>(mesh_entity);
    const auto* faces = db.GetArray<ext_mesh::MeshFace>(mesh_entity);
    if (!positions || !faces) return 0;

    std::vector<AreaTriangle> triangles;
    triangles.reserve(faces->size());

    for (const auto& face : *faces) {
        const auto& p0 = (*positions)[face.n0];
        const auto& p1 = (*positions)[face.n1];
        const auto& p2 = (*positions)[face.n2];

        float32 e1x = p1.x - p0.x, e1y = p1.y - p0.y, e1z = p1.z - p0.z;
        float32 e2x = p2.x - p0.x, e2y = p2.y - p0.y, e2z = p2.z - p0.z;

        float32 cx = e1y * e2z - e1z * e2y;
        float32 cy = e1z * e2x - e1x * e2z;
        float32 cz = e1x * e2y - e1y * e2x;
        float32 rest_area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);

        float32 e1_len = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        float32 inv_e1 = (e1_len > 1e-12f) ? (1.0f / e1_len) : 0.0f;
        float32 t1x = e1x * inv_e1, t1y = e1y * inv_e1, t1z = e1z * inv_e1;

        float32 e2_dot_t1 = e2x * t1x + e2y * t1y + e2z * t1z;
        float32 e2p_x = e2x - e2_dot_t1 * t1x;
        float32 e2p_y = e2y - e2_dot_t1 * t1y;
        float32 e2p_z = e2z - e2_dot_t1 * t1z;
        float32 e2p_len = std::sqrt(e2p_x * e2p_x + e2p_y * e2p_y + e2p_z * e2p_z);

        float32 inv_e2p = (e2p_len > 1e-12f) ? (1.0f / e2p_len) : 0.0f;

        AreaTriangle tri;
        tri.n0 = face.n0;
        tri.n1 = face.n1;
        tri.n2 = face.n2;
        tri.rest_area = rest_area;
        tri.dm_inv_00 = inv_e1;
        tri.dm_inv_01 = -e2_dot_t1 * inv_e1 * inv_e2p;
        tri.dm_inv_10 = 0.0f;
        tri.dm_inv_11 = inv_e2p;
        triangles.push_back(tri);
    }

    uint32 count = static_cast<uint32>(triangles.size());
    db.SetArray<AreaTriangle>(mesh_entity, std::move(triangles));
    db.AddComponent<AreaConstraintData>(mesh_entity, {stretch_stiffness, shear_stiffness});
    return count;
}

}  // namespace ext_dynamics
