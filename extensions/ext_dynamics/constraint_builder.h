#pragma once

#include "core_database/entity.h"
#include "core_util/types.h"

namespace mps { namespace database { class Database; } }

namespace ext_dynamics {

struct ConstraintResult {
    mps::uint32 edge_count = 0;
    mps::uint32 area_count = 0;
};

// Build spring edges and area triangles from face topology on a mesh entity.
// Reads SimPosition + MeshFace arrays, writes SpringEdge + AreaTriangle arrays.
// Updates MeshComponent::edge_count. Must be called inside a Transact block.
// Spring stiffness is configured separately via SpringConstraintData on the constraint entity.
ConstraintResult BuildConstraintsFromFaces(mps::database::Database& db,
                                           mps::database::Entity mesh_entity);

}  // namespace ext_dynamics
