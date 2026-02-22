#pragma once

#include "core_database/entity.h"
#include "core_util/types.h"

namespace mps { namespace database { class Database; } }

namespace ext_dynamics {

// Build spring edges from face topology on a mesh entity.
// Reads SimPosition + MeshFace arrays, writes SpringEdge array + SpringConstraintData.
// Updates MeshComponent::edge_count. Must be called inside a Transact block.
mps::uint32 BuildSpringConstraints(mps::database::Database& db,
                                   mps::database::Entity mesh_entity,
                                   mps::float32 stiffness);

// Build area triangles from face topology on a mesh entity.
// Reads SimPosition + MeshFace arrays, writes AreaTriangle array + AreaConstraintData.
// Must be called inside a Transact block.
mps::uint32 BuildAreaConstraints(mps::database::Database& db,
                                 mps::database::Entity mesh_entity,
                                 mps::float32 stiffness);

}  // namespace ext_dynamics
