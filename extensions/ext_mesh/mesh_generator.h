#pragma once

#include "core_database/entity.h"
#include "core_util/types.h"
#include "core_util/math.h"
#include <string>
#include <vector>

namespace mps { namespace database { class Database; } }

namespace ext_mesh {

struct MeshResult {
    mps::database::Entity mesh_entity = mps::database::kInvalidEntity;
    mps::uint32 node_count = 0;
    mps::uint32 face_count = 0;
};

// Create a grid mesh on XZ plane at Y=height_offset.
// Adds SimPosition, SimVelocity, SimMass (area-weighted), MeshFace, MeshComponent.
// Must be called inside a Transact block.
MeshResult CreateGrid(mps::database::Database& db,
                      mps::uint32 width, mps::uint32 height, mps::float32 spacing,
                      mps::util::vec3 offset, mps::float32 density = 100.0f);

// Import a triangle mesh from OBJ file (filename relative to assets/objs/).
// Adds SimPosition, SimVelocity, SimMass (area-weighted), MeshFace, MeshComponent.
// Quads are automatically triangulated. Must be called inside a Transact block.
// offset: translation applied to all vertices after scaling.
// density: surface density (kg/m²) for area-weighted mass computation.
MeshResult ImportOBJ(mps::database::Database& db,
                     const std::string& filename,
                     mps::float32 scale = 1.0f,
                     mps::util::vec3 offset = {0.0f, 0.0f, 0.0f},
                     mps::float32 density = 100.0f);

// Pin vertices on a mesh entity.
// Appends to FixedVertex array (saving original mass) and sets mass=9999999, inv_mass=0.
// Must be called inside a Transact block.
void PinVertices(mps::database::Database& db,
                 mps::database::Entity mesh_entity,
                 const std::vector<mps::uint32>& vertex_indices);

// Unpin vertices on a mesh entity.
// Removes from FixedVertex array and restores original mass/inv_mass.
// Must be called inside a Transact block.
void UnpinVertices(mps::database::Database& db,
                   mps::database::Entity mesh_entity,
                   const std::vector<mps::uint32>& vertex_indices);

}  // namespace ext_mesh
