#pragma once

#include "core_util/types.h"

namespace ext_mesh {

using namespace mps;

// Host-only metadata component on a mesh entity.
// Vertex data (SimPosition/SimVelocity/SimMass) and face topology (MeshFace)
// are stored as ArrayStorage on the same entity.
struct MeshComponent {
    uint32 vertex_count = 0;
    uint32 face_count = 0;
    uint32 edge_count = 0;
};

}  // namespace ext_mesh
