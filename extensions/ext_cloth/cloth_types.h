#pragma once

#include "core_util/types.h"

namespace ext_cloth {

using namespace mps;

// GPU-only topology structs (16 bytes each)
struct ClothEdge {
    uint32 n0 = 0;
    uint32 n1 = 0;
    float32 rest_length = 0.0f;
    float32 stiffness = 100.0f;
};

struct ClothFace {
    uint32 n0 = 0;
    uint32 n1 = 0;
    uint32 n2 = 0;
    uint32 pad = 0;
};

// CSR mapping: tells each edge thread where to write Hessian blocks
struct EdgeCSRMapping {
    uint32 block_ab = 0;  // CSR index for block (a,b)
    uint32 block_ba = 0;  // CSR index for block (b,a)
    uint32 block_aa = 0;  // diagonal index for node a
    uint32 block_bb = 0;  // diagonal index for node b
};

// Simulation parameters uniform
struct ClothSimParams {
    float32 dt = 1.0f / 60.0f;
    float32 gravity_x = 0.0f;
    float32 gravity_y = -9.81f;
    float32 gravity_z = 0.0f;

    uint32 node_count = 0;
    uint32 edge_count = 0;
    uint32 face_count = 0;
    uint32 cg_max_iter = 20;

    float32 damping = 0.999f;
    float32 cg_tolerance = 1e-6f;
    float32 pad0 = 0.0f;
    float32 pad1 = 0.0f;
};

}  // namespace ext_cloth
