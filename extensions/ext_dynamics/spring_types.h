#pragma once

#include "core_util/types.h"

namespace ext_dynamics {

using namespace mps;

// Spring edge topology (16 bytes, GPU-compatible)
// Stiffness is a uniform parameter via SpringConstraintData, not per-edge.
struct SpringEdge {
    uint32 n0 = 0;
    uint32 n1 = 0;
    float32 rest_length = 0.0f;
};

// CSR mapping: tells each edge thread where to write Hessian blocks
struct EdgeCSRMapping {
    uint32 block_ab = 0;  // CSR index for block (a,b)
    uint32 block_ba = 0;  // CSR index for block (b,a)
    uint32 block_aa = 0;  // diagonal index for node a
    uint32 block_bb = 0;  // diagonal index for node b
};

}  // namespace ext_dynamics
