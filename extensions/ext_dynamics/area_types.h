#pragma once

#include "core_util/types.h"

namespace ext_dynamics {

using namespace mps;

// Per-triangle area constraint data (32 bytes, GPU-compatible)
// Includes precomputed Dm_inv (inverse of 2x2 rest edge matrix in local coords)
struct AreaTriangle {
    uint32 n0 = 0;
    uint32 n1 = 0;
    uint32 n2 = 0;
    float32 rest_area = 0.0f;
    float32 dm_inv_00 = 0.0f;  // row 0, col 0 of Dm^{-1}
    float32 dm_inv_01 = 0.0f;  // row 0, col 1
    float32 dm_inv_10 = 0.0f;  // row 1, col 0
    float32 dm_inv_11 = 0.0f;  // row 1, col 1
};

// CSR mapping for each face: tells each face thread where to write off-diagonal Hessian blocks.
// 6 CSR indices for the 3 edges of a triangle (each edge has 2 directed entries).
// alignas(16) pads to 32 bytes for GPU alignment.
struct FaceCSRMapping {
    uint32 csr_01 = 0;  // CSR index for block (n0, n1)
    uint32 csr_10 = 0;  // CSR index for block (n1, n0)
    uint32 csr_02 = 0;  // CSR index for block (n0, n2)
    uint32 csr_20 = 0;  // CSR index for block (n2, n0)
    uint32 csr_12 = 0;  // CSR index for block (n1, n2)
    uint32 csr_21 = 0;  // CSR index for block (n2, n1)
};

}  // namespace ext_dynamics
