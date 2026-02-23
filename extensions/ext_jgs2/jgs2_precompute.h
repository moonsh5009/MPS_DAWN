#pragma once

#include "core_util/types.h"
#include <vector>
#include <utility>

namespace ext_jgs2 {

using namespace mps;

// Phase 2: Frozen Schur complement correction precomputation (CPU).
// Assembles rest-pose Hessian, Cholesky factorization, extracts per-vertex correction Δᵢ.
// Complexity: O(N³/3) factorization + O(3N³) column solves — practical for N ≤ ~2000.
struct SchurCorrection {
    // Compute per-vertex Schur complement correction.
    // Returns N×9 floats (row-major 3×3 blocks).
    //
    // host_positions:  N×3 rest positions (x0,y0,z0, x1,y1,z1, ...)
    // host_masses:     N masses
    // host_inv_masses: N inverse masses (≤0 = pinned, constrained to identity in Hessian)
    // edges:           list of (node_a, node_b) pairs
    // rest_lengths:    per-edge rest length
    // stiffness:       spring stiffness k
    // dt:              timestep
    static std::vector<float32> Compute(
        uint32 node_count,
        const std::vector<float32>& host_positions,
        const std::vector<float32>& host_masses,
        const std::vector<float32>& host_inv_masses,
        const std::vector<std::pair<uint32, uint32>>& edges,
        const std::vector<float32>& rest_lengths,
        float32 stiffness,
        float32 dt);
};

}  // namespace ext_jgs2
