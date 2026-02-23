#include "ext_jgs2/jgs2_precompute.h"
#include "core_util/logger.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace mps;
using namespace mps::util;

namespace ext_jgs2 {

namespace {

// Row-major N×N dense matrix (lower triangle used for Cholesky)
struct Matrix {
    std::vector<float64> data;
    uint32 n;

    explicit Matrix(uint32 n_) : data(uint64(n_) * n_, 0.0), n(n_) {}

    float64& operator()(uint32 r, uint32 c) { return data[uint64(r) * n + c]; }
    float64 operator()(uint32 r, uint32 c) const { return data[uint64(r) * n + c]; }
};

// In-place Cholesky factorization: H = L·Lᵀ (lower triangle stored in H)
bool CholeskyFactorize(Matrix& H) {
    uint32 n = H.n;

    for (uint32 j = 0; j < n; ++j) {
        float64 sum = H(j, j);
        for (uint32 k = 0; k < j; ++k) {
            sum -= H(j, k) * H(j, k);
        }
        if (sum <= 0.0) {
            LogError("SchurCorrection: Cholesky failed at column ", j,
                     " (pivot = ", sum, ")");
            return false;
        }
        H(j, j) = std::sqrt(sum);

        for (uint32 i = j + 1; i < n; ++i) {
            float64 s = H(i, j);
            for (uint32 k = 0; k < j; ++k) {
                s -= H(i, k) * H(j, k);
            }
            H(i, j) = s / H(j, j);
        }
    }

    return true;
}

// Solve L·x = b (forward substitution)
void ForwardSolve(const Matrix& L, const float64* b, float64* x, uint32 n) {
    for (uint32 i = 0; i < n; ++i) {
        float64 sum = b[i];
        for (uint32 k = 0; k < i; ++k) {
            sum -= L(i, k) * x[k];
        }
        x[i] = sum / L(i, i);
    }
}

// Solve Lᵀ·x = b (backward substitution)
void BackwardSolve(const Matrix& L, const float64* b, float64* x, uint32 n) {
    for (int32 i = static_cast<int32>(n) - 1; i >= 0; --i) {
        float64 sum = b[i];
        for (uint32 k = static_cast<uint32>(i) + 1; k < n; ++k) {
            sum -= L(k, i) * x[k];
        }
        x[i] = sum / L(i, i);
    }
}

// 3x3 matrix inverse (row-major)
bool Invert3x3(const float64 M[9], float64 out[9]) {
    float64 det = M[0] * (M[4] * M[8] - M[5] * M[7])
               - M[1] * (M[3] * M[8] - M[5] * M[6])
               + M[2] * (M[3] * M[7] - M[4] * M[6]);

    if (std::abs(det) < 1e-30) {
        for (uint32 i = 0; i < 9; ++i) out[i] = 0.0;
        return false;
    }

    float64 inv_det = 1.0 / det;
    out[0] = (M[4] * M[8] - M[5] * M[7]) * inv_det;
    out[1] = (M[2] * M[7] - M[1] * M[8]) * inv_det;
    out[2] = (M[1] * M[5] - M[2] * M[4]) * inv_det;
    out[3] = (M[5] * M[6] - M[3] * M[8]) * inv_det;
    out[4] = (M[0] * M[8] - M[2] * M[6]) * inv_det;
    out[5] = (M[2] * M[3] - M[0] * M[5]) * inv_det;
    out[6] = (M[3] * M[7] - M[4] * M[6]) * inv_det;
    out[7] = (M[1] * M[6] - M[0] * M[7]) * inv_det;
    out[8] = (M[0] * M[4] - M[1] * M[3]) * inv_det;
    return true;
}

}  // anonymous namespace

std::vector<float32> SchurCorrection::Compute(
    uint32 node_count,
    const std::vector<float32>& host_positions,
    const std::vector<float32>& host_masses,
    const std::vector<float32>& host_inv_masses,
    const std::vector<std::pair<uint32, uint32>>& edges,
    const std::vector<float32>& rest_lengths,
    float32 stiffness,
    float32 dt) {

    uint32 N3 = node_count * 3;
    LogInfo("SchurCorrection: ", node_count, " nodes, ", edges.size(),
            " edges (", N3, "x", N3, " Cholesky)");

    // 1. Assemble dense rest-pose Hessian H (3N × 3N)
    Matrix H(N3);

    float64 inv_dt_sq = 1.0 / (float64(dt) * float64(dt));
    float64 k = float64(stiffness);

    // Inertia: H[3i+c, 3i+c] += mass[i] / dt²
    for (uint32 i = 0; i < node_count; ++i) {
        float64 m_dt2 = float64(host_masses[i]) * inv_dt_sq;
        for (uint32 c = 0; c < 3; ++c) {
            H(i * 3 + c, i * 3 + c) += m_dt2;
        }
    }

    // Springs: energy Hessian at rest configuration
    for (uint32 e = 0; e < static_cast<uint32>(edges.size()); ++e) {
        uint32 a = edges[e].first;
        uint32 b = edges[e].second;

        float64 dx = float64(host_positions[a * 3 + 0]) - float64(host_positions[b * 3 + 0]);
        float64 dy = float64(host_positions[a * 3 + 1]) - float64(host_positions[b * 3 + 1]);
        float64 dz = float64(host_positions[a * 3 + 2]) - float64(host_positions[b * 3 + 2]);
        float64 dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < 1e-12) continue;

        float64 dir[3] = {dx / dist, dy / dist, dz / dist};
        float64 ratio = std::min(float64(rest_lengths[e]) / dist, 1.0);
        float64 coeff_i = k * (1.0 - ratio);
        float64 coeff_d = k * ratio;

        float64 block[3][3];
        for (uint32 r = 0; r < 3; ++r) {
            for (uint32 c = 0; c < 3; ++c) {
                block[r][c] = coeff_d * dir[r] * dir[c];
                if (r == c) block[r][c] += coeff_i;
            }
        }

        for (uint32 r = 0; r < 3; ++r) {
            for (uint32 c = 0; c < 3; ++c) {
                H(a * 3 + r, a * 3 + c) += block[r][c];
                H(b * 3 + r, b * 3 + c) += block[r][c];
                H(a * 3 + r, b * 3 + c) -= block[r][c];
                H(b * 3 + r, a * 3 + c) -= block[r][c];
            }
        }
    }

    // Constrain pinned DOFs: set their rows/columns to identity.
    for (uint32 i = 0; i < node_count; ++i) {
        if (host_inv_masses[i] <= 0.0f) {
            for (uint32 c = 0; c < 3; ++c) {
                uint32 idx = i * 3 + c;
                for (uint32 j = 0; j < N3; ++j) {
                    H(idx, j) = 0.0;
                    H(j, idx) = 0.0;
                }
                H(idx, idx) = 1.0;
            }
        }
    }

    // Save per-vertex diagonal blocks Hᵢᵢ before Cholesky overwrites H
    std::vector<float64> H_diag(uint64(node_count) * 9);
    for (uint32 i = 0; i < node_count; ++i) {
        for (uint32 r = 0; r < 3; ++r) {
            for (uint32 c = 0; c < 3; ++c) {
                H_diag[i * 9 + r * 3 + c] = H(i * 3 + r, i * 3 + c);
            }
        }
    }

    // 2. Cholesky factorization: H = L·Lᵀ
    if (!CholeskyFactorize(H)) {
        LogError("SchurCorrection: Cholesky failed, returning zeros");
        return std::vector<float32>(uint64(node_count) * 9, 0.0f);
    }
    const Matrix& L = H;

    // 3. Extract per-vertex (H⁻¹)ᵢᵢ via column solves, compute Δᵢ = Hᵢᵢ - Sᵢ
    std::vector<float32> result(uint64(node_count) * 9, 0.0f);

    std::vector<float64> rhs(N3, 0.0);
    std::vector<float64> y(N3);
    std::vector<float64> x(N3);

    for (uint32 i = 0; i < node_count; ++i) {
        if (host_inv_masses[i] <= 0.0f) continue;

        float64 H_inv_ii[9] = {};

        for (uint32 c = 0; c < 3; ++c) {
            uint32 col = i * 3 + c;
            rhs[col] = 1.0;

            ForwardSolve(L, rhs.data(), y.data(), N3);
            BackwardSolve(L, y.data(), x.data(), N3);

            for (uint32 r = 0; r < 3; ++r) {
                H_inv_ii[r * 3 + c] = x[i * 3 + r];
            }

            rhs[col] = 0.0;
        }

        // Sᵢ = ((H⁻¹)ᵢᵢ)⁻¹
        float64 S_i[9];
        if (!Invert3x3(H_inv_ii, S_i)) continue;

        // Δᵢ = Hᵢᵢ - Sᵢ (positive regularizer, paper Eq. 15)
        for (uint32 j = 0; j < 9; ++j) {
            float64 delta = H_diag[i * 9 + j] - S_i[j];
            if (std::isfinite(delta)) {
                result[i * 9 + j] = static_cast<float32>(delta);
            }
        }
    }

    LogInfo("SchurCorrection: done");
    return result;
}

}  // namespace ext_jgs2
