#pragma once

#include "core_simulate/projective_term.h"
#include "core_simulate/dynamics_term.h"
#include "core_simulate/solver_params.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace ext_pd {

using namespace mps;

// Jacobi iteration parameters (per-iteration uniform)
struct alignas(16) JacobiParams {
    float32 omega = 1.0f;        // Chebyshev weight
    uint32 is_first_step = 1;    // 1 = pure Jacobi (no Chebyshev blend)
    float32 _pad0 = 0.0f;
    float32 _pad1 = 0.0f;
};

// Projective Dynamics solver with Chebyshev-accelerated Jacobi iteration.
// Replaces CG with GPU-friendly Jacobi (no dot product reductions).
class PDDynamics {
public:
    PDDynamics();
    ~PDDynamics();

    // Add a projective term (call before Initialize)
    void AddTerm(std::unique_ptr<simulate::IProjectiveTerm> term);

    // Configure solver iterations (call before Initialize or anytime)
    void SetIterations(uint32 iterations) { iterations_ = iterations; }
    void SetChebyshevRho(float32 rho) { chebyshev_rho_ = rho; }

    // Initialize after all terms are added.
    // physics_buffer: DeviceDB singleton uniform (binding 0).
    void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                    WGPUBuffer physics_buffer, uint64 physics_size,
                    WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                    WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

    // Run the PD solver for one timestep (Wang 2015 single fused loop).
    void Solve(WGPUCommandEncoder encoder);

    // Adaptive ρ calibration (Wang 2015 gradient decrease rate method).
    // Runs a full PD solve with pure Jacobi, measures convergence rate,
    // and builds Chebyshev params with the estimated ρ.
    // Uses its own GPU submissions (not encoder-based). Call before first Solve().
    // Returns true if calibration was performed.
    bool CalibrateRho();

    // Whether ρ has been calibrated (manually or adaptively).
    [[nodiscard]] bool IsRhoCalibrated() const { return rho_calibrated_; }

    // Result buffers (valid after Initialize)
    [[nodiscard]] WGPUBuffer GetQCurrBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetParamsBuffer() const;
    [[nodiscard]] uint64 GetParamsSize() const;
    [[nodiscard]] uint64 GetVec4BufferSize() const;

    // Debug: dump key buffer values (first call only)
    void DebugDump();

    void Shutdown();

private:
    void BuildSparsity();
    void CreateBuffers(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                       WGPUBuffer mass_buffer);
    void CreatePipelines();
    void CacheBindGroups(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                         WGPUBuffer mass_buffer);
    void RebuildLHS(WGPUCommandEncoder encoder);
    void BuildChebyshevParams(float32 rho);

    // Terms
    std::vector<std::unique_ptr<simulate::IProjectiveTerm>> terms_;

    // Sparsity
    std::unique_ptr<simulate::SparsityBuilder> sparsity_;
    uint32 nnz_ = 0;

    // Mesh counts
    uint32 node_count_ = 0;
    uint32 edge_count_ = 0;
    uint32 face_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 node_wg_count_ = 0;

    // PD config
    uint32 iterations_ = 20;
    float32 chebyshev_rho_ = 0.0f;  // 0 = auto-calibrate; >0 = manual override
    bool rho_calibrated_ = false;

    // Physics uniform (non-owning, from DeviceDB)
    WGPUBuffer physics_buffer_ = nullptr;
    uint64 physics_size_ = 0;

    // Solver params uniform
    std::unique_ptr<gpu::GPUBuffer<simulate::SolverParams>> params_buffer_;
    simulate::SolverParams params_{};

    // Jacobi params: uniform (CopyDst) + staging (CopySrc, pre-computed per iteration)
    std::unique_ptr<gpu::GPUBuffer<JacobiParams>> jacobi_params_buffer_;
    std::unique_ptr<gpu::GPUBuffer<JacobiParams>> jacobi_staging_buffer_;

    // CSR structure
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_row_ptr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_col_idx_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> csr_values_buffer_;

    // Diagonal + inverse
    std::unique_ptr<gpu::GPUBuffer<float32>> diag_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> d_inv_buffer_;

    // Solver buffers
    std::unique_ptr<gpu::GPUBuffer<float32>> x_old_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> s_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_curr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_prev_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_new_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> rhs_buffer_;

    // Pipelines
    gpu::GPUComputePipeline pd_init_pipeline_;
    gpu::GPUComputePipeline pd_predict_pipeline_;
    gpu::GPUComputePipeline pd_copy_pipeline_;
    gpu::GPUComputePipeline pd_mass_rhs_pipeline_;
    gpu::GPUComputePipeline pd_inertial_lhs_pipeline_;
    gpu::GPUComputePipeline pd_compute_d_inv_pipeline_;
    gpu::GPUComputePipeline pd_jacobi_step_pipeline_;

    // Cached bind groups
    gpu::GPUBindGroup bg_init_;
    gpu::GPUBindGroup bg_predict_;
    gpu::GPUBindGroup bg_copy_q_from_s_;
    gpu::GPUBindGroup bg_mass_rhs_;
    gpu::GPUBindGroup bg_inertial_lhs_;
    gpu::GPUBindGroup bg_compute_d_inv_;
    gpu::GPUBindGroup bg_jacobi_step_;

    static constexpr uint32 kWorkgroupSize = 64;
};

}  // namespace ext_pd
