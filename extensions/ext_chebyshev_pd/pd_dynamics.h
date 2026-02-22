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

namespace ext_chebyshev_pd {

using namespace mps;

struct alignas(16) JacobiParams {
    float32 omega = 1.0f;
    uint32 is_first_step = 1;
    float32 _pad0 = 0.0f;
    float32 _pad1 = 0.0f;
};

class PDDynamics {
public:
    PDDynamics();
    ~PDDynamics();

    void AddTerm(std::unique_ptr<simulate::IProjectiveTerm> term);
    void SetIterations(uint32 iterations) { iterations_ = iterations; }
    void SetChebyshevRho(float32 rho) { chebyshev_rho_ = rho; }

    void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                    WGPUBuffer physics_buffer, uint64 physics_size,
                    WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                    WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

    void Solve(WGPUCommandEncoder encoder);
    bool CalibrateRho();
    [[nodiscard]] bool IsRhoCalibrated() const { return rho_calibrated_; }

    [[nodiscard]] WGPUBuffer GetQCurrBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetParamsBuffer() const;
    [[nodiscard]] uint64 GetParamsSize() const;
    [[nodiscard]] uint64 GetVec4BufferSize() const;

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

    std::vector<std::unique_ptr<simulate::IProjectiveTerm>> terms_;
    std::unique_ptr<simulate::SparsityBuilder> sparsity_;
    uint32 nnz_ = 0;

    uint32 node_count_ = 0;
    uint32 edge_count_ = 0;
    uint32 face_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 node_wg_count_ = 0;

    uint32 iterations_ = 20;
    float32 chebyshev_rho_ = 0.0f;
    bool rho_calibrated_ = false;

    WGPUBuffer physics_buffer_ = nullptr;
    uint64 physics_size_ = 0;

    std::unique_ptr<gpu::GPUBuffer<simulate::SolverParams>> params_buffer_;
    simulate::SolverParams params_{};

    std::unique_ptr<gpu::GPUBuffer<JacobiParams>> jacobi_params_buffer_;
    std::unique_ptr<gpu::GPUBuffer<JacobiParams>> jacobi_staging_buffer_;

    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_row_ptr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_col_idx_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> csr_values_buffer_;

    std::unique_ptr<gpu::GPUBuffer<float32>> diag_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> d_inv_buffer_;

    std::unique_ptr<gpu::GPUBuffer<float32>> x_old_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> s_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_curr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_prev_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_new_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> rhs_buffer_;

    gpu::GPUComputePipeline pd_init_pipeline_;
    gpu::GPUComputePipeline pd_predict_pipeline_;
    gpu::GPUComputePipeline pd_copy_pipeline_;
    gpu::GPUComputePipeline pd_mass_rhs_pipeline_;
    gpu::GPUComputePipeline pd_inertial_lhs_pipeline_;
    gpu::GPUComputePipeline pd_compute_d_inv_pipeline_;
    gpu::GPUComputePipeline pd_jacobi_step_pipeline_;

    gpu::GPUBindGroup bg_init_;
    gpu::GPUBindGroup bg_predict_;
    gpu::GPUBindGroup bg_copy_q_from_s_;
    gpu::GPUBindGroup bg_mass_rhs_;
    gpu::GPUBindGroup bg_inertial_lhs_;
    gpu::GPUBindGroup bg_compute_d_inv_;
    gpu::GPUBindGroup bg_jacobi_step_;

    static constexpr uint32 kWorkgroupSize = 64;
};

}  // namespace ext_chebyshev_pd
