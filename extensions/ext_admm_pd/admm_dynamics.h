#pragma once

#include "core_simulate/projective_term.h"
#include "core_simulate/dynamics_term.h"
#include "core_simulate/cg_solver.h"
#include "core_simulate/solver_params.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <vector>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace ext_admm_pd {

using namespace mps;

class ADMMDynamics {
public:
    ADMMDynamics();
    ~ADMMDynamics();

    void AddTerm(std::unique_ptr<simulate::IProjectiveTerm> term);
    void SetADMMIterations(uint32 iterations) { admm_iterations_ = iterations; }
    void SetCGIterations(uint32 iterations) { cg_iterations_ = iterations; }
    void SetPenaltyWeight(float32 rho) { penalty_weight_ = rho; }

    void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                    WGPUBuffer physics_buffer, uint64 physics_size,
                    WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                    WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

    void Solve(WGPUCommandEncoder encoder);

    [[nodiscard]] WGPUBuffer GetQCurrBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetParamsBuffer() const;
    [[nodiscard]] uint64 GetParamsSize() const;
    [[nodiscard]] uint64 GetVec4BufferSize() const;

    void Shutdown();

private:
    void BuildSparsity();
    void CreateBuffers(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                       WGPUBuffer mass_buffer);
    void CreatePipelines();
    void CacheBindGroups(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                         WGPUBuffer mass_buffer);

    // Internal SpMV operator for the constant PD system matrix
    class SpMVOperator : public simulate::ISpMVOperator {
    public:
        explicit SpMVOperator(ADMMDynamics& owner);
        void PrepareSolve(WGPUBuffer p_buffer, uint64 p_size,
                          WGPUBuffer ap_buffer, uint64 ap_size) override;
        void Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) override;

    private:
        ADMMDynamics& owner_;
        gpu::GPUBindGroup bind_group_;
    };

    std::vector<std::unique_ptr<simulate::IProjectiveTerm>> terms_;
    std::unique_ptr<simulate::SparsityBuilder> sparsity_;
    uint32 nnz_ = 0;

    uint32 node_count_ = 0;
    uint32 edge_count_ = 0;
    uint32 face_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 node_wg_count_ = 0;

    uint32 admm_iterations_ = 20;
    uint32 cg_iterations_ = 10;
    float32 penalty_weight_ = 1.0f;

    WGPUBuffer physics_buffer_ = nullptr;
    uint64 physics_size_ = 0;

    std::unique_ptr<gpu::GPUBuffer<simulate::SolverParams>> params_buffer_;
    simulate::SolverParams params_{};

    // CSR structure (constant LHS)
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_row_ptr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_col_idx_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> csr_values_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> diag_buffer_;

    // Solver buffers
    std::unique_ptr<gpu::GPUBuffer<float32>> x_old_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> s_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_curr_buffer_;

    // CG solver (inner solve for ADMM global step)
    std::unique_ptr<simulate::CGSolver> cg_solver_;
    std::unique_ptr<SpMVOperator> spmv_;

    // Shared PD infrastructure pipelines
    gpu::GPUComputePipeline pd_init_pipeline_;
    gpu::GPUComputePipeline pd_predict_pipeline_;
    gpu::GPUComputePipeline pd_copy_pipeline_;
    gpu::GPUComputePipeline pd_mass_rhs_pipeline_;
    gpu::GPUComputePipeline pd_inertial_lhs_pipeline_;
    gpu::GPUComputePipeline pd_fixup_pinned_pipeline_;
    gpu::GPUComputePipeline spmv_pipeline_;

    // Cached bind groups
    gpu::GPUBindGroup bg_init_;
    gpu::GPUBindGroup bg_predict_;
    gpu::GPUBindGroup bg_copy_q_from_s_;
    gpu::GPUBindGroup bg_mass_rhs_;
    gpu::GPUBindGroup bg_inertial_lhs_;
    gpu::GPUBindGroup bg_fixup_pinned_;

    static constexpr uint32 kWorkgroupSize = 64;
};

}  // namespace ext_admm_pd
