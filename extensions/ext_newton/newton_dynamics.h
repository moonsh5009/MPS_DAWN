#pragma once

#include "core_simulate/dynamics_term.h"
#include "core_simulate/cg_solver.h"
#include "core_simulate/solver_params.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace mps {
namespace simulate {

// Newton-Raphson dynamics solver.
// Orchestrates the Newton loop with pluggable IDynamicsTerm implementations.
// Computes dv_total (accumulated velocity delta) which the caller applies
// to update velocity and position.
class NewtonDynamics {
public:
    NewtonDynamics();
    ~NewtonDynamics();

    // Add a dynamics term (call before Initialize)
    void AddTerm(std::unique_ptr<IDynamicsTerm> term);

    // Configure solver iterations (call before Initialize or anytime)
    void SetNewtonIterations(uint32 iterations) { newton_iterations_ = iterations; }
    void SetCGMaxIterations(uint32 iterations) { cg_max_iterations_ = iterations; }

    // Initialize after all terms are added.
    // physics_buffer: DeviceDB singleton uniform (binding 0).
    // External buffer handles are used for bind group caching.
    void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                    WGPUBuffer physics_buffer, uint64 physics_size,
                    WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                    WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

    // Run the Newton-Raphson solver for one timestep.
    // Records all compute passes into encoder using cached bind groups.
    // Caller must submit the encoder and handle readback.
    void Solve(WGPUCommandEncoder encoder);

    // Result buffers (valid after Initialize)
    [[nodiscard]] WGPUBuffer GetDVTotalBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetParamsBuffer() const;
    [[nodiscard]] uint64 GetParamsSize() const;
    [[nodiscard]] uint64 GetVec4BufferSize() const;

    void Shutdown();

private:
    void BuildSparsity();
    void CreateBuffers();
    void CreatePipelines();
    void CacheBindGroups(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                         WGPUBuffer mass_buffer);

    // Terms
    std::vector<std::unique_ptr<IDynamicsTerm>> terms_;

    // Sparsity
    std::unique_ptr<SparsityBuilder> sparsity_;
    uint32 nnz_ = 0;

    // Mesh counts
    uint32 node_count_ = 0;
    uint32 edge_count_ = 0;
    uint32 face_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 node_wg_count_ = 0;

    // Newton config
    uint32 newton_iterations_ = 1;
    uint32 cg_max_iterations_ = 30;

    // Physics uniform (non-owning, from DeviceDB)
    WGPUBuffer physics_buffer_ = nullptr;
    uint64 physics_size_ = 0;

    // Solver params uniform
    std::unique_ptr<gpu::GPUBuffer<SolverParams>> params_buffer_;
    SolverParams params_{};

    // CSR structure
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_row_ptr_buffer_;
    std::unique_ptr<gpu::GPUBuffer<uint32>> csr_col_idx_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> csr_values_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> diag_values_buffer_;

    // Force buffer (atomic u32 for CAS-based float accumulation)
    std::unique_ptr<gpu::GPUBuffer<float32>> force_buffer_;

    // Newton solver buffers
    std::unique_ptr<gpu::GPUBuffer<float32>> x_old_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> dv_total_buffer_;

    // CG solver
    std::unique_ptr<CGSolver> cg_solver_;

    // Internal SpMV operator
    class SpMVOperator : public ISpMVOperator {
    public:
        explicit SpMVOperator(NewtonDynamics& owner);
        void PrepareSolve(WGPUBuffer p_buffer, uint64 p_size,
                          WGPUBuffer ap_buffer, uint64 ap_size) override;
        void Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) override;

    private:
        NewtonDynamics& owner_;
        gpu::GPUBindGroup bind_group_;
    };
    std::unique_ptr<SpMVOperator> spmv_;

    // Newton pipelines
    gpu::GPUComputePipeline newton_init_pipeline_;
    gpu::GPUComputePipeline newton_predict_pos_pipeline_;
    gpu::GPUComputePipeline newton_accumulate_dv_pipeline_;
    gpu::GPUComputePipeline clear_forces_pipeline_;
    gpu::GPUComputePipeline assemble_rhs_pipeline_;
    gpu::GPUComputePipeline spmv_pipeline_;
    gpu::GPUComputePipeline inertia_pipeline_;
    gpu::GPUComputePipeline gravity_pipeline_;

    // Cached bind groups (created in CacheBindGroups)
    gpu::GPUBindGroup bg_newton_init_;
    gpu::GPUBindGroup bg_predict_;
    gpu::GPUBindGroup bg_clear_forces_;
    gpu::GPUBindGroup bg_rhs_;
    gpu::GPUBindGroup bg_accumulate_;
    gpu::GPUBindGroup bg_inertia_;
    gpu::GPUBindGroup bg_gravity_;

    static constexpr uint32 kWorkgroupSize = 64;
};

}  // namespace simulate
}  // namespace mps
