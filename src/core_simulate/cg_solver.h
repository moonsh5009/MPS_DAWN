#pragma once

#include "core_util/types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace mps {
namespace simulate {

// Interface for sparse matrix-vector product used by the CG solver.
// Implementors cache the bind group (via PrepareSolve) and dispatch via Apply.
class ISpMVOperator {
public:
    virtual ~ISpMVOperator() = default;

    // Called once before the CG loop. Create bind group with CG buffers.
    virtual void PrepareSolve(WGPUBuffer p_buffer, uint64 p_size,
                              WGPUBuffer ap_buffer, uint64 ap_size) = 0;

    // Dispatch Ap = A * p (one compute pass)
    virtual void Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) = 0;
};

// Generic GPU conjugate gradient solver.
// Uses MPCG (mass-filtered CG) for pinned nodes (inv_mass == 0).
class CGSolver {
public:
    CGSolver();
    ~CGSolver();

    void Initialize(uint32 node_count, uint32 workgroup_size = 64);

    // Callers write RHS into this buffer before calling Solve()
    [[nodiscard]] WGPUBuffer GetRHSBuffer() const;

    // Solution is written here after Solve()
    [[nodiscard]] WGPUBuffer GetSolutionBuffer() const;

    // Vector size in bytes: node_count * 4 * sizeof(float32)
    [[nodiscard]] uint64 GetVectorSize() const;

    // Cache all bind groups for the CG loop. Call after Initialize().
    // Also calls spmv.PrepareSolve() with p and ap buffers.
    void CacheBindGroups(WGPUBuffer params_buffer, uint64 params_size,
                         WGPUBuffer mass_buffer, uint64 mass_size,
                         ISpMVOperator& spmv);

    // Run CG solver. RHS must already be in GetRHSBuffer().
    // Bind groups must be cached via CacheBindGroups() first.
    void Solve(WGPUCommandEncoder encoder, uint32 cg_iterations);

    void Shutdown();

private:
    void CreateBuffers();
    void CreatePipelines();

    uint32 node_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 workgroup_count_ = 0;
    uint32 dot_partial_count_ = 0;

    // CG vectors
    std::unique_ptr<gpu::GPUBuffer<float32>> cg_x_;
    std::unique_ptr<gpu::GPUBuffer<float32>> cg_r_;
    std::unique_ptr<gpu::GPUBuffer<float32>> cg_p_;
    std::unique_ptr<gpu::GPUBuffer<float32>> cg_ap_;

    // Reduction buffers
    std::unique_ptr<gpu::GPUBuffer<float32>> partial_;
    std::unique_ptr<gpu::GPUBuffer<float32>> scalar_;

    // CG constant uniforms
    struct alignas(16) DotConfig { uint32 target; uint32 count; };
    struct alignas(16) ScalarMode { uint32 mode; };
    std::unique_ptr<gpu::GPUBuffer<DotConfig>> dc_rr_;
    std::unique_ptr<gpu::GPUBuffer<DotConfig>> dc_pap_;
    std::unique_ptr<gpu::GPUBuffer<DotConfig>> dc_rr_new_;
    std::unique_ptr<gpu::GPUBuffer<ScalarMode>> mode_alpha_;
    std::unique_ptr<gpu::GPUBuffer<ScalarMode>> mode_beta_;

    // Pipelines
    gpu::GPUComputePipeline cg_init_pipeline_;
    gpu::GPUComputePipeline cg_dot_pipeline_;
    gpu::GPUComputePipeline cg_dot_final_pipeline_;
    gpu::GPUComputePipeline cg_compute_scalars_pipeline_;
    gpu::GPUComputePipeline cg_update_xr_pipeline_;
    gpu::GPUComputePipeline cg_update_p_pipeline_;

    // Cached bind groups (created in CacheBindGroups)
    gpu::GPUBindGroup bg_init_;
    gpu::GPUBindGroup bg_dot_rr_;
    gpu::GPUBindGroup bg_dot_pap_;
    gpu::GPUBindGroup bg_df_rr_;
    gpu::GPUBindGroup bg_df_pap_;
    gpu::GPUBindGroup bg_df_rr_new_;
    gpu::GPUBindGroup bg_alpha_;
    gpu::GPUBindGroup bg_beta_;
    gpu::GPUBindGroup bg_xr_;
    gpu::GPUBindGroup bg_p_;

    // Cached SpMV operator (non-owning, set in CacheBindGroups)
    ISpMVOperator* spmv_ = nullptr;
};

}  // namespace simulate
}  // namespace mps
