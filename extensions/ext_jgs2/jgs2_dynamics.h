#pragma once

#include "core_simulate/solver_params.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace ext_jgs2 {

using namespace mps;

// Assembly context for JGS2 term bind group caching.
// Passed to IJGS2Term::Initialize() — terms cache bind groups from these handles.
struct JGS2AssemblyContext {
    WGPUBuffer params_buffer;        // solver params uniform
    WGPUBuffer q_buffer;             // current iterate q (read)
    WGPUBuffer gradient_buffer;      // per-vertex gradient N×4 (atomic u32, read_write)
    WGPUBuffer hessian_diag_buffer;  // per-vertex 3×3 Hessian N×9 (atomic u32, read_write)
    uint32 node_count;
    uint32 edge_count;
    uint32 workgroup_size;
    uint64 params_size;
};

// Extension-local pluggable term interface for JGS2.
// Terms accumulate gradient and diagonal Hessian blocks per edge/face via GPU atomics.
class IJGS2Term {
public:
    virtual ~IJGS2Term() = default;
    [[nodiscard]] virtual const std::string& GetName() const = 0;
    virtual void Initialize(const JGS2AssemblyContext& ctx) = 0;
    virtual void Accumulate(WGPUCommandEncoder encoder) = 0;
    virtual void Shutdown() = 0;
};

// JGS2 block Jacobi dynamics solver (Lan et al., SIGGRAPH 2025).
// Per-vertex 3×3 block solve with optional frozen Schur complement correction.
class JGS2Dynamics {
public:
    JGS2Dynamics();
    ~JGS2Dynamics();

    void AddTerm(std::unique_ptr<IJGS2Term> term);
    void SetIterations(uint32 iterations) { iterations_ = iterations; }
    void EnableCorrection(bool enable) { enable_correction_ = enable; }

    void Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                    WGPUBuffer physics_buffer, uint64 physics_size,
                    WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                    WGPUBuffer mass_buffer, uint32 workgroup_size = 64);

    void Solve(WGPUCommandEncoder encoder);

    // Phase 2: upload precomputed correction (N×9 floats)
    void UploadCorrection(const std::vector<float32>& correction_data);

    [[nodiscard]] WGPUBuffer GetQBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetParamsBuffer() const;
    [[nodiscard]] uint64 GetParamsSize() const;
    [[nodiscard]] uint64 GetVec4BufferSize() const;

    void Shutdown();

private:
    void CreateBuffers(WGPUBuffer mass_buffer);
    void CreatePipelines();
    void CacheBindGroups(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                         WGPUBuffer mass_buffer);

    std::vector<std::unique_ptr<IJGS2Term>> terms_;

    uint32 node_count_ = 0;
    uint32 edge_count_ = 0;
    uint32 face_count_ = 0;
    uint32 workgroup_size_ = 64;
    uint32 node_wg_count_ = 0;
    uint32 iterations_ = 10;
    bool enable_correction_ = false;

    WGPUBuffer physics_buffer_ = nullptr;
    uint64 physics_size_ = 0;

    std::unique_ptr<gpu::GPUBuffer<simulate::SolverParams>> params_buffer_;
    simulate::SolverParams params_{};

    // Solver buffers
    std::unique_ptr<gpu::GPUBuffer<float32>> x_old_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> s_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> q_buffer_;
    std::unique_ptr<gpu::GPUBuffer<float32>> gradient_buffer_;      // N×4 atomic u32
    std::unique_ptr<gpu::GPUBuffer<float32>> hessian_diag_buffer_;  // N×9 atomic u32
    std::unique_ptr<gpu::GPUBuffer<float32>> correction_buffer_;    // N×9 f32

    // Pipelines (reuse pd_init, pd_predict, pd_copy from ext_pd_common)
    gpu::GPUComputePipeline pd_init_pipeline_;
    gpu::GPUComputePipeline pd_predict_pipeline_;
    gpu::GPUComputePipeline pd_copy_pipeline_;
    gpu::GPUComputePipeline accum_inertia_pipeline_;
    gpu::GPUComputePipeline local_solve_pipeline_;

    // Cached bind groups
    gpu::GPUBindGroup bg_init_;
    gpu::GPUBindGroup bg_predict_;
    gpu::GPUBindGroup bg_copy_q_from_s_;
    gpu::GPUBindGroup bg_accum_inertia_;
    gpu::GPUBindGroup bg_local_solve_;

    static constexpr uint32 kWorkgroupSize = 64;
};

}  // namespace ext_jgs2
