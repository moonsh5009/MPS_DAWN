#pragma once

#include "ext_avbd/avbd_term.h"
#include "core_util/types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include "core_simulate/solver_params.h"
#include <memory>
#include <vector>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;
struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;

namespace ext_avbd {

// VBD (Vertex Block Descent) dynamics solver.
// Multi-pass architecture: per color group dispatches terms → local_solve (inertia fused).
// Graph coloring ensures no write conflicts — no atomics needed.
// Augmented Lagrangian: per-term dual variables accelerate convergence across iterations.
class VBDDynamics {
public:
    void Initialize(mps::uint32 node_count, mps::uint32 edge_count, mps::uint32 face_count,
                    mps::uint32 iterations,
                    WGPUBuffer physics_buf, mps::uint64 physics_sz,
                    WGPUBuffer pos_buf, WGPUBuffer vel_buf, WGPUBuffer mass_buf,
                    mps::uint64 pos_sz, mps::uint64 vel_sz, mps::uint64 mass_sz,
                    const std::vector<mps::uint32>& color_offsets,
                    WGPUBuffer vertex_order_buf, mps::uint64 vertex_order_sz);

    void AddTerm(std::unique_ptr<IAVBDTerm> term);
    void Solve(WGPUCommandEncoder encoder);
    void Shutdown();

    [[nodiscard]] WGPUBuffer GetQBuffer() const;
    [[nodiscard]] WGPUBuffer GetXOldBuffer() const;
    [[nodiscard]] WGPUBuffer GetSolverParamsBuffer() const;
    [[nodiscard]] mps::uint64 GetSolverParamsSize() const;

private:
    // Pre-solve pipelines
    mps::gpu::GPUComputePipeline init_pipeline_;
    mps::gpu::GPUComputePipeline predict_pipeline_;
    mps::gpu::GPUComputePipeline copy_q_pipeline_;

    // Per-color pipelines
    mps::gpu::GPUComputePipeline local_solve_pipeline_;

    // Working buffers
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> x_old_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> s_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> q_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> gradient_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> hessian_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::simulate::SolverParams>> solver_params_buf_;

    // Pre-solve bind groups
    mps::gpu::GPUBindGroup bg_init_;
    mps::gpu::GPUBindGroup bg_predict_;
    mps::gpu::GPUBindGroup bg_copy_;

    // Per-color bind groups
    std::vector<mps::gpu::GPUBindGroup> bg_local_solve_;

    // Per-color uniform buffers
    struct alignas(16) VBDColorParams {
        mps::uint32 color_offset;
        mps::uint32 color_vertex_count;
        mps::uint32 _pad0;
        mps::uint32 _pad1;
    };
    std::vector<std::unique_ptr<mps::gpu::GPUBuffer<VBDColorParams>>> color_params_bufs_;

    // Color groups (exposed to terms via AddTerm)
    std::vector<AVBDColorGroup> color_groups_;

    // Terms
    std::vector<std::unique_ptr<IAVBDTerm>> terms_;

    // State
    std::vector<mps::uint32> color_offsets_;
    mps::uint32 color_count_ = 0;
    mps::uint32 node_count_ = 0;
    mps::uint32 edge_count_ = 0;
    mps::uint32 face_count_ = 0;
    mps::uint32 iterations_ = 10;

    // Cached for term context creation
    mps::uint64 q_sz_ = 0;
    mps::uint64 gradient_sz_ = 0;
    mps::uint64 hessian_sz_ = 0;
    WGPUBuffer vertex_order_buf_ = nullptr;
    mps::uint64 vertex_order_sz_ = 0;

    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_avbd
