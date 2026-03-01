#pragma once

#include "ext_avbd/avbd_term.h"
#include "ext_dynamics/spring_types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ext_avbd {

// CSR adjacency entry: one neighbor with rest length and edge index for AL penalty lookup.
struct SpringNeighbor {
    mps::uint32 neighbor_idx = 0;
    mps::float32 rest_length = 0.0f;
    mps::uint32 edge_index = 0;
};

// Augmented Lagrangian parameters uniform (16-byte aligned for GPU).
struct alignas(16) ALParams {
    mps::float32 stiffness = 0.0f;   // k (material stiffness, penalty cap)
    mps::float32 gamma = 0.99f;      // warmstart decay factor
    mps::float32 beta = 100000.0f;   // penalty ramp rate per iteration
    mps::uint32 edge_count = 0;
};

// Spring energy term for AVBD solver with Augmented Lagrangian penalty ramping.
// Vertex-centric: each thread gathers gradient/Hessian from CSR neighbors.
// Per-edge penalty parameter ramps from 0 toward stiffness k across iterations.
// Both gradient and Hessian use penalty (not k) for consistent Newton steps.
class AVBDSpringTerm : public IAVBDTerm {
public:
    void SetSpringData(std::span<const mps::uint32> offsets,
                       std::span<const SpringNeighbor> neighbors,
                       std::span<const ext_dynamics::SpringEdge> edges,
                       mps::float32 stiffness,
                       mps::float32 al_gamma,
                       mps::float32 al_beta);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(const AVBDTermContext& ctx) override;
    void AccumulateColor(WGPUCommandEncoder encoder, mps::uint32 color_index) override;
    void DualUpdate(WGPUCommandEncoder encoder) override;
    void WarmstartDecay(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    // CSR data
    std::vector<mps::uint32> offsets_;
    std::vector<SpringNeighbor> neighbors_;
    mps::float32 stiffness_ = 0.0f;

    // AL data
    std::vector<ext_dynamics::SpringEdge> edges_;
    mps::float32 al_gamma_ = 0.99f;
    mps::float32 al_beta_ = 100000.0f;
    mps::uint32 edge_count_ = 0;

    // CSR buffers
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> offsets_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<SpringNeighbor>> neighbors_buf_;

    // AL buffers
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> penalty_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::SpringEdge>> edge_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<ALParams>> al_params_buf_;

    // Accumulation pipeline + per-color bind groups
    mps::gpu::GPUComputePipeline pipeline_;
    std::vector<mps::gpu::GPUBindGroup> bg_per_color_;
    std::vector<mps::uint32> color_vertex_counts_;

    // AL pipelines + bind groups
    mps::gpu::GPUComputePipeline warmstart_pipeline_;
    mps::gpu::GPUComputePipeline penalty_ramp_pipeline_;
    mps::gpu::GPUBindGroup bg_warmstart_;
    mps::gpu::GPUBindGroup bg_penalty_ramp_;

    static const std::string kName;
    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_avbd
