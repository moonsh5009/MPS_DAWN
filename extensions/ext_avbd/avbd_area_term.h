#pragma once

#include "ext_avbd/avbd_term.h"
#include "ext_dynamics/area_types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_avbd {

// Face adjacency entry for vertex-centric gather.
struct FaceAdjacency {
    mps::uint32 face_idx = 0;
    mps::uint32 vertex_role = 0;  // 0=n0, 1=n1, 2=n2
};

// Area params uniform (16-byte aligned for GPU).
struct alignas(16) AVBDAreaParams {
    mps::float32 stiffness = 1.0f;        // area preservation (bulk modulus k)
    mps::float32 shear_stiffness = 0.0f;  // ARAP shear modulus (μ)
    mps::float32 _pad1 = 0.0f;
    mps::float32 _pad2 = 0.0f;
};

// SVD-based FEM area term for AVBD solver.
// Vertex-centric: each thread gathers from incident faces via face CSR.
class AVBDAreaTerm : public IAVBDTerm {
public:
    void SetAreaData(const std::vector<ext_dynamics::AreaTriangle>& triangles,
                     mps::uint32 node_count,
                     mps::float32 stretch_k, mps::float32 shear_mu);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(const AVBDTermContext& ctx) override;
    void AccumulateColor(WGPUCommandEncoder encoder, mps::uint32 color_index) override;
    void Shutdown() override;

private:
    std::vector<ext_dynamics::AreaTriangle> triangles_;
    std::vector<mps::uint32> face_offsets_;
    std::vector<FaceAdjacency> face_adjacency_;
    mps::float32 stretch_stiffness_ = 1.0f;
    mps::float32 shear_stiffness_ = 0.0f;

    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::AreaTriangle>> triangle_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> face_offsets_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<FaceAdjacency>> face_adjacency_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<AVBDAreaParams>> area_params_buf_;

    mps::gpu::GPUComputePipeline pipeline_;
    std::vector<mps::gpu::GPUBindGroup> bg_per_color_;
    std::vector<mps::uint32> color_vertex_counts_;

    static const std::string kName;
    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_avbd
