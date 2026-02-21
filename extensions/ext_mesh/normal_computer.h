#pragma once

#include "core_util/types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace ext_mesh {

// Normal computation params — 16 bytes, layout-compatible with WGSL NormalParams.
struct alignas(16) NormalParams {
    mps::uint32 node_count = 0;
    mps::uint32 face_count = 0;
};

// Vertex normal computation from triangle meshes.
// Uses fixed-point i32 atomics for scatter, then normalizes to unit vec4f.
class NormalComputer {
public:
    NormalComputer();
    ~NormalComputer();

    void Initialize(mps::uint32 node_count, mps::uint32 face_count, mps::uint32 workgroup_size = 64);

    // Record normal computation passes into encoder.
    // position_buffer: array<vec4f>, face_buffer: array<Face{n0,n1,n2,pad}>
    void Compute(WGPUCommandEncoder encoder,
                 WGPUBuffer position_buffer, mps::uint64 position_size,
                 WGPUBuffer face_buffer, mps::uint64 face_size);

    [[nodiscard]] WGPUBuffer GetNormalBuffer() const;

    void Shutdown();

private:
    void CreateBuffers();
    void CreatePipelines();

    mps::uint32 node_count_ = 0;
    mps::uint32 face_count_ = 0;
    mps::uint32 workgroup_size_ = 64;
    mps::uint32 node_wg_count_ = 0;
    mps::uint32 face_wg_count_ = 0;

    // Params uniform (owned internally)
    std::unique_ptr<mps::gpu::GPUBuffer<NormalParams>> params_buffer_;

    std::unique_ptr<mps::gpu::GPUBuffer<mps::int32>> normal_atomic_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> normal_out_;

    mps::gpu::GPUComputePipeline clear_pipeline_;
    mps::gpu::GPUComputePipeline scatter_pipeline_;
    mps::gpu::GPUComputePipeline normalize_pipeline_;
};

}  // namespace ext_mesh
