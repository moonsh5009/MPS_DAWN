#pragma once

#include "core_util/types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <vector>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace ext_avbd {

// GPU-based vertex graph coloring using greedy first-fit algorithm.
// Builds CSR adjacency from MeshEdge buffer, then iteratively colors vertices
// so no two adjacent vertices share the same color.
class GraphColoring {
public:
    // Build CSR adjacency and run greedy coloring on GPU.
    // edge_buffer: MeshEdge[] (n0, n1) GPU storage buffer.
    void Build(mps::uint32 node_count, mps::uint32 edge_count,
               WGPUBuffer edge_buffer, mps::uint64 edge_buffer_size);

    // Results (valid after Build)
    // Results (valid after Build)
    WGPUBuffer GetColorBuffer() const;     // u32[N] — color per vertex
    mps::uint32 GetColorCount() const;     // max_color + 1
    WGPUBuffer GetRowPtrBuffer() const;    // u32[N+1] — CSR row pointers
    WGPUBuffer GetColIdxBuffer() const;    // u32[2E] — CSR column indices

    // After Build(), call this to sort vertices by color (CPU readback)
    void BuildColorGroups();

    // Results (valid after BuildColorGroups)
    WGPUBuffer GetVertexOrderBuffer() const;
    const std::vector<mps::uint32>& GetColorOffsets() const;

    void Shutdown();

private:
    // Phase A: Build CSR adjacency
    void BuildCSR(WGPUBuffer edge_buffer, mps::uint64 edge_buffer_size);

    // Phase B: Greedy coloring
    void RunColoring();

    // Multi-level prefix sum
    void DispatchPrefixSum(WGPUBuffer buffer, mps::uint32 element_count);

    // Helper: create a compute pipeline
    mps::gpu::GPUComputePipeline CreatePipeline(const std::string& shader_path,
                                                 const std::string& label);

    // Params uniform struct (must match WGSL)
    struct alignas(16) ColoringParams {
        mps::uint32 node_count = 0;
        mps::uint32 edge_count = 0;
        mps::uint32 scan_count = 0;
        mps::uint32 padding = 0;
    };

    mps::uint32 node_count_ = 0;
    mps::uint32 edge_count_ = 0;
    mps::uint32 color_count_ = 0;

    // Buffers
    mps::gpu::GPUBuffer<mps::uint32> degree_buf_{mps::gpu::BufferConfig{}};
    mps::gpu::GPUBuffer<mps::uint32> row_ptr_buf_{mps::gpu::BufferConfig{}};
    mps::gpu::GPUBuffer<mps::uint32> col_idx_buf_{mps::gpu::BufferConfig{}};
    mps::gpu::GPUBuffer<mps::uint32> color_buf_{mps::gpu::BufferConfig{}};
    mps::gpu::GPUBuffer<mps::uint32> flag_buf_{mps::gpu::BufferConfig{}};
    mps::gpu::GPUBuffer<mps::uint32> max_color_buf_{mps::gpu::BufferConfig{}};

    // Prefix sum scratch buffers (one per level)
    std::vector<mps::gpu::GPUBuffer<mps::uint32>> scan_scratch_;
    // Per-level params uniform buffers
    std::vector<mps::gpu::GPUBuffer<ColoringParams>> scan_params_;

    // Main params buffer
    mps::gpu::GPUBuffer<ColoringParams> params_buf_{mps::gpu::BufferConfig{}};

    // Pipelines
    mps::gpu::GPUComputePipeline count_degrees_pipeline_;
    mps::gpu::GPUComputePipeline prefix_sum_local_pipeline_;
    mps::gpu::GPUComputePipeline prefix_sum_propagate_pipeline_;
    mps::gpu::GPUComputePipeline fill_adjacency_pipeline_;
    mps::gpu::GPUComputePipeline color_vertices_pipeline_;
    mps::gpu::GPUComputePipeline find_max_color_pipeline_;

    // Vertex order buffer (sorted by color) and per-color offsets
    mps::gpu::GPUBuffer<mps::uint32> vertex_order_buf_{mps::gpu::BufferConfig{}};
    std::vector<mps::uint32> color_offsets_;

    static constexpr mps::uint32 kWorkgroupSize = 256;
};

}  // namespace ext_avbd
