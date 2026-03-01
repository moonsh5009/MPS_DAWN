#include "ext_avbd/graph_coloring.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cstring>
#include <vector>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;

namespace ext_avbd {

// ============================================================================
// Helpers
// ============================================================================

GPUComputePipeline GraphColoring::CreatePipeline(const std::string& shader_path,
                                                  const std::string& label) {
    auto shader = ShaderLoader::CreateModule("ext_avbd/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

static GPUBindGroup MakeBindGroup(const GPUComputePipeline& pipeline,
                                   const std::string& label,
                                   std::initializer_list<std::pair<uint32, std::pair<WGPUBuffer, uint64>>> entries) {
    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline.GetHandle(), 0);
    auto builder = BindGroupBuilder(label);
    for (auto& [binding, buf_size] : entries) {
        builder = std::move(builder).AddBuffer(binding, buf_size.first, buf_size.second);
    }
    auto bg = std::move(builder).Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);
    return bg;
}

// Synchronous readback of N u32 values from a GPU buffer
static std::vector<uint32> ReadbackBufferU32(WGPUBuffer src_buffer, uint32 count) {
    auto& core = GPUCore::GetInstance();
    uint64 byte_size = static_cast<uint64>(count) * 4;

    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.size = byte_size;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(core.GetDevice(), &staging_desc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src_buffer, 0, staging, 0, byte_size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapCtx { bool done = false; bool ok = false; };
    MapCtx ctx;

    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    map_cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* c = static_cast<MapCtx*>(ud1);
        c->done = true;
        c->ok = (status == WGPUMapAsyncStatus_Success);
    };
    map_cb.userdata1 = &ctx;

    WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, byte_size, map_cb);

#ifndef __EMSCRIPTEN__
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(core.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
    while (!ctx.done) { core.ProcessEvents(); }
#endif

    std::vector<uint32> result(count, 0);
    if (ctx.ok) {
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, byte_size);
        std::memcpy(result.data(), mapped, byte_size);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
    return result;
}

// Synchronous readback of a single u32 from a GPU buffer
static uint32 ReadbackU32(WGPUBuffer src_buffer, uint64 offset = 0) {
    auto& core = GPUCore::GetInstance();

    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.size = 4;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(core.GetDevice(), &staging_desc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src_buffer, offset, staging, 0, 4);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    struct MapCtx { bool done = false; bool ok = false; };
    MapCtx ctx;

    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    map_cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* c = static_cast<MapCtx*>(ud1);
        c->done = true;
        c->ok = (status == WGPUMapAsyncStatus_Success);
    };
    map_cb.userdata1 = &ctx;

    WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, 4, map_cb);

#ifndef __EMSCRIPTEN__
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(core.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
    while (!ctx.done) { core.ProcessEvents(); }
#endif

    uint32 result = 0;
    if (ctx.ok) {
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, 4);
        std::memcpy(&result, mapped, 4);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
    return result;
}

// ============================================================================
// Build
// ============================================================================

void GraphColoring::Build(uint32 node_count, uint32 edge_count,
                           WGPUBuffer edge_buffer, uint64 edge_buffer_size) {
    node_count_ = node_count;
    edge_count_ = edge_count;

    // Create pipelines
    count_degrees_pipeline_ = CreatePipeline("gc_count_degrees.wgsl", "gc_count_degrees");
    prefix_sum_local_pipeline_ = CreatePipeline("gc_prefix_sum_local.wgsl", "gc_prefix_sum_local");
    prefix_sum_propagate_pipeline_ = CreatePipeline("gc_prefix_sum_propagate.wgsl", "gc_prefix_sum_propagate");
    fill_adjacency_pipeline_ = CreatePipeline("gc_fill_adjacency.wgsl", "gc_fill_adjacency");
    color_vertices_pipeline_ = CreatePipeline("gc_color_vertices.wgsl", "gc_color_vertices");
    find_max_color_pipeline_ = CreatePipeline("gc_find_max_color.wgsl", "gc_find_max_color");

    // Create main params buffer
    ColoringParams main_params{node_count_, edge_count_, 0, 0};
    params_buf_ = GPUBuffer<ColoringParams>(
        BufferUsage::Uniform | BufferUsage::CopyDst,
        std::span<const ColoringParams>(&main_params, 1), "gc_params");

    // Create buffers
    degree_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst,
        .size = static_cast<uint64>(node_count_) * 4,
        .label = "gc_degrees"
    });

    row_ptr_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
        .size = static_cast<uint64>(node_count_ + 1) * 4,
        .label = "gc_row_ptr"
    });

    col_idx_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage,
        .size = static_cast<uint64>(edge_count_) * 2 * 4,
        .label = "gc_col_idx"
    });

    color_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
        .size = static_cast<uint64>(node_count_) * 4,
        .label = "gc_colors"
    });

    flag_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
        .size = 4,
        .label = "gc_flag"
    });

    max_color_buf_ = GPUBuffer<uint32>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
        .size = 4,
        .label = "gc_max_color"
    });

    // Phase A: Build CSR adjacency
    BuildCSR(edge_buffer, edge_buffer_size);

    // Phase B: Greedy coloring
    RunColoring();
}

// ============================================================================
// Phase A: Build CSR Adjacency
// ============================================================================

void GraphColoring::BuildCSR(WGPUBuffer edge_buffer, uint64 edge_buffer_size) {
    auto& gpu = GPUCore::GetInstance();

    uint32 edge_wg = (edge_count_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint64 degree_size = static_cast<uint64>(node_count_) * 4;
    uint64 row_ptr_size = static_cast<uint64>(node_count_ + 1) * 4;
    uint64 params_size = sizeof(ColoringParams);

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"gc_csr_build", 12};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    // 1. Clear degree buffer
    wgpuCommandEncoderClearBuffer(encoder, degree_buf_.GetHandle(), 0, degree_size);

    // 2. Count degrees
    {
        auto bg = MakeBindGroup(count_degrees_pipeline_, "bg_count_degrees",
            {{0, {params_buf_.GetHandle(), params_size}},
             {1, {edge_buffer, edge_buffer_size}},
             {2, {degree_buf_.GetHandle(), degree_size}}});

        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(count_degrees_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg.GetHandle());
        enc.Dispatch(edge_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // 3. Clear row_ptr and copy degrees into it
    wgpuCommandEncoderClearBuffer(encoder, row_ptr_buf_.GetHandle(), 0, row_ptr_size);

    // Submit this encoder to execute count_degrees before copy
    WGPUCommandBuffer cmd1 = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd1);
    wgpuCommandBufferRelease(cmd1);
    wgpuCommandEncoderRelease(encoder);

    // 4. Copy degrees → row_ptr[0..N-1]
    {
        WGPUCommandEncoderDescriptor enc2_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder enc2 = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc2_desc);
        wgpuCommandEncoderCopyBufferToBuffer(enc2, degree_buf_.GetHandle(), 0,
                                              row_ptr_buf_.GetHandle(), 0, degree_size);
        WGPUCommandBuffer cmd2 = wgpuCommandEncoderFinish(enc2, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd2);
        wgpuCommandBufferRelease(cmd2);
        wgpuCommandEncoderRelease(enc2);
    }

    // 5. Exclusive prefix sum on row_ptr (N+1 elements)
    DispatchPrefixSum(row_ptr_buf_.GetHandle(), node_count_ + 1);

    // 6. Clear degree buffer (reuse as write counters) and fill adjacency
    {
        WGPUCommandEncoderDescriptor enc3_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        enc3_desc.label = {"gc_fill_adj", 11};
        WGPUCommandEncoder enc3 = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc3_desc);

        wgpuCommandEncoderClearBuffer(enc3, degree_buf_.GetHandle(), 0, degree_size);

        // 7. Fill adjacency
        uint64 col_idx_size = static_cast<uint64>(edge_count_) * 2 * 4;
        auto bg = MakeBindGroup(fill_adjacency_pipeline_, "bg_fill_adj",
            {{0, {params_buf_.GetHandle(), params_size}},
             {1, {edge_buffer, edge_buffer_size}},
             {2, {row_ptr_buf_.GetHandle(), row_ptr_size}},
             {3, {degree_buf_.GetHandle(), degree_size}},
             {4, {col_idx_buf_.GetHandle(), col_idx_size}}});

        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc3, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(fill_adjacency_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg.GetHandle());
        enc.Dispatch(edge_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBuffer cmd3 = wgpuCommandEncoderFinish(enc3, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd3);
        wgpuCommandBufferRelease(cmd3);
        wgpuCommandEncoderRelease(enc3);
    }
}

// ============================================================================
// Multi-level Prefix Sum
// ============================================================================

void GraphColoring::DispatchPrefixSum(WGPUBuffer buffer, uint32 element_count) {
    auto& gpu = GPUCore::GetInstance();

    // Determine how many levels we need
    struct LevelInfo {
        uint32 count;       // number of elements at this level
        uint32 num_blocks;  // number of workgroups
    };
    std::vector<LevelInfo> levels;

    uint32 count = element_count;
    while (count > 1) {
        uint32 blocks = (count + kWorkgroupSize - 1) / kWorkgroupSize;
        levels.push_back({count, blocks});
        count = blocks;
    }

    if (levels.empty()) return;

    // Create scratch buffers and params for each level
    scan_scratch_.clear();
    scan_params_.clear();

    for (uint32 i = 0; i < static_cast<uint32>(levels.size()); ++i) {
        uint64 scratch_size = static_cast<uint64>(levels[i].num_blocks) * 4;
        scan_scratch_.emplace_back(BufferConfig{
            .usage = BufferUsage::Storage | BufferUsage::CopyDst,
            .size = scratch_size,
            .label = "gc_scan_scratch_" + std::to_string(i)
        });

        ColoringParams sp{0, 0, levels[i].count, 0};
        scan_params_.emplace_back(
            BufferUsage::Uniform | BufferUsage::CopyDst,
            std::span<const ColoringParams>(&sp, 1),
            "gc_scan_params_" + std::to_string(i));
    }

    // Forward pass: local prefix sum at each level
    // Level 0 operates on 'buffer', higher levels on scratch[level-1]
    for (uint32 i = 0; i < static_cast<uint32>(levels.size()); ++i) {
        WGPUBuffer data_buf = (i == 0) ? buffer : scan_scratch_[i - 1].GetHandle();
        uint64 data_size = static_cast<uint64>(levels[i].count) * 4;
        uint64 scratch_size = static_cast<uint64>(levels[i].num_blocks) * 4;
        uint64 params_size = sizeof(ColoringParams);

        auto bg = MakeBindGroup(prefix_sum_local_pipeline_,
            "bg_scan_local_" + std::to_string(i),
            {{0, {scan_params_[i].GetHandle(), params_size}},
             {1, {data_buf, data_size}},
             {2, {scan_scratch_[i].GetHandle(), scratch_size}}});

        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(prefix_sum_local_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg.GetHandle());
        enc.Dispatch(levels[i].num_blocks);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    // Back-propagate: add scanned block sums back to each level
    // Go from second-to-last level down to level 0
    for (int32 i = static_cast<int32>(levels.size()) - 2; i >= 0; --i) {
        WGPUBuffer data_buf = (i == 0) ? buffer : scan_scratch_[i - 1].GetHandle();
        uint64 data_size = static_cast<uint64>(levels[i].count) * 4;
        uint64 offsets_size = static_cast<uint64>(levels[i].num_blocks) * 4;
        uint64 params_size = sizeof(ColoringParams);

        auto bg = MakeBindGroup(prefix_sum_propagate_pipeline_,
            "bg_scan_prop_" + std::to_string(i),
            {{0, {scan_params_[i].GetHandle(), params_size}},
             {1, {data_buf, data_size}},
             {2, {scan_scratch_[i].GetHandle(), offsets_size}}});

        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(prefix_sum_propagate_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg.GetHandle());
        enc.Dispatch(levels[i].num_blocks);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }
}

// ============================================================================
// Phase B: Greedy Coloring
// ============================================================================

void GraphColoring::RunColoring() {
    auto& gpu = GPUCore::GetInstance();

    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;
    uint64 params_size = sizeof(ColoringParams);
    uint64 row_ptr_size = static_cast<uint64>(node_count_ + 1) * 4;
    uint64 col_idx_size = static_cast<uint64>(edge_count_) * 2 * 4;
    uint64 color_size = static_cast<uint64>(node_count_) * 4;

    // Initialize color buffer to UNCOLORED (0xFFFFFFFF)
    {
        std::vector<uint32> init_colors(node_count_, 0xFFFFFFFF);
        color_buf_.WriteData(std::span<const uint32>(init_colors));
    }

    // Build bind group for coloring
    auto bg_color = MakeBindGroup(color_vertices_pipeline_, "bg_color_vertices",
        {{0, {params_buf_.GetHandle(), params_size}},
         {1, {row_ptr_buf_.GetHandle(), row_ptr_size}},
         {2, {col_idx_buf_.GetHandle(), col_idx_size}},
         {3, {color_buf_.GetHandle(), color_size}},
         {4, {flag_buf_.GetHandle(), 4}}});

    // Iterative coloring loop (batches of 64 iterations)
    static constexpr uint32 kBatchSize = 64;
    static constexpr uint32 kMaxBatches = 16;  // safety limit (64*16 = 1024 iterations)

    for (uint32 batch = 0; batch < kMaxBatches; ++batch) {
        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        enc_desc.label = {"gc_coloring_batch", 17};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

        // First (kBatchSize - 1) iterations: no flag check
        for (uint32 iter = 0; iter < kBatchSize - 1; ++iter) {
            WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
            ComputeEncoder enc(pass);
            enc.SetPipeline(color_vertices_pipeline_.GetHandle());
            enc.SetBindGroup(0, bg_color.GetHandle());
            enc.Dispatch(node_wg);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // Clear flag before last iteration
        wgpuCommandEncoderClearBuffer(encoder, flag_buf_.GetHandle(), 0, 4);

        // Last iteration: will set flag if any vertex remains uncolored
        {
            WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
            ComputeEncoder enc(pass);
            enc.SetPipeline(color_vertices_pipeline_.GetHandle());
            enc.SetBindGroup(0, bg_color.GetHandle());
            enc.Dispatch(node_wg);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // Submit
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);

        // Read back flag
        uint32 flag_val = ReadbackU32(flag_buf_.GetHandle());
        if (flag_val == 0) {
            LogInfo("GraphColoring: converged in ", (batch + 1) * kBatchSize, " iterations");
            break;
        }
    }

    // Find max color
    {
        // Clear max_color_buf
        uint32 zero = 0;
        max_color_buf_.WriteData(std::span<const uint32>(&zero, 1));

        auto bg_max = MakeBindGroup(find_max_color_pipeline_, "bg_find_max_color",
            {{0, {params_buf_.GetHandle(), params_size}},
             {1, {color_buf_.GetHandle(), color_size}},
             {2, {max_color_buf_.GetHandle(), 4}}});

        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(find_max_color_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_max.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);

        uint32 max_color = ReadbackU32(max_color_buf_.GetHandle());
        color_count_ = max_color + 1;
    }

    LogInfo("GraphColoring: ", node_count_, " vertices, ", edge_count_, " edges, ",
            "ColorSize: ", color_count_);
}

// ============================================================================
// Accessors
// ============================================================================

WGPUBuffer GraphColoring::GetColorBuffer() const {
    return color_buf_.GetHandle();
}

uint32 GraphColoring::GetColorCount() const {
    return color_count_;
}

WGPUBuffer GraphColoring::GetRowPtrBuffer() const {
    return row_ptr_buf_.GetHandle();
}

WGPUBuffer GraphColoring::GetColIdxBuffer() const {
    return col_idx_buf_.GetHandle();
}

// ============================================================================
// BuildColorGroups
// ============================================================================

void GraphColoring::BuildColorGroups() {
    if (node_count_ == 0 || color_count_ == 0) return;

    // 1. Readback color_buf_ (N u32s)
    auto colors = ReadbackBufferU32(color_buf_.GetHandle(), node_count_);

    // 2. Counting sort by color
    std::vector<uint32> color_counts(color_count_, 0);
    for (auto c : colors) {
        if (c < color_count_) color_counts[c]++;
    }

    // Build offsets (prefix sum)
    color_offsets_.resize(color_count_ + 1, 0);
    for (uint32 c = 0; c < color_count_; ++c) {
        color_offsets_[c + 1] = color_offsets_[c] + color_counts[c];
    }

    // Fill vertex_order
    std::vector<uint32> vertex_order(node_count_);
    std::vector<uint32> write_pos(color_count_, 0);
    for (uint32 v = 0; v < node_count_; ++v) {
        uint32 c = colors[v];
        if (c < color_count_) {
            vertex_order[color_offsets_[c] + write_pos[c]] = v;
            write_pos[c]++;
        }
    }

    // 3. Upload vertex_order to GPU
    vertex_order_buf_ = GPUBuffer<uint32>(
        BufferUsage::Storage,
        std::span<const uint32>(vertex_order),
        "avbd_vertex_order");

    LogInfo("GraphColoring::BuildColorGroups: ", color_count_, " colors, vertex_order uploaded");
}

WGPUBuffer GraphColoring::GetVertexOrderBuffer() const {
    return vertex_order_buf_.GetHandle();
}

const std::vector<uint32>& GraphColoring::GetColorOffsets() const {
    return color_offsets_;
}

// ============================================================================
// Shutdown
// ============================================================================

void GraphColoring::Shutdown() {
    degree_buf_ = GPUBuffer<uint32>(BufferConfig{});
    row_ptr_buf_ = GPUBuffer<uint32>(BufferConfig{});
    col_idx_buf_ = GPUBuffer<uint32>(BufferConfig{});
    color_buf_ = GPUBuffer<uint32>(BufferConfig{});
    flag_buf_ = GPUBuffer<uint32>(BufferConfig{});
    max_color_buf_ = GPUBuffer<uint32>(BufferConfig{});
    params_buf_ = GPUBuffer<ColoringParams>(BufferConfig{});

    scan_scratch_.clear();
    scan_params_.clear();

    vertex_order_buf_ = GPUBuffer<uint32>(BufferConfig{});
    color_offsets_.clear();

    count_degrees_pipeline_ = {};
    prefix_sum_local_pipeline_ = {};
    prefix_sum_propagate_pipeline_ = {};
    fill_adjacency_pipeline_ = {};
    color_vertices_pipeline_ = {};
    find_max_color_pipeline_ = {};

    node_count_ = 0;
    edge_count_ = 0;
    color_count_ = 0;
}

}  // namespace ext_avbd
