#include "ext_mesh/normal_computer.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <span>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;

namespace ext_mesh {

// ============================================================================
// Static helpers
// ============================================================================

static GPUBindGroup MakeBG(const GPUComputePipeline& pipeline,
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

static void Dispatch(WGPUCommandEncoder encoder,
                     const GPUComputePipeline& pipeline,
                     const GPUBindGroup& bg,
                     uint32 workgroup_count) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline.GetHandle());
    enc.SetBindGroup(0, bg.GetHandle());
    enc.Dispatch(workgroup_count);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

static GPUComputePipeline MakePipeline(const std::string& shader_path,
                                        const std::string& label) {
    auto shader = ShaderLoader::CreateModule("ext_mesh/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// NormalComputer
// ============================================================================

NormalComputer::NormalComputer() = default;
NormalComputer::~NormalComputer() = default;

void NormalComputer::Initialize(uint32 node_count, uint32 face_count, uint32 workgroup_size) {
    node_count_ = node_count;
    face_count_ = face_count;
    workgroup_size_ = workgroup_size;
    node_wg_count_ = (node_count + workgroup_size - 1) / workgroup_size;
    face_wg_count_ = (face_count + workgroup_size - 1) / workgroup_size;

    CreateBuffers();
    CreatePipelines();
    LogInfo("NormalComputer: initialized (", node_count_, " nodes, ", face_count_, " faces)");
}

void NormalComputer::CreateBuffers() {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 n4 = uint64(node_count_) * 4;

    normal_atomic_ = std::make_unique<GPUBuffer<int32>>(
        BufferConfig{.usage = srw, .size = n4 * sizeof(int32), .label = "normals_atomic"});
    normal_out_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw | BufferUsage::Vertex, .size = n4 * sizeof(float32), .label = "normals"});

    // Create params buffer
    NormalParams params{};
    params.node_count = node_count_;
    params.face_count = face_count_;
    params_buffer_ = std::make_unique<GPUBuffer<NormalParams>>(
        BufferUsage::Uniform, std::span<const NormalParams>(&params, 1), "normal_params");
}

void NormalComputer::CreatePipelines() {
    clear_pipeline_ = MakePipeline("clear_normals.wgsl", "clear_normals");
    scatter_pipeline_ = MakePipeline("normals_scatter.wgsl", "scatter_normals");
    normalize_pipeline_ = MakePipeline("normals_normalize.wgsl", "normalize_normals");
}

void NormalComputer::Compute(WGPUCommandEncoder encoder,
                             WGPUBuffer position_buffer, uint64 position_size,
                             WGPUBuffer face_buffer, uint64 face_size) {
    uint64 normal_i32_sz = uint64(node_count_) * 4 * sizeof(int32);
    uint64 normal_f32_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 params_sz = sizeof(NormalParams);

    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer norm_i32 = normal_atomic_->GetHandle();
    WGPUBuffer norm_out = normal_out_->GetHandle();

    auto bg_clear = MakeBG(clear_pipeline_, "bg_clear_n",
        {{0, {params_h, params_sz}}, {1, {norm_i32, normal_i32_sz}}});

    auto bg_scatter = MakeBG(scatter_pipeline_, "bg_scatter_n",
        {{0, {params_h, params_sz}}, {1, {position_buffer, position_size}},
         {2, {face_buffer, face_size}}, {3, {norm_i32, normal_i32_sz}}});

    auto bg_normalize = MakeBG(normalize_pipeline_, "bg_norm_n",
        {{0, {params_h, params_sz}}, {1, {norm_i32, normal_i32_sz}}, {2, {norm_out, normal_f32_sz}}});

    Dispatch(encoder, clear_pipeline_, bg_clear, node_wg_count_);
    Dispatch(encoder, scatter_pipeline_, bg_scatter, face_wg_count_);
    Dispatch(encoder, normalize_pipeline_, bg_normalize, node_wg_count_);
}

WGPUBuffer NormalComputer::GetNormalBuffer() const {
    return normal_out_ ? normal_out_->GetHandle() : nullptr;
}

void NormalComputer::Shutdown() {
    clear_pipeline_ = {};
    scatter_pipeline_ = {};
    normalize_pipeline_ = {};
    normal_atomic_.reset();
    normal_out_.reset();
    params_buffer_.reset();
    LogInfo("NormalComputer: shutdown");
}

}  // namespace ext_mesh
