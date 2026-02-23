#include "ext_jgs2/jgs2_spring_term.h"
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

using ext_dynamics::SpringEdge;

namespace ext_jgs2 {

const std::string JGS2SpringTerm::kName = "JGS2SpringTerm";

JGS2SpringTerm::JGS2SpringTerm(const std::vector<SpringEdge>& edges, float32 stiffness)
    : edges_(edges), stiffness_(stiffness) {}

const std::string& JGS2SpringTerm::GetName() const { return kName; }

void JGS2SpringTerm::Initialize(const JGS2AssemblyContext& ctx) {
    uint32 E = static_cast<uint32>(edges_.size());

    // Upload edge data
    edge_buffer_ = std::make_unique<GPUBuffer<SpringEdge>>(
        BufferUsage::Storage, std::span<const SpringEdge>(edges_), "jgs2_spring_edges");

    // Upload spring params uniform
    JGS2SpringParams params;
    params.stiffness = stiffness_;
    spring_params_buffer_ = std::make_unique<GPUBuffer<JGS2SpringParams>>(
        BufferUsage::Uniform, std::span<const JGS2SpringParams>(&params, 1), "jgs2_spring_params");

    // Create pipeline
    auto shader = ShaderLoader::CreateModule("ext_jgs2/jgs2_accum_spring.wgsl", "jgs2_accum_spring");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "jgs2_accum_spring";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));

    // Cache bind group
    uint64 vec_sz = uint64(ctx.node_count) * 4 * sizeof(float32);
    uint64 edge_sz = uint64(E) * sizeof(SpringEdge);
    uint64 grad_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 hess_sz = uint64(ctx.node_count) * 9 * sizeof(float32);

    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_springs_ = BindGroupBuilder("bg_jgs2_springs")
        .AddBuffer(0, ctx.params_buffer, ctx.params_size)
        .AddBuffer(1, ctx.q_buffer, vec_sz)
        .AddBuffer(2, edge_buffer_->GetHandle(), edge_sz)
        .AddBuffer(3, spring_params_buffer_->GetHandle(), sizeof(JGS2SpringParams))
        .AddBuffer(4, ctx.gradient_buffer, grad_sz)
        .AddBuffer(5, ctx.hessian_diag_buffer, hess_sz)
        .Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);

    wg_count_ = (E + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("JGS2SpringTerm: initialized (", E, " edges, k=", stiffness_, ")");
}

void JGS2SpringTerm::Accumulate(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_springs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void JGS2SpringTerm::Shutdown() {
    bg_springs_ = {};
    pipeline_ = {};
    edge_buffer_.reset();
    spring_params_buffer_.reset();
    LogInfo("JGS2SpringTerm: shutdown");
}

}  // namespace ext_jgs2
