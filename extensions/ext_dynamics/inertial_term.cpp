#include "ext_dynamics/inertial_term.h"
#include "core_simulate/sim_components.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;

namespace ext_dynamics {

const std::string InertialTerm::kName = "InertialTerm";

const std::string& InertialTerm::GetName() const { return kName; }

void InertialTerm::Initialize(const SparsityBuilder& /* sparsity */, const AssemblyContext& ctx) {
    auto shader = ShaderLoader::CreateModule("ext_dynamics/inertia_assemble.wgsl", "inertia_assemble");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "inertia_assemble";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));

    // Cache bind group
    uint64 diag_sz = uint64(ctx.node_count) * 9 * sizeof(float32);
    uint64 mass_sz = uint64(ctx.node_count) * sizeof(SimMass);

    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_inertia_ = BindGroupBuilder("bg_inertia")
        .AddBuffer(0, ctx.params_buffer, ctx.params_size)
        .AddBuffer(1, ctx.diag_buffer, diag_sz)
        .AddBuffer(2, ctx.mass_buffer, mass_sz)
        .Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);

    wg_count_ = (ctx.node_count + ctx.workgroup_size - 1) / ctx.workgroup_size;
    LogInfo("InertialTerm: initialized");
}

void InertialTerm::Assemble(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_inertia_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void InertialTerm::Shutdown() {
    bg_inertia_ = {};
    pipeline_ = {};
    LogInfo("InertialTerm: shutdown");
}

}  // namespace ext_dynamics
