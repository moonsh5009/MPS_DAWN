#include "ext_newton/area_term.h"
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

using ext_dynamics::AreaTriangle;
using ext_dynamics::FaceCSRMapping;

namespace ext_newton {

const std::string AreaTerm::kName = "AreaTerm";

AreaTerm::AreaTerm(const std::vector<AreaTriangle>& triangles, float32 stiffness)
    : triangles_(triangles), stiffness_(stiffness) {}

const std::string& AreaTerm::GetName() const { return kName; }

void AreaTerm::DeclareSparsity(simulate::SparsityBuilder& builder) {
    for (const auto& tri : triangles_) {
        builder.AddEdge(tri.n0, tri.n1);
        builder.AddEdge(tri.n1, tri.n2);
        builder.AddEdge(tri.n0, tri.n2);
    }
}

void AreaTerm::Initialize(const simulate::SparsityBuilder& sparsity, const simulate::AssemblyContext& ctx) {
    uint32 F = static_cast<uint32>(triangles_.size());
    nnz_ = sparsity.GetNNZ();

    // Build face-to-CSR mapping
    face_csr_mappings_.resize(F);
    for (uint32 f = 0; f < F; ++f) {
        uint32 a = triangles_[f].n0;
        uint32 b = triangles_[f].n1;
        uint32 c = triangles_[f].n2;
        face_csr_mappings_[f].csr_01 = sparsity.GetCSRIndex(a, b);
        face_csr_mappings_[f].csr_10 = sparsity.GetCSRIndex(b, a);
        face_csr_mappings_[f].csr_02 = sparsity.GetCSRIndex(a, c);
        face_csr_mappings_[f].csr_20 = sparsity.GetCSRIndex(c, a);
        face_csr_mappings_[f].csr_12 = sparsity.GetCSRIndex(b, c);
        face_csr_mappings_[f].csr_21 = sparsity.GetCSRIndex(c, b);
    }

    // Upload triangle buffer
    triangle_buffer_ = std::make_unique<GPUBuffer<AreaTriangle>>(
        BufferUsage::Storage, std::span<const AreaTriangle>(triangles_), "area_triangles");

    // Upload face CSR mapping buffer
    face_csr_buffer_ = std::make_unique<GPUBuffer<FaceCSRMapping>>(
        BufferUsage::Storage, std::span<const FaceCSRMapping>(face_csr_mappings_), "area_face_csr");

    // Upload area params uniform
    AreaParams params;
    params.stiffness = stiffness_;
    params.shear_stiffness = stiffness_ * 0.5f;  // 50% of area stiffness for shear resistance
    area_params_buffer_ = std::make_unique<GPUBuffer<AreaParams>>(
        BufferUsage::Uniform, std::span<const AreaParams>(&params, 1), "area_params");

    // Create pipeline
    auto shader = ShaderLoader::CreateModule("ext_newton/accumulate_area.wgsl", "accumulate_area");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "accumulate_area";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));

    // Cache bind group
    uint64 pos_sz = uint64(ctx.node_count) * 4 * sizeof(float32);
    uint64 force_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 tri_sz = uint64(F) * sizeof(AreaTriangle);
    uint64 diag_sz = uint64(ctx.node_count) * 9 * sizeof(float32);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    uint64 csr_map_sz = uint64(F) * sizeof(FaceCSRMapping);

    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_area_ = BindGroupBuilder("bg_area")
        .AddBuffer(0, ctx.physics_buffer, ctx.physics_size)
        .AddBuffer(1, ctx.params_buffer, ctx.params_size)
        .AddBuffer(2, ctx.position_buffer, pos_sz)
        .AddBuffer(3, ctx.force_buffer, force_sz)
        .AddBuffer(4, triangle_buffer_->GetHandle(), tri_sz)
        .AddBuffer(5, ctx.diag_buffer, diag_sz)
        .AddBuffer(6, area_params_buffer_->GetHandle(), sizeof(AreaParams))
        .AddBuffer(7, ctx.csr_values_buffer, csr_val_sz)
        .AddBuffer(8, face_csr_buffer_->GetHandle(), csr_map_sz)
        .Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);

    wg_count_ = (F + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("AreaTerm: initialized (", F, " triangles, nnz=", nnz_, ", stiffness=", stiffness_, ")");
}

void AreaTerm::Assemble(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_area_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void AreaTerm::Shutdown() {
    bg_area_ = {};
    pipeline_ = {};
    triangle_buffer_.reset();
    face_csr_buffer_.reset();
    area_params_buffer_.reset();
    LogInfo("AreaTerm: shutdown");
}

}  // namespace ext_newton
