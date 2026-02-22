#include "ext_pd/pd_area_term.h"
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
using namespace mps::simulate;
using namespace ext_dynamics;
using ext_newton::AreaParams;

namespace ext_pd {

const std::string PDAreaTerm::kName = "PDAreaTerm";

PDAreaTerm::PDAreaTerm(const std::vector<AreaTriangle>& triangles, float32 stiffness)
    : triangles_(triangles), stiffness_(stiffness) {}

const std::string& PDAreaTerm::GetName() const { return kName; }

void PDAreaTerm::DeclareSparsity(SparsityBuilder& builder) {
    for (const auto& tri : triangles_) {
        builder.AddEdge(tri.n0, tri.n1);
        builder.AddEdge(tri.n0, tri.n2);
        builder.AddEdge(tri.n1, tri.n2);
    }
}

void PDAreaTerm::Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) {
    uint32 F = static_cast<uint32>(triangles_.size());
    nnz_ = sparsity.GetNNZ();

    // Build face-to-CSR mapping
    face_csr_mappings_.resize(F);
    for (uint32 f = 0; f < F; ++f) {
        uint32 n0 = triangles_[f].n0;
        uint32 n1 = triangles_[f].n1;
        uint32 n2 = triangles_[f].n2;
        face_csr_mappings_[f].csr_01 = sparsity.GetCSRIndex(n0, n1);
        face_csr_mappings_[f].csr_10 = sparsity.GetCSRIndex(n1, n0);
        face_csr_mappings_[f].csr_02 = sparsity.GetCSRIndex(n0, n2);
        face_csr_mappings_[f].csr_20 = sparsity.GetCSRIndex(n2, n0);
        face_csr_mappings_[f].csr_12 = sparsity.GetCSRIndex(n1, n2);
        face_csr_mappings_[f].csr_21 = sparsity.GetCSRIndex(n2, n1);
    }

    // Upload GPU buffers
    triangle_buffer_ = std::make_unique<GPUBuffer<AreaTriangle>>(
        BufferUsage::Storage, std::span<const AreaTriangle>(triangles_), "pd_area_triangles");
    face_csr_buffer_ = std::make_unique<GPUBuffer<FaceCSRMapping>>(
        BufferUsage::Storage, std::span<const FaceCSRMapping>(face_csr_mappings_), "pd_area_csr");

    AreaParams params;
    params.stiffness = stiffness_;
    params.shear_stiffness = 0.0f;
    area_params_buffer_ = std::make_unique<GPUBuffer<AreaParams>>(
        BufferUsage::Uniform, std::span<const AreaParams>(&params, 1), "pd_area_params");

    // Create pipelines
    auto make_pipeline = [](const std::string& path, const std::string& label) -> GPUComputePipeline {
        auto shader = ShaderLoader::CreateModule("ext_pd/" + path, label);
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
    };

    lhs_pipeline_ = make_pipeline("pd_area_lhs.wgsl", "pd_area_lhs");
    project_rhs_pipeline_ = make_pipeline("pd_area_project_rhs.wgsl", "pd_area_project_rhs");

    // Cache bind groups
    uint64 tri_sz = uint64(F) * sizeof(AreaTriangle);
    uint64 csr_map_sz = uint64(F) * sizeof(FaceCSRMapping);
    uint64 diag_sz = uint64(ctx.node_count) * 9 * sizeof(float32);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    uint64 rhs_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 q_sz = uint64(ctx.node_count) * 4 * sizeof(float32);

    auto make_bg = [](const GPUComputePipeline& pipeline, const std::string& label,
                      std::initializer_list<std::pair<uint32, std::pair<WGPUBuffer, uint64>>> entries) {
        auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline.GetHandle(), 0);
        auto builder = BindGroupBuilder(label);
        for (auto& [binding, buf_size] : entries) {
            builder = std::move(builder).AddBuffer(binding, buf_size.first, buf_size.second);
        }
        auto bg = std::move(builder).Build(bgl);
        wgpuBindGroupLayoutRelease(bgl);
        return bg;
    };

    // LHS bind group (unchanged)
    bg_lhs_ = make_bg(lhs_pipeline_, "bg_pd_area_lhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.diag_buffer, diag_sz}},
         {3, {ctx.csr_values_buffer, std::max(csr_val_sz, uint64(4))}},
         {4, {face_csr_buffer_->GetHandle(), csr_map_sz}},
         {5, {area_params_buffer_->GetHandle(), sizeof(AreaParams)}}});

    // Fused project+RHS bind group
    bg_project_rhs_ = make_bg(project_rhs_pipeline_, "bg_pd_area_proj_rhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.q_buffer, q_sz}},
         {3, {ctx.rhs_buffer, rhs_sz}},
         {4, {area_params_buffer_->GetHandle(), sizeof(AreaParams)}}});

    wg_count_ = (F + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("PDAreaTerm: initialized (", F, " faces, nnz=", nnz_, ")");
}

void PDAreaTerm::AssembleLHS(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(lhs_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_lhs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void PDAreaTerm::ProjectRHS(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(project_rhs_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_project_rhs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void PDAreaTerm::Shutdown() {
    bg_lhs_ = {};
    bg_project_rhs_ = {};
    lhs_pipeline_ = {};
    project_rhs_pipeline_ = {};
    triangle_buffer_.reset();
    face_csr_buffer_.reset();
    area_params_buffer_.reset();
    LogInfo("PDAreaTerm: shutdown");
}

}  // namespace ext_pd
