#include "ext_pd_term/pd_area_term.h"
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

namespace ext_pd_term {

const std::string PDAreaTerm::kName = "PDAreaTerm";

PDAreaTerm::PDAreaTerm(const std::vector<AreaTriangle>& triangles, float32 stretch_stiffness, float32 shear_stiffness)
    : triangles_(triangles), stretch_stiffness_(stretch_stiffness), shear_stiffness_(shear_stiffness) {}

const std::string& PDAreaTerm::GetName() const { return kName; }

void PDAreaTerm::DeclareSparsity(SparsityBuilder& builder) {
    for (const auto& tri : triangles_) {
        builder.AddEdge(tri.n0, tri.n1);
        builder.AddEdge(tri.n0, tri.n2);
        builder.AddEdge(tri.n1, tri.n2);
    }
}

static GPUComputePipeline MakePipeline(const std::string& shader_dir,
                                        const std::string& shader_file,
                                        const std::string& label) {
    auto shader = ShaderLoader::CreateModule(shader_dir + shader_file, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

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

void PDAreaTerm::Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) {
    uint32 F = static_cast<uint32>(triangles_.size());
    nnz_ = sparsity.GetNNZ();

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

    triangle_buffer_ = std::make_unique<GPUBuffer<AreaTriangle>>(
        BufferUsage::Storage, std::span<const AreaTriangle>(triangles_), "pd_area_triangles");
    face_csr_buffer_ = std::make_unique<GPUBuffer<FaceCSRMapping>>(
        BufferUsage::Storage, std::span<const FaceCSRMapping>(face_csr_mappings_), "pd_area_csr");

    AreaParams params;
    params.stiffness = stretch_stiffness_;
    params.shear_stiffness = shear_stiffness_;
    area_params_buffer_ = std::make_unique<GPUBuffer<AreaParams>>(
        BufferUsage::Uniform, std::span<const AreaParams>(&params, 1), "pd_area_params");

    lhs_pipeline_ = MakePipeline("ext_pd_term/", "pd_area_lhs.wgsl", "pd_area_lhs");
    project_rhs_pipeline_ = MakePipeline("ext_pd_term/", "pd_area_project_rhs.wgsl", "pd_area_project_rhs");

    uint64 tri_sz = uint64(F) * sizeof(AreaTriangle);
    uint64 csr_map_sz = uint64(F) * sizeof(FaceCSRMapping);
    uint64 diag_sz = uint64(ctx.node_count) * 9 * sizeof(float32);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    uint64 rhs_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 q_sz = uint64(ctx.node_count) * 4 * sizeof(float32);

    bg_lhs_ = MakeBG(lhs_pipeline_, "bg_pd_area_lhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.diag_buffer, diag_sz}},
         {3, {ctx.csr_values_buffer, std::max(csr_val_sz, uint64(4))}},
         {4, {face_csr_buffer_->GetHandle(), csr_map_sz}},
         {5, {area_params_buffer_->GetHandle(), sizeof(AreaParams)}}});

    bg_project_rhs_ = MakeBG(project_rhs_pipeline_, "bg_pd_area_proj_rhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.q_buffer, q_sz}},
         {3, {ctx.rhs_buffer, rhs_sz}},
         {4, {area_params_buffer_->GetHandle(), sizeof(AreaParams)}}});

    wg_count_ = (F + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("PDAreaTerm: initialized (", F, " faces, nnz=", nnz_, ")");
}

void PDAreaTerm::AssembleLHS(WGPUCommandEncoder encoder) {
    Dispatch(encoder, lhs_pipeline_, bg_lhs_, wg_count_);
}

void PDAreaTerm::ProjectRHS(WGPUCommandEncoder encoder) {
    Dispatch(encoder, project_rhs_pipeline_, bg_project_rhs_, wg_count_);
}

void PDAreaTerm::InitializeADMM(const PDAssemblyContext& ctx) {
    uint32 F = static_cast<uint32>(triangles_.size());
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    // 2 vec4f per face (for 3x2 rotation columns)
    uint64 zu_sz = uint64(F) * 2 * 4 * sizeof(float32);

    z_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = zu_sz, .label = "admm_area_z"});
    u_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = zu_sz, .label = "admm_area_u"});

    admm_project_pipeline_ = MakePipeline("ext_pd_term/", "admm_area_project.wgsl", "admm_area_project");
    admm_rhs_pipeline_ = MakePipeline("ext_pd_term/", "admm_area_rhs.wgsl", "admm_area_rhs");
    admm_dual_pipeline_ = MakePipeline("ext_pd_term/", "admm_area_dual.wgsl", "admm_area_dual");
    admm_reset_pipeline_ = MakePipeline("ext_pd_term/", "admm_area_reset.wgsl", "admm_area_reset");

    uint64 tri_sz = uint64(F) * sizeof(AreaTriangle);
    uint64 q_sz = uint64(ctx.node_count) * 4 * sizeof(float32);
    uint64 rhs_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 s_sz = uint64(ctx.node_count) * 4 * sizeof(float32);

    bg_admm_project_ = MakeBG(admm_project_pipeline_, "bg_admm_area_project",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.q_buffer, q_sz}},
         {3, {z_buffer_->GetHandle(), zu_sz}},
         {4, {u_buffer_->GetHandle(), zu_sz}}});

    bg_admm_rhs_ = MakeBG(admm_rhs_pipeline_, "bg_admm_area_rhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {z_buffer_->GetHandle(), zu_sz}},
         {3, {u_buffer_->GetHandle(), zu_sz}},
         {4, {ctx.rhs_buffer, rhs_sz}},
         {5, {area_params_buffer_->GetHandle(), sizeof(AreaParams)}}});

    bg_admm_dual_ = MakeBG(admm_dual_pipeline_, "bg_admm_area_dual",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.q_buffer, q_sz}},
         {3, {z_buffer_->GetHandle(), zu_sz}},
         {4, {u_buffer_->GetHandle(), zu_sz}}});

    bg_admm_reset_ = MakeBG(admm_reset_pipeline_, "bg_admm_area_reset",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {triangle_buffer_->GetHandle(), tri_sz}},
         {2, {ctx.s_buffer, s_sz}},
         {3, {z_buffer_->GetHandle(), zu_sz}},
         {4, {u_buffer_->GetHandle(), zu_sz}}});

    LogInfo("PDAreaTerm: ADMM initialized (", F, " faces)");
}

void PDAreaTerm::ProjectLocal(WGPUCommandEncoder encoder) {
    Dispatch(encoder, admm_project_pipeline_, bg_admm_project_, wg_count_);
}

void PDAreaTerm::AssembleADMMRHS(WGPUCommandEncoder encoder) {
    Dispatch(encoder, admm_rhs_pipeline_, bg_admm_rhs_, wg_count_);
}

void PDAreaTerm::UpdateDual(WGPUCommandEncoder encoder) {
    Dispatch(encoder, admm_dual_pipeline_, bg_admm_dual_, wg_count_);
}

void PDAreaTerm::ResetDual(WGPUCommandEncoder encoder) {
    Dispatch(encoder, admm_reset_pipeline_, bg_admm_reset_, wg_count_);
}

void PDAreaTerm::Shutdown() {
    bg_lhs_ = {};
    bg_project_rhs_ = {};
    bg_admm_project_ = {};
    bg_admm_rhs_ = {};
    bg_admm_dual_ = {};
    bg_admm_reset_ = {};

    lhs_pipeline_ = {};
    project_rhs_pipeline_ = {};
    admm_project_pipeline_ = {};
    admm_rhs_pipeline_ = {};
    admm_dual_pipeline_ = {};
    admm_reset_pipeline_ = {};

    triangle_buffer_.reset();
    face_csr_buffer_.reset();
    area_params_buffer_.reset();
    z_buffer_.reset();
    u_buffer_.reset();
    LogInfo("PDAreaTerm: shutdown");
}

}  // namespace ext_pd_term
