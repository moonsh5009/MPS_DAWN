#include "ext_pd/pd_spring_term.h"
#include "ext_newton/spring_term.h"
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
using ext_newton::SpringParams;

namespace ext_pd {

const std::string PDSpringTerm::kName = "PDSpringTerm";

PDSpringTerm::PDSpringTerm(const std::vector<SpringEdge>& edges, float32 stiffness)
    : edges_(edges), stiffness_(stiffness) {}

const std::string& PDSpringTerm::GetName() const { return kName; }

void PDSpringTerm::DeclareSparsity(SparsityBuilder& builder) {
    for (const auto& edge : edges_) {
        builder.AddEdge(edge.n0, edge.n1);
    }
}

void PDSpringTerm::Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) {
    uint32 E = static_cast<uint32>(edges_.size());
    nnz_ = sparsity.GetNNZ();

    // Build edge-to-CSR mapping
    edge_csr_mappings_.resize(E);
    for (uint32 e = 0; e < E; ++e) {
        uint32 a = edges_[e].n0;
        uint32 b = edges_[e].n1;
        edge_csr_mappings_[e].block_ab = sparsity.GetCSRIndex(a, b);
        edge_csr_mappings_[e].block_ba = sparsity.GetCSRIndex(b, a);
        edge_csr_mappings_[e].block_aa = a;
        edge_csr_mappings_[e].block_bb = b;
    }

    // Upload GPU buffers
    edge_buffer_ = std::make_unique<GPUBuffer<SpringEdge>>(
        BufferUsage::Storage, std::span<const SpringEdge>(edges_), "pd_spring_edges");
    edge_csr_buffer_ = std::make_unique<GPUBuffer<EdgeCSRMapping>>(
        BufferUsage::Storage, std::span<const EdgeCSRMapping>(edge_csr_mappings_), "pd_spring_csr");

    SpringParams params;
    params.stiffness = stiffness_;
    spring_params_buffer_ = std::make_unique<GPUBuffer<SpringParams>>(
        BufferUsage::Uniform, std::span<const SpringParams>(&params, 1), "pd_spring_params");

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

    lhs_pipeline_ = make_pipeline("pd_spring_lhs.wgsl", "pd_spring_lhs");
    project_rhs_pipeline_ = make_pipeline("pd_spring_project_rhs.wgsl", "pd_spring_project_rhs");

    // Cache bind groups
    uint64 edge_sz = uint64(E) * sizeof(SpringEdge);
    uint64 csr_map_sz = uint64(E) * sizeof(EdgeCSRMapping);
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
    bg_lhs_ = make_bg(lhs_pipeline_, "bg_pd_spring_lhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {edge_buffer_->GetHandle(), edge_sz}},
         {2, {ctx.diag_buffer, diag_sz}},
         {3, {ctx.csr_values_buffer, std::max(csr_val_sz, uint64(4))}},
         {4, {edge_csr_buffer_->GetHandle(), csr_map_sz}},
         {5, {spring_params_buffer_->GetHandle(), sizeof(SpringParams)}}});

    // Fused project+RHS bind group
    bg_project_rhs_ = make_bg(project_rhs_pipeline_, "bg_pd_spring_proj_rhs",
        {{0, {ctx.params_buffer, ctx.params_size}},
         {1, {edge_buffer_->GetHandle(), edge_sz}},
         {2, {ctx.q_buffer, q_sz}},
         {3, {ctx.rhs_buffer, rhs_sz}},
         {4, {spring_params_buffer_->GetHandle(), sizeof(SpringParams)}}});

    wg_count_ = (E + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("PDSpringTerm: initialized (", E, " edges, nnz=", nnz_, ")");
}

void PDSpringTerm::AssembleLHS(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(lhs_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_lhs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void PDSpringTerm::ProjectRHS(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(project_rhs_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_project_rhs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void PDSpringTerm::Shutdown() {
    bg_lhs_ = {};
    bg_project_rhs_ = {};
    lhs_pipeline_ = {};
    project_rhs_pipeline_ = {};
    edge_buffer_.reset();
    edge_csr_buffer_.reset();
    spring_params_buffer_.reset();
    LogInfo("PDSpringTerm: shutdown");
}

}  // namespace ext_pd
