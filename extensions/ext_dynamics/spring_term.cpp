#include "ext_dynamics/spring_term.h"
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

namespace ext_dynamics {

const std::string SpringTerm::kName = "SpringTerm";

SpringTerm::SpringTerm(const std::vector<SpringEdge>& edges, float32 stiffness)
    : edges_(edges), stiffness_(stiffness) {}

const std::string& SpringTerm::GetName() const { return kName; }

void SpringTerm::DeclareSparsity(simulate::SparsityBuilder& builder) {
    for (const auto& edge : edges_) {
        builder.AddEdge(edge.n0, edge.n1);
    }
}

void SpringTerm::Initialize(const simulate::SparsityBuilder& sparsity, const simulate::AssemblyContext& ctx) {
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
        BufferUsage::Storage, std::span<const SpringEdge>(edges_), "spring_edges");
    edge_csr_buffer_ = std::make_unique<GPUBuffer<EdgeCSRMapping>>(
        BufferUsage::Storage, std::span<const EdgeCSRMapping>(edge_csr_mappings_), "spring_edge_csr");

    // Upload spring params uniform
    SpringParams params;
    params.stiffness = stiffness_;
    spring_params_buffer_ = std::make_unique<GPUBuffer<SpringParams>>(
        BufferUsage::Uniform, std::span<const SpringParams>(&params, 1), "spring_params");

    // Create pipeline
    auto shader = ShaderLoader::CreateModule("ext_dynamics/accumulate_springs.wgsl", "accumulate_springs");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "accumulate_springs";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));

    // Cache bind group
    uint64 pos_sz = uint64(ctx.node_count) * 4 * sizeof(float32);
    uint64 force_sz = uint64(ctx.node_count) * 4 * sizeof(uint32);
    uint64 edge_sz = uint64(E) * sizeof(SpringEdge);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    uint64 diag_sz = uint64(ctx.node_count) * 9 * sizeof(float32);
    uint64 csr_map_sz = uint64(E) * sizeof(EdgeCSRMapping);

    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_springs_ = BindGroupBuilder("bg_springs")
        .AddBuffer(0, ctx.params_buffer, ctx.params_size)
        .AddBuffer(1, ctx.position_buffer, pos_sz)
        .AddBuffer(2, ctx.force_buffer, force_sz)
        .AddBuffer(3, edge_buffer_->GetHandle(), edge_sz)
        .AddBuffer(4, ctx.csr_values_buffer, csr_val_sz)
        .AddBuffer(5, ctx.diag_buffer, diag_sz)
        .AddBuffer(6, edge_csr_buffer_->GetHandle(), csr_map_sz)
        .AddBuffer(7, spring_params_buffer_->GetHandle(), sizeof(SpringParams))
        .Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);

    wg_count_ = (E + ctx.workgroup_size - 1) / ctx.workgroup_size;

    LogInfo("SpringTerm: initialized (", E, " edges, nnz=", nnz_, ")");
}

void SpringTerm::Assemble(WGPUCommandEncoder encoder) {
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_springs_.GetHandle());
    enc.Dispatch(wg_count_);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void SpringTerm::Shutdown() {
    bg_springs_ = {};
    pipeline_ = {};
    edge_buffer_.reset();
    edge_csr_buffer_.reset();
    spring_params_buffer_.reset();
    LogInfo("SpringTerm: shutdown");
}

}  // namespace ext_dynamics
