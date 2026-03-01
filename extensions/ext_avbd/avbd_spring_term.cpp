#include "ext_avbd/avbd_spring_term.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;

namespace ext_avbd {

const std::string AVBDSpringTerm::kName = "AVBDSpringTerm";

void AVBDSpringTerm::SetSpringData(std::span<const uint32> offsets,
                                    std::span<const SpringNeighbor> neighbors,
                                    std::span<const ext_dynamics::SpringEdge> edges,
                                    float32 stiffness,
                                    float32 al_gamma,
                                    float32 al_beta) {
    offsets_.assign(offsets.begin(), offsets.end());
    neighbors_.assign(neighbors.begin(), neighbors.end());
    edges_.assign(edges.begin(), edges.end());
    stiffness_ = stiffness;
    al_gamma_ = al_gamma;
    al_beta_ = al_beta;
    edge_count_ = static_cast<uint32>(edges.size());
}

const std::string& AVBDSpringTerm::GetName() const { return kName; }

void AVBDSpringTerm::Initialize(const AVBDTermContext& ctx) {
    // Upload CSR adjacency buffers
    offsets_buf_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage,
        std::span<const uint32>(offsets_),
        "avbd_spring_offsets");

    if (!neighbors_.empty()) {
        neighbors_buf_ = std::make_unique<GPUBuffer<SpringNeighbor>>(
            BufferUsage::Storage,
            std::span<const SpringNeighbor>(neighbors_),
            "avbd_spring_neighbors");
    } else {
        SpringNeighbor dummy{};
        neighbors_buf_ = std::make_unique<GPUBuffer<SpringNeighbor>>(
            BufferUsage::Storage,
            std::span<const SpringNeighbor>(&dummy, 1),
            "avbd_spring_neighbors");
    }

    uint64 offsets_sz = offsets_.size() * sizeof(uint32);
    uint64 neighbors_sz = !neighbors_.empty()
        ? neighbors_.size() * sizeof(SpringNeighbor)
        : sizeof(SpringNeighbor);

    // ---------------------------------------------------------------
    // Augmented Lagrangian buffers
    // ---------------------------------------------------------------

    // Penalty buffer (zero-initialized, E × f32, persistent across frames)
    // Warmstart shader will clamp to PENALTY_MIN (1.0) at frame start.
    penalty_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage,
        .size = static_cast<uint64>(edge_count_) * sizeof(float32),
        .label = "avbd_al_penalty"
    });

    // Edge buffer (E × SpringEdge)
    edge_buf_ = std::make_unique<GPUBuffer<ext_dynamics::SpringEdge>>(
        BufferUsage::Storage,
        std::span<const ext_dynamics::SpringEdge>(edges_),
        "avbd_al_edges");

    // AL params uniform
    ALParams al{};
    al.stiffness = stiffness_;
    al.gamma = al_gamma_;
    al.beta = al_beta_;
    al.edge_count = edge_count_;
    al_params_buf_ = std::make_unique<GPUBuffer<ALParams>>(
        BufferUsage::Uniform,
        std::span<const ALParams>(&al, 1),
        "avbd_al_params");

    uint64 penalty_sz = static_cast<uint64>(edge_count_) * sizeof(float32);
    uint64 edge_buf_sz = static_cast<uint64>(edge_count_) * sizeof(ext_dynamics::SpringEdge);
    uint64 al_params_sz = sizeof(ALParams);

    // ---------------------------------------------------------------
    // AL pipelines
    // ---------------------------------------------------------------
    auto make_pipeline = [](const std::string& shader_path, const std::string& label) -> GPUComputePipeline {
        auto shader = ShaderLoader::CreateModule("ext_avbd/" + shader_path, label);
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
    };

    warmstart_pipeline_ = make_pipeline("avbd_warmstart_penalty.wgsl", "avbd_warmstart_penalty");
    penalty_ramp_pipeline_ = make_pipeline("avbd_penalty_ramp.wgsl", "avbd_penalty_ramp");

    // Warmstart bind group: {0: al_params, 1: penalty}
    {
        auto bgl = wgpuComputePipelineGetBindGroupLayout(warmstart_pipeline_.GetHandle(), 0);
        bg_warmstart_ = BindGroupBuilder("bg_avbd_warmstart")
            .AddBuffer(0, al_params_buf_->GetHandle(), al_params_sz)
            .AddBuffer(1, penalty_buf_->GetHandle(), penalty_sz)
            .Build(bgl);
        wgpuBindGroupLayoutRelease(bgl);
    }

    // Penalty ramp bind group: {0: al_params, 1: q, 2: edges, 3: penalty}
    {
        auto bgl = wgpuComputePipelineGetBindGroupLayout(penalty_ramp_pipeline_.GetHandle(), 0);
        bg_penalty_ramp_ = BindGroupBuilder("bg_avbd_penalty_ramp")
            .AddBuffer(0, al_params_buf_->GetHandle(), al_params_sz)
            .AddBuffer(1, ctx.q_buf, ctx.q_sz)
            .AddBuffer(2, edge_buf_->GetHandle(), edge_buf_sz)
            .AddBuffer(3, penalty_buf_->GetHandle(), penalty_sz)
            .Build(bgl);
        wgpuBindGroupLayoutRelease(bgl);
    }

    // ---------------------------------------------------------------
    // Accumulation pipeline
    // ---------------------------------------------------------------
    auto shader = ShaderLoader::CreateModule("ext_avbd/avbd_accum_spring.wgsl", "avbd_accum_spring");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "avbd_accum_spring";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(
        GPUCore::GetInstance().GetDevice(), &desc));

    // Per-color bind groups
    // Bindings: 0=VBDColorParams, 1=q, 2=gradient(rw), 3=hessian(rw),
    //           4=vertex_order, 5=spring_offsets, 6=spring_neighbors, 7=penalty(read)
    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_per_color_.clear();
    color_vertex_counts_.clear();

    for (uint32 i = 0; i < static_cast<uint32>(ctx.color_groups.size()); ++i) {
        const auto& cg = ctx.color_groups[i];
        color_vertex_counts_.push_back(cg.color_vertex_count);

        auto bg = BindGroupBuilder("bg_avbd_spring_" + std::to_string(i))
            .AddBuffer(0, cg.params_buf, cg.params_sz)
            .AddBuffer(1, ctx.q_buf, ctx.q_sz)
            .AddBuffer(2, ctx.gradient_buf, ctx.gradient_sz)
            .AddBuffer(3, ctx.hessian_buf, ctx.hessian_sz)
            .AddBuffer(4, ctx.vertex_order_buf, ctx.vertex_order_sz)
            .AddBuffer(5, offsets_buf_->GetHandle(), offsets_sz)
            .AddBuffer(6, neighbors_buf_->GetHandle(), neighbors_sz)
            .AddBuffer(7, penalty_buf_->GetHandle(), penalty_sz)
            .Build(bgl);
        bg_per_color_.push_back(std::move(bg));
    }
    wgpuBindGroupLayoutRelease(bgl);

    LogInfo("AVBDSpringTerm: initialized (", neighbors_.size(),
            " CSR entries, k=", stiffness_, ", γ=", al_gamma_,
            ", β=", al_beta_, ", ", edge_count_, " edges)");
}

void AVBDSpringTerm::AccumulateColor(WGPUCommandEncoder encoder, uint32 color_index) {
    if (color_index >= bg_per_color_.size()) return;

    uint32 count = color_vertex_counts_[color_index];
    uint32 wg = (count + kWorkgroupSize - 1) / kWorkgroupSize;
    if (wg == 0) return;

    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_per_color_[color_index].GetHandle());
    enc.Dispatch(wg);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void AVBDSpringTerm::WarmstartDecay(WGPUCommandEncoder encoder) {
    if (edge_count_ == 0) return;

    uint32 wg = (edge_count_ + kWorkgroupSize - 1) / kWorkgroupSize;
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(warmstart_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_warmstart_.GetHandle());
    enc.Dispatch(wg);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void AVBDSpringTerm::DualUpdate(WGPUCommandEncoder encoder) {
    if (edge_count_ == 0) return;

    uint32 wg = (edge_count_ + kWorkgroupSize - 1) / kWorkgroupSize;
    WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
    ComputeEncoder enc(pass);
    enc.SetPipeline(penalty_ramp_pipeline_.GetHandle());
    enc.SetBindGroup(0, bg_penalty_ramp_.GetHandle());
    enc.Dispatch(wg);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void AVBDSpringTerm::Shutdown() {
    bg_per_color_.clear();
    color_vertex_counts_.clear();
    pipeline_ = {};

    offsets_buf_.reset();
    neighbors_buf_.reset();

    // AL cleanup
    bg_warmstart_ = {};
    bg_penalty_ramp_ = {};
    warmstart_pipeline_ = {};
    penalty_ramp_pipeline_ = {};
    penalty_buf_.reset();
    edge_buf_.reset();
    al_params_buf_.reset();

    LogInfo("AVBDSpringTerm: shutdown");
}

}  // namespace ext_avbd
