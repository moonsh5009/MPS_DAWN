#include "ext_avbd/avbd_area_term.h"
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

namespace ext_avbd {

const std::string AVBDAreaTerm::kName = "AVBDAreaTerm";

void AVBDAreaTerm::SetAreaData(const std::vector<AreaTriangle>& triangles,
                                uint32 node_count,
                                float32 stretch_k, float32 shear_mu) {
    triangles_ = triangles;
    stretch_stiffness_ = stretch_k;
    shear_stiffness_ = shear_mu;

    uint32 F = static_cast<uint32>(triangles.size());

    // Build face CSR adjacency: for each vertex, list its incident faces + role
    face_offsets_.assign(node_count + 1, 0);

    // Count per-vertex incident faces
    for (const auto& tri : triangles) {
        face_offsets_[tri.n0 + 1]++;
        face_offsets_[tri.n1 + 1]++;
        face_offsets_[tri.n2 + 1]++;
    }

    // Prefix sum
    for (uint32 i = 0; i < node_count; ++i) {
        face_offsets_[i + 1] += face_offsets_[i];
    }

    // Fill adjacency entries
    face_adjacency_.resize(face_offsets_[node_count]);
    std::vector<uint32> cursor(node_count, 0);
    for (uint32 f = 0; f < F; ++f) {
        const auto& tri = triangles[f];
        uint32 s0 = face_offsets_[tri.n0] + cursor[tri.n0]++;
        face_adjacency_[s0] = {f, 0};
        uint32 s1 = face_offsets_[tri.n1] + cursor[tri.n1]++;
        face_adjacency_[s1] = {f, 1};
        uint32 s2 = face_offsets_[tri.n2] + cursor[tri.n2]++;
        face_adjacency_[s2] = {f, 2};
    }

    LogInfo("AVBDAreaTerm: built face CSR (", F, " faces, ",
            face_adjacency_.size(), " adjacency entries)");
}

const std::string& AVBDAreaTerm::GetName() const { return kName; }

void AVBDAreaTerm::Initialize(const AVBDTermContext& ctx) {
    uint32 F = static_cast<uint32>(triangles_.size());

    // Upload buffers
    triangle_buf_ = std::make_unique<GPUBuffer<AreaTriangle>>(
        BufferUsage::Storage,
        std::span<const AreaTriangle>(triangles_),
        "avbd_area_triangles");

    face_offsets_buf_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage,
        std::span<const uint32>(face_offsets_),
        "avbd_face_offsets");

    if (!face_adjacency_.empty()) {
        face_adjacency_buf_ = std::make_unique<GPUBuffer<FaceAdjacency>>(
            BufferUsage::Storage,
            std::span<const FaceAdjacency>(face_adjacency_),
            "avbd_face_adjacency");
    } else {
        FaceAdjacency dummy{};
        face_adjacency_buf_ = std::make_unique<GPUBuffer<FaceAdjacency>>(
            BufferUsage::Storage,
            std::span<const FaceAdjacency>(&dummy, 1),
            "avbd_face_adjacency");
    }

    AVBDAreaParams ap{};
    ap.stiffness = stretch_stiffness_;
    ap.shear_stiffness = shear_stiffness_;
    area_params_buf_ = std::make_unique<GPUBuffer<AVBDAreaParams>>(
        BufferUsage::Uniform,
        std::span<const AVBDAreaParams>(&ap, 1),
        "avbd_area_params");

    uint64 tri_sz = static_cast<uint64>(F) * sizeof(AreaTriangle);
    uint64 offsets_sz = face_offsets_.size() * sizeof(uint32);
    uint64 adj_sz = !face_adjacency_.empty()
        ? face_adjacency_.size() * sizeof(FaceAdjacency)
        : sizeof(FaceAdjacency);

    // Create pipeline
    auto shader = ShaderLoader::CreateModule("ext_avbd/avbd_accum_area.wgsl", "avbd_accum_area");
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    std::string label = "avbd_accum_area";
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    pipeline_ = GPUComputePipeline(wgpuDeviceCreateComputePipeline(
        GPUCore::GetInstance().GetDevice(), &desc));

    // Per-color bind groups
    // Bindings: 0=VBDColorParams, 1=AreaParams, 2=q, 3=gradient(rw), 4=hessian(rw),
    //           5=vertex_order, 6=face_offsets, 7=face_adjacency, 8=triangles
    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline_.GetHandle(), 0);
    bg_per_color_.clear();
    color_vertex_counts_.clear();

    for (uint32 i = 0; i < static_cast<uint32>(ctx.color_groups.size()); ++i) {
        const auto& cg = ctx.color_groups[i];
        color_vertex_counts_.push_back(cg.color_vertex_count);

        auto bg = BindGroupBuilder("bg_avbd_area_" + std::to_string(i))
            .AddBuffer(0, cg.params_buf, cg.params_sz)
            .AddBuffer(1, area_params_buf_->GetHandle(), sizeof(AVBDAreaParams))
            .AddBuffer(2, ctx.q_buf, ctx.q_sz)
            .AddBuffer(3, ctx.gradient_buf, ctx.gradient_sz)
            .AddBuffer(4, ctx.hessian_buf, ctx.hessian_sz)
            .AddBuffer(5, ctx.vertex_order_buf, ctx.vertex_order_sz)
            .AddBuffer(6, face_offsets_buf_->GetHandle(), offsets_sz)
            .AddBuffer(7, face_adjacency_buf_->GetHandle(), adj_sz)
            .AddBuffer(8, triangle_buf_->GetHandle(), tri_sz)
            .Build(bgl);
        bg_per_color_.push_back(std::move(bg));
    }
    wgpuBindGroupLayoutRelease(bgl);

    LogInfo("AVBDAreaTerm: initialized (", F, " triangles, stretch=",
            stretch_stiffness_, ", shear=", shear_stiffness_, ")");
}

void AVBDAreaTerm::AccumulateColor(WGPUCommandEncoder encoder, uint32 color_index) {
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

void AVBDAreaTerm::Shutdown() {
    bg_per_color_.clear();
    color_vertex_counts_.clear();
    pipeline_ = {};
    triangle_buf_.reset();
    face_offsets_buf_.reset();
    face_adjacency_buf_.reset();
    area_params_buf_.reset();
    LogInfo("AVBDAreaTerm: shutdown");
}

}  // namespace ext_avbd
