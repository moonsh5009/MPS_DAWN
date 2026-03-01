#include "ext_avbd/vbd_dynamics.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <algorithm>
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;

namespace ext_avbd {

// ============================================================================
// Helpers
// ============================================================================

static GPUBindGroup MakeBindGroup(const GPUComputePipeline& pipeline,
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

static GPUComputePipeline MakePipeline(const std::string& shader_path, const std::string& label) {
    auto shader = ShaderLoader::CreateModule("ext_avbd/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// Initialize
// ============================================================================

void VBDDynamics::Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                              uint32 iterations,
                              WGPUBuffer physics_buf, uint64 physics_sz,
                              WGPUBuffer pos_buf, WGPUBuffer vel_buf, WGPUBuffer mass_buf,
                              uint64 pos_sz, uint64 vel_sz, uint64 mass_sz,
                              const std::vector<uint32>& color_offsets,
                              WGPUBuffer vertex_order_buf, uint64 vertex_order_sz) {
    node_count_ = node_count;
    edge_count_ = edge_count;
    face_count_ = face_count;
    iterations_ = iterations;
    color_offsets_ = color_offsets;
    color_count_ = static_cast<uint32>(color_offsets.size()) - 1;
    vertex_order_buf_ = vertex_order_buf;
    vertex_order_sz_ = vertex_order_sz;

    // Create solver params buffer
    SolverParams params{};
    params.node_count = node_count;
    params.edge_count = edge_count;
    params.face_count = face_count;
    solver_params_buf_ = std::make_unique<GPUBuffer<SolverParams>>(
        BufferUsage::Uniform | BufferUsage::CopyDst,
        std::span<const SolverParams>(&params, 1),
        "avbd_solver_params");
    uint64 solver_params_sz = sizeof(SolverParams);

    // Create working buffers
    uint64 vec4_size = static_cast<uint64>(node_count) * 16;  // N x vec4f
    q_sz_ = vec4_size;
    gradient_sz_ = vec4_size;                                   // N x vec4f
    hessian_sz_ = static_cast<uint64>(node_count) * 48;        // N x 3 x vec4f

    x_old_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst,
        .size = vec4_size,
        .label = "avbd_x_old"
    });

    s_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst,
        .size = vec4_size,
        .label = "avbd_s"
    });

    q_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage | BufferUsage::CopyDst,
        .size = vec4_size,
        .label = "avbd_q"
    });

    gradient_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage,
        .size = gradient_sz_,
        .label = "avbd_gradient"
    });

    hessian_buf_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{
        .usage = BufferUsage::Storage,
        .size = hessian_sz_,
        .label = "avbd_hessian"
    });

    // Create pipelines
    init_pipeline_ = MakePipeline("avbd_init.wgsl", "avbd_init");
    predict_pipeline_ = MakePipeline("avbd_predict.wgsl", "avbd_predict");
    copy_q_pipeline_ = MakePipeline("avbd_copy_q.wgsl", "avbd_copy_q");
    local_solve_pipeline_ = MakePipeline("avbd_local_solve.wgsl", "avbd_local_solve");

    // Pre-solve bind groups (unchanged layout)
    bg_init_ = MakeBindGroup(init_pipeline_, "bg_avbd_init",
        {{0, {solver_params_buf_->GetHandle(), solver_params_sz}},
         {1, {pos_buf, pos_sz}},
         {2, {x_old_buf_->GetHandle(), vec4_size}}});

    bg_predict_ = MakeBindGroup(predict_pipeline_, "bg_avbd_predict",
        {{0, {physics_buf, physics_sz}},
         {1, {solver_params_buf_->GetHandle(), solver_params_sz}},
         {2, {x_old_buf_->GetHandle(), vec4_size}},
         {3, {vel_buf, vel_sz}},
         {4, {mass_buf, mass_sz}},
         {5, {s_buf_->GetHandle(), vec4_size}}});

    bg_copy_ = MakeBindGroup(copy_q_pipeline_, "bg_avbd_copy",
        {{0, {solver_params_buf_->GetHandle(), solver_params_sz}},
         {1, {s_buf_->GetHandle(), vec4_size}},
         {2, {q_buf_->GetHandle(), vec4_size}}});

    // Per-color uniform buffers + bind groups for local_solve (inertia fused)
    bg_local_solve_.clear();
    color_params_bufs_.clear();
    color_groups_.clear();

    for (uint32 c = 0; c < color_count_; ++c) {
        VBDColorParams cp{};
        cp.color_offset = color_offsets_[c];
        cp.color_vertex_count = color_offsets_[c + 1] - color_offsets_[c];

        auto cpb = std::make_unique<GPUBuffer<VBDColorParams>>(
            BufferUsage::Uniform,
            std::span<const VBDColorParams>(&cp, 1),
            "avbd_color_params_" + std::to_string(c));
        uint64 cp_sz = sizeof(VBDColorParams);

        // Local solve bind group (with inertia fused in)
        // 0=physics, 1=color_params, 2=q(rw), 3=s, 4=mass, 5=gradient(rw), 6=hessian(rw), 7=vertex_order
        auto bg_ls = MakeBindGroup(local_solve_pipeline_,
            "bg_avbd_local_solve_" + std::to_string(c),
            {{0, {physics_buf, physics_sz}},
             {1, {cpb->GetHandle(), cp_sz}},
             {2, {q_buf_->GetHandle(), vec4_size}},
             {3, {s_buf_->GetHandle(), vec4_size}},
             {4, {mass_buf, mass_sz}},
             {5, {gradient_buf_->GetHandle(), gradient_sz_}},
             {6, {hessian_buf_->GetHandle(), hessian_sz_}},
             {7, {vertex_order_buf, vertex_order_sz}}});
        bg_local_solve_.push_back(std::move(bg_ls));

        color_groups_.push_back({cp.color_offset, cp.color_vertex_count,
                                  cpb->GetHandle(), cp_sz});

        color_params_bufs_.push_back(std::move(cpb));
    }

    LogInfo("VBDDynamics: initialized (", node_count, " nodes, ", color_count_,
            " colors, ", iterations_, " iterations)");
}

// ============================================================================
// AddTerm
// ============================================================================

void VBDDynamics::AddTerm(std::unique_ptr<IAVBDTerm> term) {
    AVBDTermContext ctx{
        .node_count = node_count_,
        .edge_count = edge_count_,
        .face_count = face_count_,
        .q_buf = q_buf_->GetHandle(),
        .q_sz = q_sz_,
        .gradient_buf = gradient_buf_->GetHandle(),
        .gradient_sz = gradient_sz_,
        .hessian_buf = hessian_buf_->GetHandle(),
        .hessian_sz = hessian_sz_,
        .vertex_order_buf = vertex_order_buf_,
        .vertex_order_sz = vertex_order_sz_,
        .color_groups = color_groups_,
    };

    term->Initialize(ctx);
    LogInfo("VBDDynamics: added term '", term->GetName(), "'");
    terms_.push_back(std::move(term));
}

// ============================================================================
// Solve
// ============================================================================

void VBDDynamics::Solve(WGPUCommandEncoder encoder) {
    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;

    // Augmented Lagrangian: warmstart decay (λ *= γ) at frame start
    for (auto& term : terms_) {
        term->WarmstartDecay(encoder);
    }

    // Phase 1: init (x_old = pos)
    {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(init_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_init_.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Phase 2: predict (s = x_old + dt*v + dt²*gravity)
    {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(predict_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_predict_.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Phase 3: copy (q = s)
    {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(copy_q_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_copy_.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Phase 4: VBD iteration loop
    for (uint32 iter = 0; iter < iterations_; ++iter) {
        // GS sweep: for each color: terms → local_solve (inertia fused)
        for (uint32 c = 0; c < color_count_; ++c) {
            uint32 color_vertex_count = color_offsets_[c + 1] - color_offsets_[c];
            uint32 color_wg = (color_vertex_count + kWorkgroupSize - 1) / kWorkgroupSize;
            if (color_wg == 0) continue;

            // Pass 1..N: Term accumulation (ADD to zero'd gradient/hessian)
            for (auto& term : terms_) {
                term->AccumulateColor(encoder, c);
            }

            // Pass N+1: Local solve (add inertia + solve + zero gradient/hessian)
            {
                WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
                WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
                ComputeEncoder enc(pass);
                enc.SetPipeline(local_solve_pipeline_.GetHandle());
                enc.SetBindGroup(0, bg_local_solve_[c].GetHandle());
                enc.Dispatch(color_wg);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
            }
        }

        // Augmented Lagrangian: dual update (λ = k*C + λ) after each iteration
        for (auto& term : terms_) {
            term->DualUpdate(encoder);
        }
    }
}

// ============================================================================
// Accessors
// ============================================================================

WGPUBuffer VBDDynamics::GetQBuffer() const {
    return q_buf_ ? q_buf_->GetHandle() : nullptr;
}

WGPUBuffer VBDDynamics::GetXOldBuffer() const {
    return x_old_buf_ ? x_old_buf_->GetHandle() : nullptr;
}

WGPUBuffer VBDDynamics::GetSolverParamsBuffer() const {
    return solver_params_buf_ ? solver_params_buf_->GetHandle() : nullptr;
}

uint64 VBDDynamics::GetSolverParamsSize() const {
    return sizeof(SolverParams);
}

// ============================================================================
// Shutdown
// ============================================================================

void VBDDynamics::Shutdown() {
    for (auto& term : terms_) {
        term->Shutdown();
    }
    terms_.clear();

    bg_init_ = {};
    bg_predict_ = {};
    bg_copy_ = {};
    bg_local_solve_.clear();
    color_params_bufs_.clear();
    color_groups_.clear();

    init_pipeline_ = {};
    predict_pipeline_ = {};
    copy_q_pipeline_ = {};
    local_solve_pipeline_ = {};

    x_old_buf_.reset();
    s_buf_.reset();
    q_buf_.reset();
    gradient_buf_.reset();
    hessian_buf_.reset();
    solver_params_buf_.reset();

    color_offsets_.clear();
    color_count_ = 0;
    node_count_ = 0;
    edge_count_ = 0;
    face_count_ = 0;
    iterations_ = 10;
    vertex_order_buf_ = nullptr;
    vertex_order_sz_ = 0;
    q_sz_ = 0;
    gradient_sz_ = 0;
    hessian_sz_ = 0;

    LogInfo("VBDDynamics: shutdown");
}

}  // namespace ext_avbd
