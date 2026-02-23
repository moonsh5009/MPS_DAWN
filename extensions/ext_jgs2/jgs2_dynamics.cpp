#include "ext_jgs2/jgs2_dynamics.h"
#include "core_simulate/sim_components.h"
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

namespace ext_jgs2 {

// ============================================================================
// Static helpers
// ============================================================================

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

static GPUComputePipeline MakePipeline(const std::string& shader_path,
                                        const std::string& label) {
    auto shader = ShaderLoader::CreateModule(shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// JGS2Dynamics
// ============================================================================

JGS2Dynamics::JGS2Dynamics() = default;

JGS2Dynamics::~JGS2Dynamics() = default;

void JGS2Dynamics::AddTerm(std::unique_ptr<IJGS2Term> term) {
    terms_.push_back(std::move(term));
}

void JGS2Dynamics::Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                               WGPUBuffer physics_buffer, uint64 physics_size,
                               WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                               WGPUBuffer mass_buffer, uint32 workgroup_size) {
    node_count_ = node_count;
    edge_count_ = edge_count;
    face_count_ = face_count;
    workgroup_size_ = workgroup_size;
    node_wg_count_ = (node_count + workgroup_size - 1) / workgroup_size;
    physics_buffer_ = physics_buffer;
    physics_size_ = physics_size;

    CreateBuffers(mass_buffer);
    CreatePipelines();

    // Build JGS2AssemblyContext for term bind group caching
    JGS2AssemblyContext ctx{};
    ctx.params_buffer = params_buffer_->GetHandle();
    ctx.q_buffer = q_buffer_->GetHandle();
    ctx.gradient_buffer = gradient_buffer_->GetHandle();
    ctx.hessian_diag_buffer = hessian_diag_buffer_->GetHandle();
    ctx.node_count = node_count;
    ctx.edge_count = edge_count;
    ctx.workgroup_size = workgroup_size;
    ctx.params_size = sizeof(SolverParams);

    // Initialize terms with context for bind group caching
    for (auto& term : terms_) {
        term->Initialize(ctx);
    }

    // Cache dynamics bind groups
    CacheBindGroups(position_buffer, velocity_buffer, mass_buffer);

    LogInfo("JGS2Dynamics: initialized (", node_count_, " nodes, ",
            edge_count_, " edges, ", terms_.size(), " terms, ",
            iterations_, " iterations, correction=", enable_correction_ ? "on" : "off", ")");
}

void JGS2Dynamics::CreateBuffers(WGPUBuffer mass_buffer) {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);

    // Solver params uniform
    params_.node_count = node_count_;
    params_.edge_count = edge_count_;
    params_.face_count = face_count_;
    params_buffer_ = std::make_unique<GPUBuffer<SolverParams>>(
        BufferUsage::Uniform, std::span<const SolverParams>(&params_, 1), "jgs2_solver_params");

    // Solver buffers
    x_old_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "jgs2_x_old"});
    s_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "jgs2_s"});
    q_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "jgs2_q"});

    // Gradient buffer (N×4 atomic u32)
    gradient_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(node_count_) * 4 * sizeof(uint32),
                     .label = "jgs2_gradient"});

    // Hessian diagonal buffer (N×9 float32 as atomic u32)
    hessian_diag_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(node_count_) * 9 * sizeof(float32),
                     .label = "jgs2_hessian_diag"});

    // Correction buffer (N×9 float32, initialized to zeros)
    uint64 corr_sz = uint64(node_count_) * 9 * sizeof(float32);
    correction_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = corr_sz, .label = "jgs2_correction"});

    // Zero the correction buffer (Phase 1: no correction)
    auto& gpu = GPUCore::GetInstance();
    WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &ed);
    wgpuCommandEncoderClearBuffer(enc, correction_buffer_->GetHandle(), 0, corr_sz);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
}

void JGS2Dynamics::CreatePipelines() {
    // Reuse shared PD infrastructure shaders
    pd_init_pipeline_ = MakePipeline("ext_pd_common/pd_init.wgsl", "jgs2_init");
    pd_predict_pipeline_ = MakePipeline("ext_pd_common/pd_predict.wgsl", "jgs2_predict");
    pd_copy_pipeline_ = MakePipeline("ext_pd_common/pd_copy_vec4.wgsl", "jgs2_copy");

    // JGS2-specific pipelines
    accum_inertia_pipeline_ = MakePipeline("ext_jgs2/jgs2_accum_inertia.wgsl", "jgs2_accum_inertia");
    local_solve_pipeline_ = MakePipeline("ext_jgs2/jgs2_local_solve.wgsl", "jgs2_local_solve");
}

void JGS2Dynamics::CacheBindGroups(WGPUBuffer position_buffer,
                                    WGPUBuffer velocity_buffer,
                                    WGPUBuffer mass_buffer) {
    uint64 params_sz = sizeof(SolverParams);
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 mass_sz = uint64(node_count_) * sizeof(SimMass);
    uint64 grad_sz = uint64(node_count_) * 4 * sizeof(uint32);
    uint64 hess_sz = uint64(node_count_) * 9 * sizeof(float32);
    uint64 corr_sz = uint64(node_count_) * 9 * sizeof(float32);

    WGPUBuffer phys_h = physics_buffer_;
    uint64 phys_sz = physics_size_;
    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer x_old_h = x_old_buffer_->GetHandle();
    WGPUBuffer s_h = s_buffer_->GetHandle();
    WGPUBuffer q_h = q_buffer_->GetHandle();
    WGPUBuffer grad_h = gradient_buffer_->GetHandle();
    WGPUBuffer hess_h = hessian_diag_buffer_->GetHandle();
    WGPUBuffer corr_h = correction_buffer_->GetHandle();

    // pd_init: x_old = positions
    bg_init_ = MakeBG(pd_init_pipeline_, "bg_jgs2_init",
        {{0, {params_h, params_sz}},
         {1, {position_buffer, vec_sz}},
         {2, {x_old_h, vec_sz}}});

    // pd_predict: s = x_old + dt*v + dt²*g
    bg_predict_ = MakeBG(pd_predict_pipeline_, "bg_jgs2_predict",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {x_old_h, vec_sz}},
         {3, {velocity_buffer, vec_sz}}, {4, {mass_buffer, mass_sz}},
         {5, {s_h, vec_sz}}});

    // pd_copy: q = s (initial guess)
    bg_copy_q_from_s_ = MakeBG(pd_copy_pipeline_, "bg_jgs2_copy_q_s",
        {{0, {params_h, params_sz}},
         {1, {s_h, vec_sz}},
         {2, {q_h, vec_sz}}});

    // jgs2_accum_inertia: gradient + Hessian from inertia
    bg_accum_inertia_ = MakeBG(accum_inertia_pipeline_, "bg_jgs2_inertia",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {q_h, vec_sz}}, {3, {s_h, vec_sz}},
         {4, {mass_buffer, mass_sz}},
         {5, {grad_h, grad_sz}}, {6, {hess_h, hess_sz}}});

    // jgs2_local_solve: δx = -H⁻¹·g, q += δx
    bg_local_solve_ = MakeBG(local_solve_pipeline_, "bg_jgs2_solve",
        {{0, {params_h, params_sz}},
         {1, {grad_h, grad_sz}}, {2, {hess_h, hess_sz}},
         {3, {q_h, vec_sz}}, {4, {mass_buffer, mass_sz}},
         {5, {corr_h, corr_sz}}});
}

void JGS2Dynamics::Solve(WGPUCommandEncoder encoder) {
    // 1. Init: x_old = positions
    Dispatch(encoder, pd_init_pipeline_, bg_init_, node_wg_count_);

    // 2. Predict: s = x_old + dt*v + dt²*g
    Dispatch(encoder, pd_predict_pipeline_, bg_predict_, node_wg_count_);

    // 3. Initial guess: q = s
    Dispatch(encoder, pd_copy_pipeline_, bg_copy_q_from_s_, node_wg_count_);

    // 4. Block Jacobi iteration loop
    for (uint32 k = 0; k < iterations_; ++k) {
        // a. Accumulate inertia (atomicStore — initializes gradient + Hessian)
        Dispatch(encoder, accum_inertia_pipeline_, bg_accum_inertia_, node_wg_count_);

        // b. Accumulate spring terms (atomicAddFloat on top of inertia)
        for (auto& term : terms_) {
            term->Accumulate(encoder);
        }

        // c. Per-vertex block solve: δx = -H⁻¹·g, q += δx
        Dispatch(encoder, local_solve_pipeline_, bg_local_solve_, node_wg_count_);
    }
}

void JGS2Dynamics::UploadCorrection(const std::vector<float32>& correction_data) {
    if (!correction_buffer_) return;
    correction_buffer_->WriteData(std::span<const float32>(correction_data));
    enable_correction_ = true;
    LogInfo("JGS2Dynamics: correction uploaded (", correction_data.size() / 9, " vertices)");
}

WGPUBuffer JGS2Dynamics::GetQBuffer() const {
    return q_buffer_ ? q_buffer_->GetHandle() : nullptr;
}

WGPUBuffer JGS2Dynamics::GetXOldBuffer() const {
    return x_old_buffer_ ? x_old_buffer_->GetHandle() : nullptr;
}

WGPUBuffer JGS2Dynamics::GetParamsBuffer() const {
    return params_buffer_ ? params_buffer_->GetHandle() : nullptr;
}

uint64 JGS2Dynamics::GetParamsSize() const {
    return sizeof(SolverParams);
}

uint64 JGS2Dynamics::GetVec4BufferSize() const {
    return uint64(node_count_) * 4 * sizeof(float32);
}

void JGS2Dynamics::Shutdown() {
    for (auto& term : terms_) {
        term->Shutdown();
    }
    terms_.clear();

    bg_init_ = {};
    bg_predict_ = {};
    bg_copy_q_from_s_ = {};
    bg_accum_inertia_ = {};
    bg_local_solve_ = {};

    pd_init_pipeline_ = {};
    pd_predict_pipeline_ = {};
    pd_copy_pipeline_ = {};
    accum_inertia_pipeline_ = {};
    local_solve_pipeline_ = {};

    params_buffer_.reset();
    x_old_buffer_.reset();
    s_buffer_.reset();
    q_buffer_.reset();
    gradient_buffer_.reset();
    hessian_diag_buffer_.reset();
    correction_buffer_.reset();

    LogInfo("JGS2Dynamics: shutdown");
}

}  // namespace ext_jgs2
