#include "core_simulate/cg_solver.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;

namespace mps {
namespace simulate {

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
    auto shader = ShaderLoader::CreateModule("core_simulate/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// CGSolver
// ============================================================================

CGSolver::CGSolver() = default;
CGSolver::~CGSolver() = default;

void CGSolver::Initialize(uint32 node_count, uint32 workgroup_size) {
    node_count_ = node_count;
    workgroup_size_ = workgroup_size;
    workgroup_count_ = (node_count + workgroup_size - 1) / workgroup_size;
    dot_partial_count_ = workgroup_count_;

    CreateBuffers();
    CreatePipelines();
    LogInfo("CGSolver: initialized (", node_count_, " nodes)");
}

void CGSolver::CreateBuffers() {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);

    cg_x_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_x"});
    cg_r_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_r"});
    cg_p_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_p"});
    cg_ap_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_ap"});

    uint64 partial_sz = uint64(dot_partial_count_) * sizeof(float32);
    uint64 scalar_sz = 8 * sizeof(float32);
    partial_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = partial_sz, .label = "cg_partials"});
    scalar_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = scalar_sz, .label = "cg_scalars"});

    // Dot product config uniforms
    DotConfig dc_rr{0, dot_partial_count_};
    DotConfig dc_pap{1, dot_partial_count_};
    DotConfig dc_rr_new{2, dot_partial_count_};
    dc_rr_ = std::make_unique<GPUBuffer<DotConfig>>(BufferUsage::Uniform, std::span<const DotConfig>(&dc_rr, 1), "dc_rr");
    dc_pap_ = std::make_unique<GPUBuffer<DotConfig>>(BufferUsage::Uniform, std::span<const DotConfig>(&dc_pap, 1), "dc_pap");
    dc_rr_new_ = std::make_unique<GPUBuffer<DotConfig>>(BufferUsage::Uniform, std::span<const DotConfig>(&dc_rr_new, 1), "dc_rr_new");

    // Scalar mode uniforms
    ScalarMode mode_alpha{0};
    ScalarMode mode_beta{1};
    mode_alpha_ = std::make_unique<GPUBuffer<ScalarMode>>(BufferUsage::Uniform, std::span<const ScalarMode>(&mode_alpha, 1), "cg_mode_alpha");
    mode_beta_ = std::make_unique<GPUBuffer<ScalarMode>>(BufferUsage::Uniform, std::span<const ScalarMode>(&mode_beta, 1), "cg_mode_beta");
}

void CGSolver::CreatePipelines() {
    cg_init_pipeline_ = MakePipeline("cg_init.wgsl", "cg_init");
    cg_dot_pipeline_ = MakePipeline("cg_dot.wgsl", "cg_dot");
    cg_dot_final_pipeline_ = MakePipeline("cg_dot_final.wgsl", "cg_dot_final");
    cg_compute_scalars_pipeline_ = MakePipeline("cg_compute_scalars.wgsl", "cg_compute_scalars");
    cg_update_xr_pipeline_ = MakePipeline("cg_update_xr.wgsl", "cg_update_xr");
    cg_update_p_pipeline_ = MakePipeline("cg_update_p.wgsl", "cg_update_p");
}

WGPUBuffer CGSolver::GetRHSBuffer() const { return cg_r_ ? cg_r_->GetHandle() : nullptr; }
WGPUBuffer CGSolver::GetSolutionBuffer() const { return cg_x_ ? cg_x_->GetHandle() : nullptr; }
uint64 CGSolver::GetVectorSize() const { return uint64(node_count_) * 4 * sizeof(float32); }

void CGSolver::CacheBindGroups(WGPUBuffer physics_buffer, uint64 physics_size,
                               WGPUBuffer params_buffer, uint64 params_size,
                               WGPUBuffer mass_buffer, uint64 mass_size,
                               ISpMVOperator& spmv) {
    uint64 vec_sz = GetVectorSize();
    uint64 partial_sz = uint64(dot_partial_count_) * sizeof(float32);
    uint64 scalar_sz = 8 * sizeof(float32);

    WGPUBuffer x_h = cg_x_->GetHandle();
    WGPUBuffer r_h = cg_r_->GetHandle();
    WGPUBuffer p_h = cg_p_->GetHandle();
    WGPUBuffer ap_h = cg_ap_->GetHandle();
    WGPUBuffer partial_h = partial_->GetHandle();
    WGPUBuffer scalar_h = scalar_->GetHandle();

    bg_init_ = MakeBG(cg_init_pipeline_, "bg_cg_init",
        {{0, {params_buffer, params_size}},
         {1, {x_h, vec_sz}}, {2, {r_h, vec_sz}}, {3, {p_h, vec_sz}}});

    bg_dot_rr_ = MakeBG(cg_dot_pipeline_, "bg_dot_rr",
        {{0, {params_buffer, params_size}},
         {1, {r_h, vec_sz}}, {2, {r_h, vec_sz}}, {3, {partial_h, partial_sz}}});

    bg_dot_pap_ = MakeBG(cg_dot_pipeline_, "bg_dot_pap",
        {{0, {params_buffer, params_size}},
         {1, {p_h, vec_sz}}, {2, {ap_h, vec_sz}}, {3, {partial_h, partial_sz}}});

    bg_df_rr_ = MakeBG(cg_dot_final_pipeline_, "bg_df_rr",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_rr_->GetHandle(), sizeof(DotConfig)}}});
    bg_df_pap_ = MakeBG(cg_dot_final_pipeline_, "bg_df_pap",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_pap_->GetHandle(), sizeof(DotConfig)}}});
    bg_df_rr_new_ = MakeBG(cg_dot_final_pipeline_, "bg_df_rr_new",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_rr_new_->GetHandle(), sizeof(DotConfig)}}});

    bg_alpha_ = MakeBG(cg_compute_scalars_pipeline_, "bg_alpha",
        {{0, {scalar_h, scalar_sz}}, {1, {mode_alpha_->GetHandle(), sizeof(ScalarMode)}}});
    bg_beta_ = MakeBG(cg_compute_scalars_pipeline_, "bg_beta",
        {{0, {scalar_h, scalar_sz}}, {1, {mode_beta_->GetHandle(), sizeof(ScalarMode)}}});

    bg_xr_ = MakeBG(cg_update_xr_pipeline_, "bg_xr",
        {{0, {params_buffer, params_size}},
         {1, {x_h, vec_sz}}, {2, {r_h, vec_sz}},
         {3, {p_h, vec_sz}}, {4, {ap_h, vec_sz}}, {5, {scalar_h, scalar_sz}},
         {6, {mass_buffer, mass_size}}});

    bg_p_ = MakeBG(cg_update_p_pipeline_, "bg_p",
        {{0, {params_buffer, params_size}},
         {1, {r_h, vec_sz}}, {2, {p_h, vec_sz}},
         {3, {scalar_h, scalar_sz}}, {4, {mass_buffer, mass_size}}});

    // Prepare SpMV operator with CG buffers and cache pointer
    spmv.PrepareSolve(p_h, vec_sz, ap_h, vec_sz);
    spmv_ = &spmv;

    LogInfo("CGSolver: bind groups cached");
}

void CGSolver::Solve(WGPUCommandEncoder encoder, uint32 cg_iterations) {
    uint64 scalar_sz = 8 * sizeof(float32);
    WGPUBuffer scalar_h = scalar_->GetHandle();

    // Clear scalar buffer
    wgpuCommandEncoderClearBuffer(encoder, scalar_h, 0, scalar_sz);

    // CG init: x = 0, p = r
    Dispatch(encoder, cg_init_pipeline_, bg_init_, workgroup_count_);

    // Initial rr = dot(r, r) → scalars[0]
    Dispatch(encoder, cg_dot_pipeline_, bg_dot_rr_, workgroup_count_);
    Dispatch(encoder, cg_dot_final_pipeline_, bg_df_rr_, 1);

    // CG iterations
    for (uint32 cit = 0; cit < cg_iterations; ++cit) {
        // Ap = A * p (SpMV operator dispatches with its own cached bind group)
        spmv_->Apply(encoder, workgroup_count_);

        // pAp = dot(p, Ap) → scalars[1]
        Dispatch(encoder, cg_dot_pipeline_, bg_dot_pap_, workgroup_count_);
        Dispatch(encoder, cg_dot_final_pipeline_, bg_df_pap_, 1);

        // alpha = rr / pAp → scalars[3]
        Dispatch(encoder, cg_compute_scalars_pipeline_, bg_alpha_, 1);

        // x += alpha*p, r -= alpha*Ap
        Dispatch(encoder, cg_update_xr_pipeline_, bg_xr_, workgroup_count_);

        // rr_new = dot(r, r) → scalars[2]
        Dispatch(encoder, cg_dot_pipeline_, bg_dot_rr_, workgroup_count_);
        Dispatch(encoder, cg_dot_final_pipeline_, bg_df_rr_new_, 1);

        // beta = rr_new / rr, advance rr = rr_new
        Dispatch(encoder, cg_compute_scalars_pipeline_, bg_beta_, 1);

        // p = r + beta * p
        Dispatch(encoder, cg_update_p_pipeline_, bg_p_, workgroup_count_);
    }
}

void CGSolver::Shutdown() {
    // Release cached bind groups
    bg_init_ = {};
    bg_dot_rr_ = {};
    bg_dot_pap_ = {};
    bg_df_rr_ = {};
    bg_df_pap_ = {};
    bg_df_rr_new_ = {};
    bg_alpha_ = {};
    bg_beta_ = {};
    bg_xr_ = {};
    bg_p_ = {};
    spmv_ = nullptr;

    cg_init_pipeline_ = {};
    cg_dot_pipeline_ = {};
    cg_dot_final_pipeline_ = {};
    cg_compute_scalars_pipeline_ = {};
    cg_update_xr_pipeline_ = {};
    cg_update_p_pipeline_ = {};

    cg_x_.reset();
    cg_r_.reset();
    cg_p_.reset();
    cg_ap_.reset();
    partial_.reset();
    scalar_.reset();
    dc_rr_.reset();
    dc_pap_.reset();
    dc_rr_new_.reset();
    mode_alpha_.reset();
    mode_beta_.reset();

    LogInfo("CGSolver: shutdown");
}

}  // namespace simulate
}  // namespace mps
