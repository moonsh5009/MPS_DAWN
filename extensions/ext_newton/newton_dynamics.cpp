#include "ext_newton/newton_dynamics.h"
#include "core_simulate/sim_components.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>

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
    auto shader = ShaderLoader::CreateModule("ext_newton/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// SpMVOperator (internal)
// ============================================================================

NewtonDynamics::SpMVOperator::SpMVOperator(NewtonDynamics& owner)
    : owner_(owner) {}

void NewtonDynamics::SpMVOperator::PrepareSolve(
    WGPUBuffer p_buffer, uint64 p_size,
    WGPUBuffer ap_buffer, uint64 ap_size) {
    uint64 row_ptr_sz = owner_.csr_row_ptr_buffer_->GetByteLength();
    uint64 col_idx_sz = owner_.csr_col_idx_buffer_->GetByteLength();
    uint64 csr_val_sz = owner_.csr_values_buffer_->GetByteLength();
    uint64 diag_sz = uint64(owner_.node_count_) * 9 * sizeof(float32);

    bind_group_ = MakeBG(owner_.spmv_pipeline_, "bg_spmv",
        {{0, {owner_.params_buffer_->GetHandle(), sizeof(DynamicsParams)}},
         {1, {p_buffer, p_size}},
         {2, {ap_buffer, ap_size}},
         {3, {owner_.csr_row_ptr_buffer_->GetHandle(), row_ptr_sz}},
         {4, {owner_.csr_col_idx_buffer_->GetHandle(), col_idx_sz}},
         {5, {owner_.csr_values_buffer_->GetHandle(), csr_val_sz}},
         {6, {owner_.diag_values_buffer_->GetHandle(), diag_sz}}});
}

void NewtonDynamics::SpMVOperator::Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) {
    Dispatch(encoder, owner_.spmv_pipeline_, bind_group_, workgroup_count);
}

// ============================================================================
// NewtonDynamics
// ============================================================================

NewtonDynamics::NewtonDynamics() = default;

NewtonDynamics::~NewtonDynamics() = default;

void NewtonDynamics::AddTerm(std::unique_ptr<IDynamicsTerm> term) {
    terms_.push_back(std::move(term));
}

void NewtonDynamics::Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
                                 WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                                 WGPUBuffer mass_buffer, uint32 workgroup_size) {
    node_count_ = node_count;
    edge_count_ = edge_count;
    face_count_ = face_count;
    workgroup_size_ = workgroup_size;
    node_wg_count_ = (node_count + workgroup_size - 1) / workgroup_size;

    BuildSparsity();
    CreateBuffers();
    CreatePipelines();

    // Initialize CG solver
    cg_solver_ = std::make_unique<CGSolver>();
    cg_solver_->Initialize(node_count, workgroup_size);

    // Initialize SpMV operator
    spmv_ = std::make_unique<SpMVOperator>(*this);

    // Build AssemblyContext for term bind group caching
    AssemblyContext ctx{};
    ctx.position_buffer = position_buffer;
    ctx.velocity_buffer = velocity_buffer;
    ctx.mass_buffer = mass_buffer;
    ctx.force_buffer = force_buffer_->GetHandle();
    ctx.diag_buffer = diag_values_buffer_->GetHandle();
    ctx.csr_values_buffer = csr_values_buffer_->GetHandle();
    ctx.params_buffer = params_buffer_->GetHandle();
    ctx.dv_total_buffer = dv_total_buffer_->GetHandle();
    ctx.node_count = node_count;
    ctx.edge_count = edge_count;
    ctx.workgroup_size = workgroup_size;
    ctx.params_size = sizeof(DynamicsParams);

    // Initialize terms with context for bind group caching
    for (auto& term : terms_) {
        term->Initialize(*sparsity_, ctx);
    }

    // Cache Newton and CG bind groups
    CacheBindGroups(position_buffer, velocity_buffer, mass_buffer);

    LogInfo("NewtonDynamics: initialized (", node_count_, " nodes, ",
            edge_count_, " edges, nnz=", nnz_, ", ", terms_.size(), " terms)");
}

void NewtonDynamics::BuildSparsity() {
    sparsity_ = std::make_unique<SparsityBuilder>(node_count_);

    for (auto& term : terms_) {
        term->DeclareSparsity(*sparsity_);
    }

    sparsity_->Build();
    nnz_ = sparsity_->GetNNZ();
}

void NewtonDynamics::CreateBuffers() {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);

    // Params uniform
    params_.node_count = node_count_;
    params_.edge_count = edge_count_;
    params_.face_count = face_count_;
    params_buffer_ = std::make_unique<GPUBuffer<DynamicsParams>>(
        BufferUsage::Uniform, std::span<const DynamicsParams>(&params_, 1), "dynamics_params");

    // CSR structure (ensure minimum 4 bytes so GPU buffers are always valid)
    const auto& row_ptr = sparsity_->GetRowPtr();
    const auto& col_idx = sparsity_->GetColIdx();
    csr_row_ptr_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage, std::span<const uint32>(row_ptr), "csr_row_ptr");
    uint64 col_idx_sz = std::max(uint64(col_idx.size()) * sizeof(uint32), uint64(4));
    csr_col_idx_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferConfig{.usage = srw, .size = col_idx_sz, .label = "csr_col_idx"});
    if (!col_idx.empty()) {
        csr_col_idx_buffer_->WriteData(std::span<const uint32>(col_idx));
    }
    uint64 csr_val_min = std::max(uint64(nnz_) * 9 * sizeof(float32), uint64(4));
    csr_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = csr_val_min, .label = "csr_values"});
    diag_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(node_count_) * 9 * sizeof(float32), .label = "diag_values"});

    // Force buffer (atomic u32, N*4)
    force_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(node_count_) * 4 * sizeof(int32), .label = "forces"});

    // Newton solver buffers
    x_old_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "x_old"});
    dv_total_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "dv_total"});
}

void NewtonDynamics::CreatePipelines() {
    newton_init_pipeline_ = MakePipeline("newton_init.wgsl", "newton_init");
    newton_predict_pos_pipeline_ = MakePipeline("newton_predict_pos.wgsl", "newton_predict_pos");
    newton_accumulate_dv_pipeline_ = MakePipeline("newton_accumulate_dv.wgsl", "newton_accumulate_dv");
    clear_forces_pipeline_ = MakePipeline("clear_forces.wgsl", "clear_forces");
    assemble_rhs_pipeline_ = MakePipeline("assemble_rhs.wgsl", "assemble_rhs");
    spmv_pipeline_ = MakePipeline("cg_spmv.wgsl", "cg_spmv");
}

void NewtonDynamics::CacheBindGroups(WGPUBuffer position_buffer,
                                     WGPUBuffer velocity_buffer,
                                     WGPUBuffer mass_buffer) {
    uint64 params_sz = sizeof(DynamicsParams);
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 mass_sz = uint64(node_count_) * sizeof(simulate::SimMass);
    uint64 force_sz = uint64(node_count_) * 4 * sizeof(int32);

    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer force_h = force_buffer_->GetHandle();
    WGPUBuffer x_old_h = x_old_buffer_->GetHandle();
    WGPUBuffer dv_total_h = dv_total_buffer_->GetHandle();
    WGPUBuffer cg_x_h = cg_solver_->GetSolutionBuffer();
    WGPUBuffer rhs_h = cg_solver_->GetRHSBuffer();

    // Newton init bind group
    bg_newton_init_ = MakeBG(newton_init_pipeline_, "bg_newton_init",
        {{0, {params_h, params_sz}}, {1, {position_buffer, vec_sz}},
         {2, {x_old_h, vec_sz}}, {3, {dv_total_h, vec_sz}}});

    // Predict positions bind group
    bg_predict_ = MakeBG(newton_predict_pos_pipeline_, "bg_predict",
        {{0, {params_h, params_sz}}, {1, {position_buffer, vec_sz}},
         {2, {x_old_h, vec_sz}}, {3, {velocity_buffer, vec_sz}},
         {4, {dv_total_h, vec_sz}}, {5, {mass_buffer, mass_sz}}});

    // Clear forces bind group
    bg_clear_forces_ = MakeBG(clear_forces_pipeline_, "bg_clear_f",
        {{0, {params_h, params_sz}}, {1, {force_h, force_sz}}});

    // Assemble RHS bind group
    bg_rhs_ = MakeBG(assemble_rhs_pipeline_, "bg_rhs",
        {{0, {params_h, params_sz}}, {1, {force_h, force_sz}},
         {2, {dv_total_h, vec_sz}}, {3, {mass_buffer, mass_sz}},
         {4, {rhs_h, vec_sz}}});

    // Accumulate dv bind group
    bg_accumulate_ = MakeBG(newton_accumulate_dv_pipeline_, "bg_accum_dv",
        {{0, {params_h, params_sz}}, {1, {dv_total_h, vec_sz}},
         {2, {cg_x_h, vec_sz}}});

    // Cache CG solver bind groups
    cg_solver_->CacheBindGroups(params_h, params_sz, mass_buffer, mass_sz, *spmv_);
}

void NewtonDynamics::Solve(WGPUCommandEncoder encoder, float32 dt,
                           uint32 newton_iterations,
                           uint32 cg_iterations) {
    // Clamp dt
    dt = std::min(dt, 1.0f / 30.0f);

    // Update params
    params_.dt = dt;
    params_.cg_max_iter = cg_iterations;
    params_buffer_->WriteData(std::span<const DynamicsParams>(&params_, 1));

    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    WGPUBuffer diag_h = diag_values_buffer_->GetHandle();

    // ---- Newton Init: save x_old, zero dv_total ----
    Dispatch(encoder, newton_init_pipeline_, bg_newton_init_, node_wg_count_);

    // ---- Newton Outer Loop ----
    for (uint32 nit = 0; nit < newton_iterations; ++nit) {
        // Predict positions: x_temp = x_old + dt*(v + dv_total)
        Dispatch(encoder, newton_predict_pos_pipeline_, bg_predict_, node_wg_count_);

        // Clear forces
        Dispatch(encoder, clear_forces_pipeline_, bg_clear_forces_, node_wg_count_);

        // Clear diagonal Hessian buffer
        wgpuCommandEncoderClearBuffer(encoder, diag_h, 0, diag_sz);

        // Clear off-diagonal CSR values (if any edges)
        if (csr_val_sz > 0) {
            wgpuCommandEncoderClearBuffer(encoder, csr_values_buffer_->GetHandle(), 0, csr_val_sz);
        }

        // Assemble contributions from all terms (using cached bind groups)
        for (auto& term : terms_) {
            term->Assemble(encoder);
        }

        // Assemble RHS: b = dt*F - M*dv_total → writes to CG r buffer
        Dispatch(encoder, assemble_rhs_pipeline_, bg_rhs_, node_wg_count_);

        // CG Solve (uses cached bind groups)
        cg_solver_->Solve(encoder, cg_iterations);

        // Accumulate CG solution: dv_total += cg_x
        Dispatch(encoder, newton_accumulate_dv_pipeline_, bg_accumulate_, node_wg_count_);
    }
}

void NewtonDynamics::SetGravity(float32 gx, float32 gy, float32 gz) {
    params_.gravity_x = gx;
    params_.gravity_y = gy;
    params_.gravity_z = gz;
}

WGPUBuffer NewtonDynamics::GetDVTotalBuffer() const {
    return dv_total_buffer_ ? dv_total_buffer_->GetHandle() : nullptr;
}

WGPUBuffer NewtonDynamics::GetXOldBuffer() const {
    return x_old_buffer_ ? x_old_buffer_->GetHandle() : nullptr;
}

WGPUBuffer NewtonDynamics::GetParamsBuffer() const {
    return params_buffer_ ? params_buffer_->GetHandle() : nullptr;
}

uint64 NewtonDynamics::GetParamsSize() const {
    return sizeof(DynamicsParams);
}

uint64 NewtonDynamics::GetVec4BufferSize() const {
    return uint64(node_count_) * 4 * sizeof(float32);
}

void NewtonDynamics::Shutdown() {
    for (auto& term : terms_) {
        term->Shutdown();
    }
    terms_.clear();

    if (cg_solver_) cg_solver_->Shutdown();
    cg_solver_.reset();
    spmv_.reset();

    // Release cached bind groups
    bg_newton_init_ = {};
    bg_predict_ = {};
    bg_clear_forces_ = {};
    bg_rhs_ = {};
    bg_accumulate_ = {};

    newton_init_pipeline_ = {};
    newton_predict_pos_pipeline_ = {};
    newton_accumulate_dv_pipeline_ = {};
    clear_forces_pipeline_ = {};
    assemble_rhs_pipeline_ = {};
    spmv_pipeline_ = {};

    params_buffer_.reset();
    csr_row_ptr_buffer_.reset();
    csr_col_idx_buffer_.reset();
    csr_values_buffer_.reset();
    diag_values_buffer_.reset();
    force_buffer_.reset();
    x_old_buffer_.reset();
    dv_total_buffer_.reset();
    sparsity_.reset();

    LogInfo("NewtonDynamics: shutdown");
}

}  // namespace simulate
}  // namespace mps
