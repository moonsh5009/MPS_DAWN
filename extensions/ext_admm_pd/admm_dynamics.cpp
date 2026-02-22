#include "ext_admm_pd/admm_dynamics.h"
#include "core_simulate/sim_components.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <span>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;

namespace ext_admm_pd {

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
// SpMVOperator (inner class)
// ============================================================================

ADMMDynamics::SpMVOperator::SpMVOperator(ADMMDynamics& owner)
    : owner_(owner) {}

void ADMMDynamics::SpMVOperator::PrepareSolve(
    WGPUBuffer p_buffer, uint64 p_size,
    WGPUBuffer ap_buffer, uint64 ap_size) {
    uint64 row_ptr_sz = owner_.csr_row_ptr_buffer_->GetByteLength();
    uint64 col_idx_sz = owner_.csr_col_idx_buffer_->GetByteLength();
    uint64 csr_val_sz = owner_.csr_values_buffer_->GetByteLength();
    uint64 diag_sz = uint64(owner_.node_count_) * 9 * sizeof(float32);

    bind_group_ = MakeBG(owner_.spmv_pipeline_, "bg_admm_spmv",
        {{0, {owner_.params_buffer_->GetHandle(), sizeof(SolverParams)}},
         {1, {p_buffer, p_size}},
         {2, {ap_buffer, ap_size}},
         {3, {owner_.csr_row_ptr_buffer_->GetHandle(), row_ptr_sz}},
         {4, {owner_.csr_col_idx_buffer_->GetHandle(), col_idx_sz}},
         {5, {owner_.csr_values_buffer_->GetHandle(), csr_val_sz}},
         {6, {owner_.diag_buffer_->GetHandle(), diag_sz}}});
}

void ADMMDynamics::SpMVOperator::Apply(WGPUCommandEncoder encoder, uint32 workgroup_count) {
    Dispatch(encoder, owner_.spmv_pipeline_, bind_group_, workgroup_count);
}

// ============================================================================
// ADMMDynamics
// ============================================================================

ADMMDynamics::ADMMDynamics() = default;

ADMMDynamics::~ADMMDynamics() = default;

void ADMMDynamics::AddTerm(std::unique_ptr<IProjectiveTerm> term) {
    terms_.push_back(std::move(term));
}

void ADMMDynamics::Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
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

    BuildSparsity();
    CreateBuffers(position_buffer, velocity_buffer, mass_buffer);
    CreatePipelines();

    // Build PDAssemblyContext for term bind group caching.
    // Note: rhs_buffer points to CG solver's RHS buffer so terms scatter directly into it.
    PDAssemblyContext ctx{};
    ctx.physics_buffer = physics_buffer;
    ctx.physics_size = physics_size;
    ctx.q_buffer = q_curr_buffer_->GetHandle();
    ctx.s_buffer = s_buffer_->GetHandle();
    ctx.mass_buffer = mass_buffer;
    ctx.rhs_buffer = cg_solver_->GetRHSBuffer();  // terms write RHS directly into CG buffer
    ctx.diag_buffer = diag_buffer_->GetHandle();
    ctx.csr_values_buffer = csr_values_buffer_->GetHandle();
    ctx.params_buffer = params_buffer_->GetHandle();
    ctx.node_count = node_count;
    ctx.edge_count = edge_count;
    ctx.workgroup_size = workgroup_size;
    ctx.params_size = sizeof(SolverParams);

    // Initialize terms (Chebyshev path: LHS + RHS bind groups)
    for (auto& term : terms_) {
        term->Initialize(*sparsity_, ctx);
    }

    // Initialize ADMM-specific term resources (z/u buffers, ADMM pipelines)
    for (auto& term : terms_) {
        term->InitializeADMM(ctx);
    }

    // Cache solver bind groups
    CacheBindGroups(position_buffer, velocity_buffer, mass_buffer);

    // Build constant LHS once at init
    {
        auto& gpu_inst = GPUCore::GetInstance();
        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu_inst.GetDevice(), &enc_desc);

        uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);
        uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);

        // Clear diagonal and CSR values
        wgpuCommandEncoderClearBuffer(encoder, diag_buffer_->GetHandle(), 0, diag_sz);
        if (csr_val_sz > 0) {
            wgpuCommandEncoderClearBuffer(encoder, csr_values_buffer_->GetHandle(), 0, csr_val_sz);
        }

        // Inertial LHS: diag += M/dt^2 * I3
        Dispatch(encoder, pd_inertial_lhs_pipeline_, bg_inertial_lhs_, node_wg_count_);

        // Term LHS contributions (constant w * S^T * S)
        for (auto& term : terms_) {
            term->AssembleLHS(encoder);
        }

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu_inst.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    // Cache CG bind groups (after LHS is built since SpMV needs diag/csr)
    uint64 mass_sz = uint64(node_count) * sizeof(SimMass);
    spmv_ = std::make_unique<SpMVOperator>(*this);
    cg_solver_->CacheBindGroups(physics_buffer, physics_size,
                                 params_buffer_->GetHandle(), sizeof(SolverParams),
                                 mass_buffer, mass_sz,
                                 *spmv_);

    LogInfo("ADMMDynamics: initialized (", node_count_, " nodes, ",
            edge_count_, " edges, ", face_count_, " faces, nnz=", nnz_,
            ", ", terms_.size(), " terms, admm_iters=", admm_iterations_,
            ", cg_iters=", cg_iterations_, ")");
}

void ADMMDynamics::BuildSparsity() {
    sparsity_ = std::make_unique<SparsityBuilder>(node_count_);

    for (auto& term : terms_) {
        term->DeclareSparsity(*sparsity_);
    }

    sparsity_->Build();
    nnz_ = sparsity_->GetNNZ();
}

void ADMMDynamics::CreateBuffers(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                                  WGPUBuffer mass_buffer) {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);

    // Solver params uniform
    params_.node_count = node_count_;
    params_.edge_count = edge_count_;
    params_.face_count = face_count_;
    params_buffer_ = std::make_unique<GPUBuffer<SolverParams>>(
        BufferUsage::Uniform, std::span<const SolverParams>(&params_, 1), "admm_solver_params");

    // CSR structure
    const auto& row_ptr = sparsity_->GetRowPtr();
    const auto& col_idx = sparsity_->GetColIdx();
    csr_row_ptr_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage, std::span<const uint32>(row_ptr), "admm_csr_row_ptr");
    uint64 col_idx_sz = std::max(uint64(col_idx.size()) * sizeof(uint32), uint64(4));
    csr_col_idx_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferConfig{.usage = srw, .size = col_idx_sz, .label = "admm_csr_col_idx"});
    if (!col_idx.empty()) {
        csr_col_idx_buffer_->WriteData(std::span<const uint32>(col_idx));
    }
    uint64 csr_val_min = std::max(uint64(nnz_) * 9 * sizeof(float32), uint64(4));
    csr_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = csr_val_min, .label = "admm_csr_values"});

    // Diagonal
    diag_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = diag_sz, .label = "admm_diag"});

    // Solver buffers
    x_old_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "admm_x_old"});
    s_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "admm_s"});
    q_curr_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "admm_q_curr"});

    // CG solver (inner solve)
    cg_solver_ = std::make_unique<CGSolver>();
    cg_solver_->Initialize(node_count_, workgroup_size_);
}

void ADMMDynamics::CreatePipelines() {
    // Shared PD infrastructure shaders (from ext_pd_common)
    pd_init_pipeline_ = MakePipeline("ext_pd_common/pd_init.wgsl", "admm_pd_init");
    pd_predict_pipeline_ = MakePipeline("ext_pd_common/pd_predict.wgsl", "admm_pd_predict");
    pd_copy_pipeline_ = MakePipeline("ext_pd_common/pd_copy_vec4.wgsl", "admm_pd_copy");
    pd_mass_rhs_pipeline_ = MakePipeline("ext_pd_common/pd_mass_rhs.wgsl", "admm_pd_mass_rhs");
    pd_inertial_lhs_pipeline_ = MakePipeline("ext_pd_common/pd_inertial_lhs.wgsl", "admm_pd_inertial_lhs");

    // ADMM-specific SpMV (reuses Newton's cg_spmv shader pattern)
    spmv_pipeline_ = MakePipeline("ext_admm_pd/admm_cg_spmv.wgsl", "admm_cg_spmv");
}

void ADMMDynamics::CacheBindGroups(WGPUBuffer position_buffer,
                                    WGPUBuffer velocity_buffer,
                                    WGPUBuffer mass_buffer) {
    uint64 params_sz = sizeof(SolverParams);
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 mass_sz = uint64(node_count_) * sizeof(SimMass);
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);

    WGPUBuffer phys_h = physics_buffer_;
    uint64 phys_sz = physics_size_;
    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer x_old_h = x_old_buffer_->GetHandle();
    WGPUBuffer s_h = s_buffer_->GetHandle();
    WGPUBuffer q_curr_h = q_curr_buffer_->GetHandle();

    // pd_init: x_old = positions
    bg_init_ = MakeBG(pd_init_pipeline_, "bg_admm_init",
        {{0, {params_h, params_sz}},
         {1, {position_buffer, vec_sz}},
         {2, {x_old_h, vec_sz}}});

    // pd_predict: s = x_old + dt*v + dt^2*g
    bg_predict_ = MakeBG(pd_predict_pipeline_, "bg_admm_predict",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {x_old_h, vec_sz}},
         {3, {velocity_buffer, vec_sz}}, {4, {mass_buffer, mass_sz}},
         {5, {s_h, vec_sz}}});

    // pd_copy: q_curr = s (initial guess)
    bg_copy_q_from_s_ = MakeBG(pd_copy_pipeline_, "bg_admm_copy_q_s",
        {{0, {params_h, params_sz}},
         {1, {s_h, vec_sz}},
         {2, {q_curr_h, vec_sz}}});

    // pd_mass_rhs: rhs += (M/dt^2) * s (writes to CG RHS buffer)
    uint64 rhs_sz = uint64(node_count_) * 4 * sizeof(uint32);
    bg_mass_rhs_ = MakeBG(pd_mass_rhs_pipeline_, "bg_admm_mass_rhs",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {mass_buffer, mass_sz}},
         {3, {s_h, vec_sz}}, {4, {cg_solver_->GetRHSBuffer(), rhs_sz}}});

    // pd_inertial_lhs: diag += (M/dt^2) * I3
    bg_inertial_lhs_ = MakeBG(pd_inertial_lhs_pipeline_, "bg_admm_inertial_lhs",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {mass_buffer, mass_sz}},
         {3, {diag_buffer_->GetHandle(), diag_sz}}});
}

void ADMMDynamics::Solve(WGPUCommandEncoder encoder) {
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 rhs_sz = uint64(node_count_) * 4 * sizeof(uint32);

    // Step 1: x_old = positions
    Dispatch(encoder, pd_init_pipeline_, bg_init_, node_wg_count_);

    // Step 2: s = x_old + dt*v + dt^2*g
    Dispatch(encoder, pd_predict_pipeline_, bg_predict_, node_wg_count_);

    // Step 3: q = s (initial guess)
    Dispatch(encoder, pd_copy_pipeline_, bg_copy_q_from_s_, node_wg_count_);

    // Step 4: Reset dual variables for all terms (z=S*s, u=0)
    for (auto& term : terms_) {
        term->ResetDual(encoder);
    }

    // ADMM loop
    for (uint32 k = 0; k < admm_iterations_; ++k) {
        // --- Global step: solve A*q = rhs via CG ---

        // Clear CG RHS buffer
        wgpuCommandEncoderClearBuffer(encoder, cg_solver_->GetRHSBuffer(), 0, rhs_sz);

        // Inertial RHS: rhs += (M/dt^2) * s
        Dispatch(encoder, pd_mass_rhs_pipeline_, bg_mass_rhs_, node_wg_count_);

        // Term ADMM RHS: rhs += w * S^T * (z - u)
        for (auto& term : terms_) {
            term->AssembleADMMRHS(encoder);
        }

        // CG solve: A*x = rhs
        cg_solver_->Solve(encoder, cg_iterations_);

        // Copy CG solution to q_curr
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            cg_solver_->GetSolutionBuffer(), 0,
            q_curr_buffer_->GetHandle(), 0, vec_sz);

        // --- Local step: z = project(S*q + u) ---
        for (auto& term : terms_) {
            term->ProjectLocal(encoder);
        }

        // --- Dual step: u += S*q - z ---
        for (auto& term : terms_) {
            term->UpdateDual(encoder);
        }
    }
}

WGPUBuffer ADMMDynamics::GetQCurrBuffer() const {
    return q_curr_buffer_ ? q_curr_buffer_->GetHandle() : nullptr;
}

WGPUBuffer ADMMDynamics::GetXOldBuffer() const {
    return x_old_buffer_ ? x_old_buffer_->GetHandle() : nullptr;
}

WGPUBuffer ADMMDynamics::GetParamsBuffer() const {
    return params_buffer_ ? params_buffer_->GetHandle() : nullptr;
}

uint64 ADMMDynamics::GetParamsSize() const {
    return sizeof(SolverParams);
}

uint64 ADMMDynamics::GetVec4BufferSize() const {
    return uint64(node_count_) * 4 * sizeof(float32);
}

void ADMMDynamics::Shutdown() {
    for (auto& term : terms_) {
        term->Shutdown();
    }
    terms_.clear();

    if (cg_solver_) cg_solver_->Shutdown();
    cg_solver_.reset();
    spmv_.reset();

    bg_init_ = {};
    bg_predict_ = {};
    bg_copy_q_from_s_ = {};
    bg_mass_rhs_ = {};
    bg_inertial_lhs_ = {};

    pd_init_pipeline_ = {};
    pd_predict_pipeline_ = {};
    pd_copy_pipeline_ = {};
    pd_mass_rhs_pipeline_ = {};
    pd_inertial_lhs_pipeline_ = {};
    spmv_pipeline_ = {};

    params_buffer_.reset();
    csr_row_ptr_buffer_.reset();
    csr_col_idx_buffer_.reset();
    csr_values_buffer_.reset();
    diag_buffer_.reset();
    x_old_buffer_.reset();
    s_buffer_.reset();
    q_curr_buffer_.reset();
    sparsity_.reset();

    LogInfo("ADMMDynamics: shutdown");
}

}  // namespace ext_admm_pd
