#include "ext_pd/pd_dynamics.h"
#include "core_simulate/sim_components.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_simulate/simulate_config.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cmath>
#include <span>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;

namespace ext_pd {

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
    auto shader = ShaderLoader::CreateModule("ext_pd/" + shader_path, label);
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = {label.data(), label.size()};
    desc.layout = nullptr;
    desc.compute.module = shader.GetHandle();
    std::string entry = "cs_main";
    desc.compute.entryPoint = {entry.data(), entry.size()};
    return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
}

// ============================================================================
// PDDynamics
// ============================================================================

PDDynamics::PDDynamics() = default;

PDDynamics::~PDDynamics() = default;

void PDDynamics::AddTerm(std::unique_ptr<IProjectiveTerm> term) {
    terms_.push_back(std::move(term));
}

void PDDynamics::Initialize(uint32 node_count, uint32 edge_count, uint32 face_count,
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

    // Build PDAssemblyContext for term bind group caching
    PDAssemblyContext ctx{};
    ctx.physics_buffer = physics_buffer;
    ctx.physics_size = physics_size;
    ctx.q_buffer = q_curr_buffer_->GetHandle();
    ctx.s_buffer = s_buffer_->GetHandle();
    ctx.mass_buffer = mass_buffer;
    ctx.rhs_buffer = rhs_buffer_->GetHandle();
    ctx.diag_buffer = diag_buffer_->GetHandle();
    ctx.csr_values_buffer = csr_values_buffer_->GetHandle();
    ctx.params_buffer = params_buffer_->GetHandle();
    ctx.node_count = node_count;
    ctx.edge_count = edge_count;
    ctx.workgroup_size = workgroup_size;
    ctx.params_size = sizeof(SolverParams);

    // Initialize terms with context for bind group caching
    for (auto& term : terms_) {
        term->Initialize(*sparsity_, ctx);
    }

    // Cache bind groups
    CacheBindGroups(position_buffer, velocity_buffer, mass_buffer);

    // Build LHS once at init (dt is known from GlobalPhysicsParams)
    {
        auto& gpu_inst = GPUCore::GetInstance();
        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu_inst.GetDevice(), &enc_desc);
        RebuildLHS(encoder);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu_inst.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    // Build Chebyshev params: manual ρ immediately, auto ρ deferred to CalibrateRho()
    if (chebyshev_rho_ > 0.0f) {
        BuildChebyshevParams(chebyshev_rho_);
        rho_calibrated_ = true;
    } else {
        // Fill staging with pure Jacobi as temporary fallback until calibration
        std::vector<JacobiParams> pure_jacobi(iterations_, {1.0f, 1, 0.0f, 0.0f});
        jacobi_staging_buffer_->WriteData(std::span<const JacobiParams>(pure_jacobi));
        rho_calibrated_ = false;
    }

    LogInfo("PDDynamics: initialized (", node_count_, " nodes, ",
            edge_count_, " edges, ", face_count_, " faces, nnz=", nnz_,
            ", ", terms_.size(), " terms, rho=",
            rho_calibrated_ ? std::to_string(chebyshev_rho_) : "auto-pending", ")");
}

void PDDynamics::BuildSparsity() {
    sparsity_ = std::make_unique<SparsityBuilder>(node_count_);

    for (auto& term : terms_) {
        term->DeclareSparsity(*sparsity_);
    }

    sparsity_->Build();
    nnz_ = sparsity_->GetNNZ();
}

void PDDynamics::CreateBuffers(WGPUBuffer position_buffer, WGPUBuffer velocity_buffer,
                                WGPUBuffer mass_buffer) {
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);

    // Solver params uniform
    params_.node_count = node_count_;
    params_.edge_count = edge_count_;
    params_.face_count = face_count_;
    params_buffer_ = std::make_unique<GPUBuffer<SolverParams>>(
        BufferUsage::Uniform, std::span<const SolverParams>(&params_, 1), "pd_solver_params");

    // Jacobi params uniform (CopyDst so command encoder can update per-iteration)
    jacobi_params_buffer_ = std::make_unique<GPUBuffer<JacobiParams>>(
        BufferConfig{.usage = BufferUsage::Uniform | BufferUsage::CopyDst,
                     .size = sizeof(JacobiParams), .label = "pd_jacobi_params"});

    // Chebyshev staging buffer (filled after LHS build when ρ is known)
    jacobi_staging_buffer_ = std::make_unique<GPUBuffer<JacobiParams>>(
        BufferConfig{.usage = BufferUsage::Storage | BufferUsage::CopySrc | BufferUsage::CopyDst,
                     .size = uint64(iterations_) * sizeof(JacobiParams),
                     .label = "pd_jacobi_staging"});

    // CSR structure
    const auto& row_ptr = sparsity_->GetRowPtr();
    const auto& col_idx = sparsity_->GetColIdx();
    csr_row_ptr_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage, std::span<const uint32>(row_ptr), "pd_csr_row_ptr");
    uint64 col_idx_sz = std::max(uint64(col_idx.size()) * sizeof(uint32), uint64(4));
    csr_col_idx_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferConfig{.usage = srw, .size = col_idx_sz, .label = "pd_csr_col_idx"});
    if (!col_idx.empty()) {
        csr_col_idx_buffer_->WriteData(std::span<const uint32>(col_idx));
    }
    uint64 csr_val_min = std::max(uint64(nnz_) * 9 * sizeof(float32), uint64(4));
    csr_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = csr_val_min, .label = "pd_csr_values"});

    // Diagonal + inverse
    diag_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = diag_sz, .label = "pd_diag"});
    d_inv_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = diag_sz, .label = "pd_d_inv"});

    // Solver buffers
    x_old_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "pd_x_old"});
    s_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "pd_s"});
    q_curr_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "pd_q_curr"});
    q_prev_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "pd_q_prev"});
    q_new_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = vec_sz, .label = "pd_q_new"});
    rhs_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(node_count_) * 4 * sizeof(uint32), .label = "pd_rhs"});
}

void PDDynamics::CreatePipelines() {
    pd_init_pipeline_ = MakePipeline("pd_init.wgsl", "pd_init");
    pd_predict_pipeline_ = MakePipeline("pd_predict.wgsl", "pd_predict");
    pd_copy_pipeline_ = MakePipeline("pd_copy_vec4.wgsl", "pd_copy");
    pd_mass_rhs_pipeline_ = MakePipeline("pd_mass_rhs.wgsl", "pd_mass_rhs");
    pd_inertial_lhs_pipeline_ = MakePipeline("pd_inertial_lhs.wgsl", "pd_inertial_lhs");
    pd_compute_d_inv_pipeline_ = MakePipeline("pd_compute_d_inv.wgsl", "pd_compute_d_inv");
    pd_jacobi_step_pipeline_ = MakePipeline("pd_jacobi_step.wgsl", "pd_jacobi_step");
}

void PDDynamics::CacheBindGroups(WGPUBuffer position_buffer,
                                  WGPUBuffer velocity_buffer,
                                  WGPUBuffer mass_buffer) {
    uint64 params_sz = sizeof(SolverParams);
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 mass_sz = uint64(node_count_) * sizeof(SimMass);
    uint64 rhs_sz = uint64(node_count_) * 4 * sizeof(uint32);
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);
    uint64 row_ptr_sz = csr_row_ptr_buffer_->GetByteLength();
    uint64 col_idx_sz = csr_col_idx_buffer_->GetByteLength();
    uint64 csr_val_sz = csr_values_buffer_->GetByteLength();

    WGPUBuffer phys_h = physics_buffer_;
    uint64 phys_sz = physics_size_;
    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer x_old_h = x_old_buffer_->GetHandle();
    WGPUBuffer s_h = s_buffer_->GetHandle();
    WGPUBuffer q_curr_h = q_curr_buffer_->GetHandle();
    WGPUBuffer q_prev_h = q_prev_buffer_->GetHandle();
    WGPUBuffer q_new_h = q_new_buffer_->GetHandle();
    WGPUBuffer rhs_h = rhs_buffer_->GetHandle();
    WGPUBuffer diag_h = diag_buffer_->GetHandle();
    WGPUBuffer d_inv_h = d_inv_buffer_->GetHandle();

    // pd_init: x_old = positions
    bg_init_ = MakeBG(pd_init_pipeline_, "bg_pd_init",
        {{0, {params_h, params_sz}},
         {1, {position_buffer, vec_sz}},
         {2, {x_old_h, vec_sz}}});

    // pd_predict: s = x_old + dt*v + dt²*g
    bg_predict_ = MakeBG(pd_predict_pipeline_, "bg_pd_predict",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {x_old_h, vec_sz}},
         {3, {velocity_buffer, vec_sz}}, {4, {mass_buffer, mass_sz}},
         {5, {s_h, vec_sz}}});

    // pd_copy: q_curr = s (initial guess)
    bg_copy_q_from_s_ = MakeBG(pd_copy_pipeline_, "bg_pd_copy_q_s",
        {{0, {params_h, params_sz}},
         {1, {s_h, vec_sz}},
         {2, {q_curr_h, vec_sz}}});

    // pd_mass_rhs: rhs += (M/dt²) * s
    bg_mass_rhs_ = MakeBG(pd_mass_rhs_pipeline_, "bg_pd_mass_rhs",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {mass_buffer, mass_sz}},
         {3, {s_h, vec_sz}}, {4, {rhs_h, rhs_sz}}});

    // pd_inertial_lhs: diag += (M/dt²) * I₃
    bg_inertial_lhs_ = MakeBG(pd_inertial_lhs_pipeline_, "bg_pd_inertial_lhs",
        {{0, {phys_h, phys_sz}}, {1, {params_h, params_sz}},
         {2, {mass_buffer, mass_sz}},
         {3, {diag_h, diag_sz}}});

    // pd_compute_d_inv: d_inv = inverse(diag)
    bg_compute_d_inv_ = MakeBG(pd_compute_d_inv_pipeline_, "bg_pd_d_inv",
        {{0, {params_h, params_sz}},
         {1, {diag_h, diag_sz}},
         {2, {d_inv_h, diag_sz}}});

    // pd_jacobi_step: fused SpMV + Jacobi + Chebyshev
    // rhs_buffer_ stores float bits as u32 via atomicAddFloat; binding as vec4f reinterprets correctly
    bg_jacobi_step_ = MakeBG(pd_jacobi_step_pipeline_, "bg_pd_jacobi_step",
        {{0, {params_h, params_sz}},
         {1, {q_curr_h, vec_sz}},
         {2, {csr_row_ptr_buffer_->GetHandle(), row_ptr_sz}},
         {3, {csr_col_idx_buffer_->GetHandle(), col_idx_sz}},
         {4, {csr_values_buffer_->GetHandle(), csr_val_sz}},
         {5, {rhs_h, rhs_sz}},
         {6, {d_inv_h, diag_sz}},
         {7, {q_prev_h, vec_sz}},
         {8, {q_new_h, vec_sz}},
         {9, {jacobi_params_buffer_->GetHandle(), sizeof(JacobiParams)}},
         {10, {mass_buffer, mass_sz}}});
}

void PDDynamics::RebuildLHS(WGPUCommandEncoder encoder) {
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);

    // Clear diagonal and CSR values
    wgpuCommandEncoderClearBuffer(encoder, diag_buffer_->GetHandle(), 0, diag_sz);
    if (csr_val_sz > 0) {
        wgpuCommandEncoderClearBuffer(encoder, csr_values_buffer_->GetHandle(), 0, csr_val_sz);
    }

    // Inertial LHS: diag += M/dt² * I₃
    Dispatch(encoder, pd_inertial_lhs_pipeline_, bg_inertial_lhs_, node_wg_count_);

    // Term LHS contributions
    for (auto& term : terms_) {
        term->AssembleLHS(encoder);
    }

    // Compute D⁻¹
    Dispatch(encoder, pd_compute_d_inv_pipeline_, bg_compute_d_inv_, node_wg_count_);
}

void PDDynamics::Solve(WGPUCommandEncoder encoder) {
    // Init: x_old = positions
    Dispatch(encoder, pd_init_pipeline_, bg_init_, node_wg_count_);

    // Predict: s = x_old + dt*v + dt²*g
    Dispatch(encoder, pd_predict_pipeline_, bg_predict_, node_wg_count_);

    // Initial guess: q_curr = s
    Dispatch(encoder, pd_copy_pipeline_, bg_copy_q_from_s_, node_wg_count_);

    // Wang 2015 single fused loop with correct Chebyshev 3-buffer rotation.
    // q_prev = q_{k-1}, q_curr = q_k, q_new = q_{k+1}
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 rhs_sz = uint64(node_count_) * 4 * sizeof(uint32);

    // Save initial guess: q_prev = q_0 (before iteration loop)
    wgpuCommandEncoderCopyBufferToBuffer(encoder,
        q_curr_buffer_->GetHandle(), 0,
        q_prev_buffer_->GetHandle(), 0, vec_sz);

    for (uint32 k = 0; k < iterations_; ++k) {
        // Clear RHS
        wgpuCommandEncoderClearBuffer(encoder, rhs_buffer_->GetHandle(), 0, rhs_sz);

        // Inertial RHS: rhs += (M/dt²) * s
        Dispatch(encoder, pd_mass_rhs_pipeline_, bg_mass_rhs_, node_wg_count_);

        // Fused local projection + RHS assembly per term
        for (auto& term : terms_) {
            term->ProjectRHS(encoder);
        }

        // Copy pre-computed Jacobi params for this iteration (staging → uniform)
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            jacobi_staging_buffer_->GetHandle(), uint64(k) * sizeof(JacobiParams),
            jacobi_params_buffer_->GetHandle(), 0, sizeof(JacobiParams));

        // Fused SpMV + Jacobi + Chebyshev: q_new = ω*(D⁻¹*(b-(A-D)*q_curr) - q_prev) + q_prev
        Dispatch(encoder, pd_jacobi_step_pipeline_, bg_jacobi_step_, node_wg_count_);

        // 3-buffer rotation: prev ← curr, curr ← new
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            q_curr_buffer_->GetHandle(), 0,
            q_prev_buffer_->GetHandle(), 0, vec_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            q_new_buffer_->GetHandle(), 0,
            q_curr_buffer_->GetHandle(), 0, vec_sz);
    }
}

WGPUBuffer PDDynamics::GetQCurrBuffer() const {
    return q_curr_buffer_ ? q_curr_buffer_->GetHandle() : nullptr;
}

WGPUBuffer PDDynamics::GetXOldBuffer() const {
    return x_old_buffer_ ? x_old_buffer_->GetHandle() : nullptr;
}

WGPUBuffer PDDynamics::GetParamsBuffer() const {
    return params_buffer_ ? params_buffer_->GetHandle() : nullptr;
}

uint64 PDDynamics::GetParamsSize() const {
    return sizeof(SolverParams);
}

uint64 PDDynamics::GetVec4BufferSize() const {
    return uint64(node_count_) * 4 * sizeof(float32);
}

// ---------------------------------------------------------------------------
// Debug readback — reads GPU buffers to CPU and logs values for sample nodes.
// ---------------------------------------------------------------------------
static std::vector<float32> ReadbackBuffer(WGPUBuffer src, uint64 size) {
    auto& gpu = GPUCore::GetInstance();

    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    bd.size  = size;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(gpu.GetDevice(), &bd);

    WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &ed);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, staging, 0, size);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    simulate::WaitForGPU();

    struct Ctx { bool done = false; };
    Ctx ctx;
    WGPUBufferMapCallbackInfo mi = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mi.mode = WGPUCallbackMode_WaitAnyOnly;
    mi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* ud, void*) {
        static_cast<Ctx*>(ud)->done = true;
    };
    mi.userdata1 = &ctx;
    WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, mi);
    WGPUFutureWaitInfo wi = WGPU_FUTURE_WAIT_INFO_INIT;
    wi.future = future;
    wgpuInstanceWaitAny(gpu.GetWGPUInstance(), 1, &wi, UINT64_MAX);

    const float32* ptr = static_cast<const float32*>(
        wgpuBufferGetConstMappedRange(staging, 0, size));
    std::vector<float32> result(ptr, ptr + size / sizeof(float32));
    wgpuBufferUnmap(staging);
    wgpuBufferRelease(staging);
    return result;
}

void PDDynamics::DebugDump() {
    simulate::WaitForGPU();

    uint64 vec_sz  = uint64(node_count_) * 4 * sizeof(float32);
    uint64 diag_sz = uint64(node_count_) * 9 * sizeof(float32);

    auto x_old = ReadbackBuffer(x_old_buffer_->GetHandle(), vec_sz);
    auto s     = ReadbackBuffer(s_buffer_->GetHandle(), vec_sz);
    auto rhs   = ReadbackBuffer(rhs_buffer_->GetHandle(), vec_sz);
    auto diag  = ReadbackBuffer(diag_buffer_->GetHandle(), diag_sz);
    auto d_inv = ReadbackBuffer(d_inv_buffer_->GetHandle(), diag_sz);
    auto q     = ReadbackBuffer(q_curr_buffer_->GetHandle(), vec_sz);
    auto q_prev_d = ReadbackBuffer(q_prev_buffer_->GetHandle(), vec_sz);

    // Also read jacobi staging (first 5 entries)
    uint32 jp_count = std::min(uint32(5), iterations_);
    auto jp_data = ReadbackBuffer(jacobi_staging_buffer_->GetHandle(),
                                   uint64(jp_count) * sizeof(JacobiParams));

    LogInfo("===== PD DEBUG DUMP (first frame) =====");

    // Jacobi params
    const JacobiParams* jp = reinterpret_cast<const JacobiParams*>(jp_data.data());
    for (uint32 k = 0; k < jp_count; ++k) {
        LogInfo("[PD] jacobi[", k, "] omega=", jp[k].omega,
                " is_first=", jp[k].is_first_step);
    }

    // Sample nodes: 0 (pinned), 1, 64, 2048
    uint32 samples[] = {0, 1, 64, 2048};
    for (auto n : samples) {
        if (n >= node_count_) continue;
        uint32 v = n * 4;
        uint32 d = n * 9;

        LogInfo("[PD] --- node ", n, " ---");
        LogInfo("[PD] x_old    = (", x_old[v], ", ", x_old[v+1], ", ", x_old[v+2], ")");
        LogInfo("[PD] s        = (", s[v], ", ", s[v+1], ", ", s[v+2], ")");
        LogInfo("[PD] rhs      = (", rhs[v], ", ", rhs[v+1], ", ", rhs[v+2], ")");
        LogInfo("[PD] q_curr   = (", q[v], ", ", q[v+1], ", ", q[v+2], ")");
        LogInfo("[PD] q_prev   = (", q_prev_d[v], ", ", q_prev_d[v+1], ", ", q_prev_d[v+2], ")");
        LogInfo("[PD] diag     = (", diag[d], ", ", diag[d+4], ", ", diag[d+8],
                ") off=(", diag[d+1], ", ", diag[d+2], ", ", diag[d+3], ")");
        LogInfo("[PD] d_inv    = (", d_inv[d], ", ", d_inv[d+4], ", ", d_inv[d+8],
                ") off=(", d_inv[d+1], ", ", d_inv[d+2], ", ", d_inv[d+3], ")");

        // Compute expected values for sanity check
        // Diff: q - x_old (displacement due to solve)
        float32 dx = q[v] - x_old[v];
        float32 dy = q[v+1] - x_old[v+1];
        float32 dz = q[v+2] - x_old[v+2];
        LogInfo("[PD] q-x_old  = (", dx, ", ", dy, ", ", dz, ")");

        // Diff: q - s (correction from constraints)
        float32 cx = q[v] - s[v];
        float32 cy = q[v+1] - s[v+1];
        float32 cz = q[v+2] - s[v+2];
        LogInfo("[PD] q-s      = (", cx, ", ", cy, ", ", cz, ")");
    }

    // CSR check: first few off-diagonal values
    uint64 csr_val_sz = std::min(uint64(nnz_) * 9 * sizeof(float32), uint64(90 * sizeof(float32)));
    if (csr_val_sz > 0) {
        auto csr = ReadbackBuffer(csr_values_buffer_->GetHandle(), csr_val_sz);
        LogInfo("[PD] CSR first 10 blocks (diag entries only):");
        uint32 blocks = std::min(uint32(10), nnz_);
        for (uint32 b = 0; b < blocks; ++b) {
            LogInfo("[PD]   csr[", b, "] = (", csr[b*9], ", ", csr[b*9+4], ", ", csr[b*9+8], ")");
        }
    }

    LogInfo("===== END PD DEBUG DUMP =====");
}

bool PDDynamics::CalibrateRho() {
    if (rho_calibrated_) return false;

    auto& gpu = GPUCore::GetInstance();
    uint64 vec_sz = uint64(node_count_) * 4 * sizeof(float32);
    uint64 rhs_sz = uint64(node_count_) * 4 * sizeof(uint32);

    // Use a subset of iterations for calibration (pure Jacobi + readback)
    const uint32 cal_iters = std::min(iterations_, uint32(15));

    LogInfo("PDDynamics: calibrating rho with ", cal_iters, " pure Jacobi iterations...");

    // Setup: init + predict + q_curr=s + q_prev=s
    {
        WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &ed);

        Dispatch(enc, pd_init_pipeline_, bg_init_, node_wg_count_);
        Dispatch(enc, pd_predict_pipeline_, bg_predict_, node_wg_count_);
        Dispatch(enc, pd_copy_pipeline_, bg_copy_q_from_s_, node_wg_count_);
        wgpuCommandEncoderCopyBufferToBuffer(enc,
            q_curr_buffer_->GetHandle(), 0,
            q_prev_buffer_->GetHandle(), 0, vec_sz);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }
    simulate::WaitForGPU();

    // Save initial q_0 for delta measurement
    auto q_prev_data = ReadbackBuffer(q_curr_buffer_->GetHandle(), vec_sz);

    // Write pure-Jacobi params (ω=1, is_first=1)
    JacobiParams pure_jacobi = {1.0f, 1, 0.0f, 0.0f};

    std::vector<float32> delta_norms;

    for (uint32 k = 0; k < cal_iters; ++k) {
        // One PD iteration with ω=1
        {
            WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &ed);

            wgpuCommandEncoderClearBuffer(enc, rhs_buffer_->GetHandle(), 0, rhs_sz);
            Dispatch(enc, pd_mass_rhs_pipeline_, bg_mass_rhs_, node_wg_count_);

            for (auto& term : terms_) {
                term->ProjectRHS(enc);
            }

            // Set Jacobi params to ω=1
            jacobi_staging_buffer_->WriteData(std::span<const JacobiParams>(&pure_jacobi, 1));
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                jacobi_staging_buffer_->GetHandle(), 0,
                jacobi_params_buffer_->GetHandle(), 0, sizeof(JacobiParams));

            Dispatch(enc, pd_jacobi_step_pipeline_, bg_jacobi_step_, node_wg_count_);

            // Rotate: prev ← curr, curr ← new
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                q_curr_buffer_->GetHandle(), 0,
                q_prev_buffer_->GetHandle(), 0, vec_sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc,
                q_new_buffer_->GetHandle(), 0,
                q_curr_buffer_->GetHandle(), 0, vec_sz);

            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }
        simulate::WaitForGPU();

        // Readback current q
        auto q_curr_data = ReadbackBuffer(q_curr_buffer_->GetHandle(), vec_sz);

        // Compute ||q_curr - q_prev||
        float64 delta_sq = 0.0;
        for (uint32 i = 0; i < node_count_; ++i) {
            for (uint32 c = 0; c < 3; ++c) {
                float64 d = float64(q_curr_data[i * 4 + c]) - float64(q_prev_data[i * 4 + c]);
                delta_sq += d * d;
            }
        }
        delta_norms.push_back(float32(std::sqrt(delta_sq)));
        q_prev_data = std::move(q_curr_data);
    }

    // Compute convergence ratios (skip first 2 for warmup)
    std::vector<float32> ratios;
    for (uint32 k = 2; k < uint32(delta_norms.size()); ++k) {
        if (delta_norms[k - 1] > 1e-12f) {
            ratios.push_back(delta_norms[k] / delta_norms[k - 1]);
        }
    }

    float32 rho_est = 0.95f;  // Fallback
    if (!ratios.empty()) {
        // Sort and use 75th percentile (conservative but robust)
        std::sort(ratios.begin(), ratios.end());
        uint32 idx = std::min(uint32(float32(ratios.size()) * 0.75f),
                              uint32(ratios.size() - 1));
        rho_est = ratios[idx];

        // Safety margin: slightly overestimate (never underestimate!)
        // Underestimating ρ causes Chebyshev to amplify modes above ρ_est.
        rho_est = rho_est * 1.05f;
        rho_est = std::clamp(rho_est, 0.5f, 0.9999f);
    }

    // Log calibration results
    float32 sq = std::sqrt(1.0f - rho_est * rho_est);
    float32 sigma = (1.0f - sq) / rho_est;
    float32 log_sigma = std::log(sigma);
    uint32 iters_1pct = (log_sigma != 0.0f)
        ? static_cast<uint32>(std::ceil(std::log(0.01f) / log_sigma))
        : 99999;

    LogInfo("PDDynamics: calibrated rho=", rho_est,
            " (sigma=", sigma, ", iters_for_1%=", iters_1pct,
            ", configured=", iterations_, ")");

    for (uint32 k = 0; k < uint32(delta_norms.size()); ++k) {
        if (k > 0 && delta_norms[k - 1] > 1e-12f) {
            LogInfo("  iter ", k, ": ||delta||=", delta_norms[k],
                    " ratio=", delta_norms[k] / delta_norms[k - 1]);
        } else {
            LogInfo("  iter ", k, ": ||delta||=", delta_norms[k]);
        }
    }

    BuildChebyshevParams(rho_est);
    rho_calibrated_ = true;
    return true;
}

void PDDynamics::BuildChebyshevParams(float32 rho) {
    std::vector<JacobiParams> all_params(iterations_);
    float32 omega = 1.0f;
    for (uint32 k = 0; k < iterations_; ++k) {
        if (k == 0) {
            omega = 1.0f;
            all_params[k] = {omega, 1, 0.0f, 0.0f};
        } else if (k == 1) {
            omega = 2.0f / (2.0f - rho * rho);
            all_params[k] = {omega, 0, 0.0f, 0.0f};
        } else {
            omega = 4.0f / (4.0f - rho * rho * omega);
            all_params[k] = {omega, 0, 0.0f, 0.0f};
        }
    }
    jacobi_staging_buffer_->WriteData(std::span<const JacobiParams>(all_params));
}

void PDDynamics::Shutdown() {
    for (auto& term : terms_) {
        term->Shutdown();
    }
    terms_.clear();

    bg_init_ = {};
    bg_predict_ = {};
    bg_copy_q_from_s_ = {};
    bg_mass_rhs_ = {};
    bg_inertial_lhs_ = {};
    bg_compute_d_inv_ = {};
    bg_jacobi_step_ = {};

    pd_init_pipeline_ = {};
    pd_predict_pipeline_ = {};
    pd_copy_pipeline_ = {};
    pd_mass_rhs_pipeline_ = {};
    pd_inertial_lhs_pipeline_ = {};
    pd_compute_d_inv_pipeline_ = {};
    pd_jacobi_step_pipeline_ = {};

    params_buffer_.reset();
    jacobi_params_buffer_.reset();
    jacobi_staging_buffer_.reset();
    csr_row_ptr_buffer_.reset();
    csr_col_idx_buffer_.reset();
    csr_values_buffer_.reset();
    diag_buffer_.reset();
    d_inv_buffer_.reset();
    x_old_buffer_.reset();
    s_buffer_.reset();
    q_curr_buffer_.reset();
    q_prev_buffer_.reset();
    q_new_buffer_.reset();
    rhs_buffer_.reset();
    sparsity_.reset();

    LogInfo("PDDynamics: shutdown");
}

}  // namespace ext_pd
