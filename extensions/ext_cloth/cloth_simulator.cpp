#include "ext_cloth/cloth_simulator.h"
#include "ext_cloth/cloth_components.h"
#include "core_system/system.h"
#include "core_database/database.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_gpu/compute_pipeline_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <span>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::database;

namespace ext_cloth {

const std::string ClothSimulator::kName = "ClothSimulator";

// ============================================================================
// Helper: create bind group from pipeline-derived layout
// ============================================================================

static WGPUBindGroup MakeBindGroup(WGPUComputePipeline pipeline,
                                    const std::string& label,
                                    std::initializer_list<std::pair<uint32, std::pair<WGPUBuffer, uint64>>> entries) {
    auto bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    auto builder = BindGroupBuilder(label);
    for (auto& [binding, buf_size] : entries) {
        builder = std::move(builder).AddBuffer(binding, buf_size.first, buf_size.second);
    }
    auto bg = std::move(builder).Build(bgl);
    wgpuBindGroupLayoutRelease(bgl);
    return bg;
}

// ============================================================================
// Constructor
// ============================================================================

ClothSimulator::ClothSimulator(system::System& system)
    : system_(system) {}

const std::string& ClothSimulator::GetName() const {
    return kName;
}

// ============================================================================
// Initialize
// ============================================================================

void ClothSimulator::Initialize(Database& db) {
    CreateMesh(db);
    BuildCSRSparsity();
    CreateGPUBuffers();
    CreateComputePipelines();
    initialized_ = true;
    LogInfo("ClothSimulator: initialized (", mesh_data_.positions.size(),
            " nodes, ", mesh_data_.edges.size(), " edges, ",
            mesh_data_.faces.size(), " faces, nnz=", nnz_, ")");
}

void ClothSimulator::CreateMesh(Database& db) {
    mesh_data_ = GenerateGrid(32, 32, 0.1f, 50000.0f, 3.0f);

    for (uint32 i = 0; i < static_cast<uint32>(mesh_data_.positions.size()); ++i) {
        Entity entity = db.CreateEntity();
        db.AddComponent<ClothPosition>(entity, mesh_data_.positions[i]);
        db.AddComponent<ClothVelocity>(entity, mesh_data_.velocities[i]);
        db.AddComponent<ClothMass>(entity, mesh_data_.masses[i]);
    }
}

void ClothSimulator::BuildCSRSparsity() {
    uint32 N = static_cast<uint32>(mesh_data_.positions.size());
    uint32 E = static_cast<uint32>(mesh_data_.edges.size());

    // Adjacency per node (sorted neighbors)
    std::vector<std::set<uint32>> adjacency(N);
    for (uint32 e = 0; e < E; ++e) {
        uint32 a = mesh_data_.edges[e].n0;
        uint32 b = mesh_data_.edges[e].n1;
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }

    // Build CSR (off-diagonal)
    csr_row_ptr_.resize(N + 1, 0);
    csr_col_idx_.clear();
    for (uint32 i = 0; i < N; ++i) {
        csr_row_ptr_[i] = static_cast<uint32>(csr_col_idx_.size());
        for (uint32 j : adjacency[i]) {
            csr_col_idx_.push_back(j);
        }
    }
    csr_row_ptr_[N] = static_cast<uint32>(csr_col_idx_.size());
    nnz_ = static_cast<uint32>(csr_col_idx_.size());

    // (row, col) → CSR index lookup
    std::map<std::pair<uint32, uint32>, uint32> csr_lookup;
    for (uint32 i = 0; i < N; ++i) {
        for (uint32 idx = csr_row_ptr_[i]; idx < csr_row_ptr_[i + 1]; ++idx) {
            csr_lookup[{i, csr_col_idx_[idx]}] = idx;
        }
    }

    // Edge → CSR mapping
    edge_csr_mappings_.resize(E);
    for (uint32 e = 0; e < E; ++e) {
        uint32 a = mesh_data_.edges[e].n0;
        uint32 b = mesh_data_.edges[e].n1;
        edge_csr_mappings_[e].block_ab = csr_lookup[{a, b}];
        edge_csr_mappings_[e].block_ba = csr_lookup[{b, a}];
        edge_csr_mappings_[e].block_aa = a;
        edge_csr_mappings_[e].block_bb = b;
    }
}

void ClothSimulator::CreateGPUBuffers() {
    uint32 N = static_cast<uint32>(mesh_data_.positions.size());
    uint32 E = static_cast<uint32>(mesh_data_.edges.size());
    uint32 F = static_cast<uint32>(mesh_data_.faces.size());
    auto srw = BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc;

    // Topology
    edge_buffer_ = std::make_unique<GPUBuffer<ClothEdge>>(
        BufferUsage::Storage, std::span<const ClothEdge>(mesh_data_.edges), "cloth_edges");
    face_buffer_ = std::make_unique<GPUBuffer<ClothFace>>(
        BufferUsage::Storage, std::span<const ClothFace>(mesh_data_.faces), "cloth_faces");
    edge_csr_buffer_ = std::make_unique<GPUBuffer<EdgeCSRMapping>>(
        BufferUsage::Storage, std::span<const EdgeCSRMapping>(edge_csr_mappings_), "cloth_edge_csr");

    // Face indices for rendering
    std::vector<uint32> face_idx;
    face_idx.reserve(F * 3);
    for (const auto& f : mesh_data_.faces) {
        face_idx.push_back(f.n0);
        face_idx.push_back(f.n1);
        face_idx.push_back(f.n2);
    }
    face_index_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Index | BufferUsage::Storage,
        std::span<const uint32>(face_idx), "cloth_face_idx");

    // CSR structure
    csr_row_ptr_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage, std::span<const uint32>(csr_row_ptr_), "csr_row_ptr");
    csr_col_idx_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Storage, std::span<const uint32>(csr_col_idx_), "csr_col_idx");
    csr_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(nnz_) * 9 * sizeof(float32), .label = "csr_values"});
    diag_values_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(N) * 9 * sizeof(float32), .label = "diag_values"});

    // Forces (i32 atomic, N*4)
    force_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(N) * 4 * sizeof(int32), .label = "forces"});

    // Normals: i32 atomic for scatter, f32 for renderer output
    normal_atomic_buffer_ = std::make_unique<GPUBuffer<int32>>(
        BufferConfig{.usage = srw, .size = uint64(N) * 4 * sizeof(int32), .label = "normals_atomic"});
    normal_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw | BufferUsage::Vertex, .size = uint64(N) * 4 * sizeof(float32), .label = "normals"});

    // Newton solver buffers (N*4 floats each)
    uint64 vec_sz = uint64(N) * 4 * sizeof(float32);
    x_old_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "x_old"});
    dv_total_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "dv_total"});

    // CG vectors (N*4 floats each)
    cg_x_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_x"});
    cg_r_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_r"});
    cg_p_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_p"});
    cg_ap_buffer_ = std::make_unique<GPUBuffer<float32>>(BufferConfig{.usage = srw, .size = vec_sz, .label = "cg_ap"});

    workgroup_count_ = (N + kWorkgroupSize - 1) / kWorkgroupSize;
    dot_partial_count_ = workgroup_count_;
    cg_partial_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = uint64(dot_partial_count_) * sizeof(float32), .label = "cg_partials"});
    cg_scalar_buffer_ = std::make_unique<GPUBuffer<float32>>(
        BufferConfig{.usage = srw, .size = 8 * sizeof(float32), .label = "cg_scalars"});

    // Params uniform
    ClothSimParams p{};
    p.node_count = N;
    p.edge_count = E;
    p.face_count = F;
    params_buffer_ = std::make_unique<GPUBuffer<ClothSimParams>>(
        BufferUsage::Uniform, std::span<const ClothSimParams>(&p, 1), "cloth_params");

    // CG constant uniforms (dot product target slots + scalar mode)
    DotConfig dc_rr{0, dot_partial_count_, 0, 0};
    DotConfig dc_pap{1, dot_partial_count_, 0, 0};
    DotConfig dc_rr_new{2, dot_partial_count_, 0, 0};
    dc_rr_buf_ = std::make_unique<GPUBuffer<DotConfig>>(
        BufferUsage::Uniform, std::span<const DotConfig>(&dc_rr, 1), "dc_rr");
    dc_pap_buf_ = std::make_unique<GPUBuffer<DotConfig>>(
        BufferUsage::Uniform, std::span<const DotConfig>(&dc_pap, 1), "dc_pap");
    dc_rr_new_buf_ = std::make_unique<GPUBuffer<DotConfig>>(
        BufferUsage::Uniform, std::span<const DotConfig>(&dc_rr_new, 1), "dc_rr_new");

    ScalarMode mode_alpha{0, 0, 0, 0};
    ScalarMode mode_beta{1, 0, 0, 0};
    mode_alpha_buf_ = std::make_unique<GPUBuffer<ScalarMode>>(
        BufferUsage::Uniform, std::span<const ScalarMode>(&mode_alpha, 1), "cg_mode_alpha");
    mode_beta_buf_ = std::make_unique<GPUBuffer<ScalarMode>>(
        BufferUsage::Uniform, std::span<const ScalarMode>(&mode_beta, 1), "cg_mode_beta");
}

void ClothSimulator::CreateComputePipelines() {
    // Helper to create a pipeline from a shader file
    auto make_pipeline = [](const std::string& shader_path, const std::string& label,
                             WGPUComputePipeline& out) {
        auto shader = ShaderLoader::CreateModule("ext_cloth/" + shader_path, label);
        // Use auto layout (null pipeline layout → Dawn infers from shader)
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;  // auto layout
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        out = wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc);
    };

    make_pipeline("newton_init.wgsl", "newton_init", newton_init_pipeline_);
    make_pipeline("newton_predict_pos.wgsl", "newton_predict_pos", newton_predict_pos_pipeline_);
    make_pipeline("newton_accumulate_dv.wgsl", "newton_accumulate_dv", newton_accumulate_dv_pipeline_);
    make_pipeline("clear_forces.wgsl", "clear_forces", clear_forces_pipeline_);
    make_pipeline("accumulate_gravity.wgsl", "accumulate_gravity", accumulate_gravity_pipeline_);
    make_pipeline("accumulate_springs.wgsl", "accumulate_springs", accumulate_springs_pipeline_);
    make_pipeline("assemble_rhs.wgsl", "assemble_rhs", assemble_rhs_pipeline_);
    make_pipeline("cg_init.wgsl", "cg_init", cg_init_pipeline_);
    make_pipeline("cg_spmv.wgsl", "cg_spmv", cg_spmv_pipeline_);
    make_pipeline("cg_dot.wgsl", "cg_dot", cg_dot_pipeline_);
    make_pipeline("cg_dot_final.wgsl", "cg_dot_final", cg_dot_final_pipeline_);
    make_pipeline("cg_compute_scalars.wgsl", "cg_compute_scalars", cg_compute_scalars_pipeline_);
    make_pipeline("cg_update_xr.wgsl", "cg_update_xr", cg_update_xr_pipeline_);
    make_pipeline("cg_update_p.wgsl", "cg_update_p", cg_update_p_pipeline_);
    make_pipeline("update_velocity.wgsl", "update_velocity", update_velocity_pipeline_);
    make_pipeline("update_position.wgsl", "update_position", update_position_pipeline_);
    make_pipeline("clear_normals.wgsl", "clear_normals", clear_normals_pipeline_);
    make_pipeline("compute_normals_scatter.wgsl", "scatter_normals", scatter_normals_pipeline_);
    make_pipeline("compute_normals_normalize.wgsl", "normalize_normals", normalize_normals_pipeline_);

    LogInfo("ClothSimulator: 19 compute pipelines created (auto layout)");
}

// ============================================================================
// Update (per frame)
// ============================================================================

void ClothSimulator::Update(Database& db, float32 dt) {
    if (!initialized_) return;

    uint32 N = static_cast<uint32>(mesh_data_.positions.size());
    uint32 E = static_cast<uint32>(mesh_data_.edges.size());
    uint32 F = static_cast<uint32>(mesh_data_.faces.size());

    dt = std::min(dt, 1.0f / 30.0f);

    constexpr uint32 newton_iters = 3;
    constexpr uint32 cg_iters = 10;

    // Update params
    ClothSimParams p{};
    p.dt = dt;
    p.gravity_y = -9.81f;
    p.node_count = N;
    p.edge_count = E;
    p.face_count = F;
    p.cg_max_iter = cg_iters;
    p.damping = 0.999f;
    p.cg_tolerance = 1e-6f;
    params_buffer_->WriteData(std::span<const ClothSimParams>(&p, 1));

    auto& gpu = GPUCore::GetInstance();
    uint32 node_wg = (N + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32 edge_wg = (E + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32 face_wg = (F + kWorkgroupSize - 1) / kWorkgroupSize;

    // Buffer handles
    WGPUBuffer pos_h = system_.GetDeviceBuffer<ClothPosition>();
    WGPUBuffer vel_h = system_.GetDeviceBuffer<ClothVelocity>();
    WGPUBuffer mass_h = system_.GetDeviceBuffer<ClothMass>();

    // Buffer sizes
    uint64 params_sz = sizeof(ClothSimParams);
    uint64 force_sz = uint64(N) * 4 * sizeof(int32);
    uint64 mass_sz = uint64(N) * sizeof(ClothMass);
    uint64 pos_sz = uint64(N) * sizeof(ClothPosition);
    uint64 vel_sz = uint64(N) * sizeof(ClothVelocity);
    uint64 edge_sz = uint64(E) * sizeof(ClothEdge);
    uint64 csr_val_sz = uint64(nnz_) * 9 * sizeof(float32);
    uint64 diag_sz = uint64(N) * 9 * sizeof(int32);
    uint64 csr_map_sz = uint64(E) * sizeof(EdgeCSRMapping);
    uint64 vec_sz = uint64(N) * 4 * sizeof(float32);
    uint64 partial_sz = uint64(dot_partial_count_) * sizeof(float32);
    uint64 scalar_sz = 8 * sizeof(float32);
    uint64 face_sz = uint64(F) * sizeof(ClothFace);
    uint64 normal_i32_sz = uint64(N) * 4 * sizeof(int32);
    uint64 row_ptr_sz = csr_row_ptr_buffer_->GetByteLength();
    uint64 col_idx_sz = csr_col_idx_buffer_->GetByteLength();
    // Shorthand handles
    WGPUBuffer params_h = params_buffer_->GetHandle();
    WGPUBuffer force_h = force_buffer_->GetHandle();
    WGPUBuffer edge_h = edge_buffer_->GetHandle();
    WGPUBuffer csr_val_h = csr_values_buffer_->GetHandle();
    WGPUBuffer diag_h = diag_values_buffer_->GetHandle();
    WGPUBuffer csr_map_h = edge_csr_buffer_->GetHandle();
    WGPUBuffer row_ptr_h = csr_row_ptr_buffer_->GetHandle();
    WGPUBuffer col_idx_h = csr_col_idx_buffer_->GetHandle();
    WGPUBuffer cg_x_h = cg_x_buffer_->GetHandle();
    WGPUBuffer cg_r_h = cg_r_buffer_->GetHandle();
    WGPUBuffer cg_p_h = cg_p_buffer_->GetHandle();
    WGPUBuffer cg_ap_h = cg_ap_buffer_->GetHandle();
    WGPUBuffer partial_h = cg_partial_buffer_->GetHandle();
    WGPUBuffer scalar_h = cg_scalar_buffer_->GetHandle();
    WGPUBuffer face_h = face_buffer_->GetHandle();
    WGPUBuffer norm_i32_h = normal_atomic_buffer_->GetHandle();
    WGPUBuffer norm_h = normal_buffer_->GetHandle();
    WGPUBuffer x_old_h = x_old_buffer_->GetHandle();
    WGPUBuffer dv_total_h = dv_total_buffer_->GetHandle();

    // ---- Create all bind groups (per-frame, from pipeline-derived layouts) ----

    // newton_init: @0 params, @1 positions(read), @2 x_old(rw), @3 dv_total(rw)
    auto bg_newton_init = MakeBindGroup(newton_init_pipeline_, "bg_newton_init",
        {{0, {params_h, params_sz}}, {1, {pos_h, pos_sz}}, {2, {x_old_h, vec_sz}}, {3, {dv_total_h, vec_sz}}});

    // newton_predict_pos: @0 params, @1 positions(rw), @2 x_old, @3 velocities, @4 dv_total, @5 mass
    auto bg_predict_pos = MakeBindGroup(newton_predict_pos_pipeline_, "bg_predict_pos",
        {{0, {params_h, params_sz}}, {1, {pos_h, pos_sz}}, {2, {x_old_h, vec_sz}},
         {3, {vel_h, vel_sz}}, {4, {dv_total_h, vec_sz}}, {5, {mass_h, mass_sz}}});

    // newton_accumulate_dv: @0 params, @1 dv_total(rw), @2 cg_x
    auto bg_accumulate_dv = MakeBindGroup(newton_accumulate_dv_pipeline_, "bg_accumulate_dv",
        {{0, {params_h, params_sz}}, {1, {dv_total_h, vec_sz}}, {2, {cg_x_h, vec_sz}}});

    // clear_forces: @0 params, @1 forces(rw)
    auto bg_clear_forces = MakeBindGroup(clear_forces_pipeline_, "bg_clear_forces",
        {{0, {params_h, params_sz}}, {1, {force_h, force_sz}}});

    // accumulate_gravity: @0 params, @1 forces(rw), @2 mass
    auto bg_gravity = MakeBindGroup(accumulate_gravity_pipeline_, "bg_gravity",
        {{0, {params_h, params_sz}}, {1, {force_h, force_sz}}, {2, {mass_h, mass_sz}}});

    // accumulate_springs: @0 params, @1 pos, @2 forces, @3 edges, @4 csr_values, @5 diag, @6 edge_csr_map
    auto bg_springs = MakeBindGroup(accumulate_springs_pipeline_, "bg_springs",
        {{0, {params_h, params_sz}}, {1, {pos_h, pos_sz}}, {2, {force_h, force_sz}},
         {3, {edge_h, edge_sz}}, {4, {csr_val_h, csr_val_sz}}, {5, {diag_h, diag_sz}},
         {6, {csr_map_h, csr_map_sz}}});

    // assemble_rhs (simplified): @0 params, @1 forces, @2 dv_total, @3 mass, @4 rhs
    auto bg_rhs = MakeBindGroup(assemble_rhs_pipeline_, "bg_rhs",
        {{0, {params_h, params_sz}}, {1, {force_h, force_sz}}, {2, {dv_total_h, vec_sz}},
         {3, {mass_h, mass_sz}}, {4, {cg_r_h, vec_sz}}});

    // cg_init: @0 params, @1 x(rw), @2 r(read), @3 p(rw)
    auto bg_cg_init = MakeBindGroup(cg_init_pipeline_, "bg_cg_init",
        {{0, {params_h, params_sz}}, {1, {cg_x_h, vec_sz}}, {2, {cg_r_h, vec_sz}}, {3, {cg_p_h, vec_sz}}});

    // cg_spmv: @0 params, @1 p, @2 ap, @3 mass, @4 row_ptr, @5 col_idx, @6 csr_val, @7 diag
    auto bg_spmv = MakeBindGroup(cg_spmv_pipeline_, "bg_spmv",
        {{0, {params_h, params_sz}}, {1, {cg_p_h, vec_sz}}, {2, {cg_ap_h, vec_sz}},
         {3, {mass_h, mass_sz}}, {4, {row_ptr_h, row_ptr_sz}}, {5, {col_idx_h, col_idx_sz}},
         {6, {csr_val_h, csr_val_sz}}, {7, {diag_h, diag_sz}}});

    // cg_dot (r,r): @0 params, @1 vec_a, @2 vec_b, @3 partials
    auto bg_dot_rr = MakeBindGroup(cg_dot_pipeline_, "bg_dot_rr",
        {{0, {params_h, params_sz}}, {1, {cg_r_h, vec_sz}}, {2, {cg_r_h, vec_sz}}, {3, {partial_h, partial_sz}}});

    // cg_dot (p, Ap): same layout
    auto bg_dot_pap = MakeBindGroup(cg_dot_pipeline_, "bg_dot_pap",
        {{0, {params_h, params_sz}}, {1, {cg_p_h, vec_sz}}, {2, {cg_ap_h, vec_sz}}, {3, {partial_h, partial_sz}}});

    // cg_dot_final: @0 partials, @1 scalars, @2 dot_config(uniform)
    auto bg_dot_final_rr = MakeBindGroup(cg_dot_final_pipeline_, "bg_df_rr",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_rr_buf_->GetHandle(), sizeof(DotConfig)}}});
    auto bg_dot_final_pap = MakeBindGroup(cg_dot_final_pipeline_, "bg_df_pap",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_pap_buf_->GetHandle(), sizeof(DotConfig)}}});
    auto bg_dot_final_rr_new = MakeBindGroup(cg_dot_final_pipeline_, "bg_df_rr_new",
        {{0, {partial_h, partial_sz}}, {1, {scalar_h, scalar_sz}}, {2, {dc_rr_new_buf_->GetHandle(), sizeof(DotConfig)}}});

    // cg_compute_scalars: @0 scalars(rw), @1 mode(uniform)
    auto bg_scalars_alpha = MakeBindGroup(cg_compute_scalars_pipeline_, "bg_scalars_alpha",
        {{0, {scalar_h, scalar_sz}}, {1, {mode_alpha_buf_->GetHandle(), sizeof(ScalarMode)}}});
    auto bg_scalars_beta = MakeBindGroup(cg_compute_scalars_pipeline_, "bg_scalars_beta",
        {{0, {scalar_h, scalar_sz}}, {1, {mode_beta_buf_->GetHandle(), sizeof(ScalarMode)}}});

    // cg_update_xr: @0 params, @1 x, @2 r, @3 p, @4 ap, @5 scalars, @6 mass
    auto bg_update_xr = MakeBindGroup(cg_update_xr_pipeline_, "bg_xr",
        {{0, {params_h, params_sz}}, {1, {cg_x_h, vec_sz}}, {2, {cg_r_h, vec_sz}},
         {3, {cg_p_h, vec_sz}}, {4, {cg_ap_h, vec_sz}}, {5, {scalar_h, scalar_sz}},
         {6, {mass_h, mass_sz}}});

    // cg_update_p: @0 params, @1 r, @2 p, @3 scalars, @4 mass
    auto bg_update_p = MakeBindGroup(cg_update_p_pipeline_, "bg_p",
        {{0, {params_h, params_sz}}, {1, {cg_r_h, vec_sz}}, {2, {cg_p_h, vec_sz}}, {3, {scalar_h, scalar_sz}},
         {4, {mass_h, mass_sz}}});

    // update_velocity: @0 params, @1 vel, @2 dv_total, @3 mass
    auto bg_vel = MakeBindGroup(update_velocity_pipeline_, "bg_vel",
        {{0, {params_h, params_sz}}, {1, {vel_h, vel_sz}}, {2, {dv_total_h, vec_sz}}, {3, {mass_h, mass_sz}}});

    // update_position: @0 params, @1 pos, @2 x_old, @3 vel, @4 mass
    auto bg_pos = MakeBindGroup(update_position_pipeline_, "bg_pos",
        {{0, {params_h, params_sz}}, {1, {pos_h, pos_sz}}, {2, {x_old_h, vec_sz}},
         {3, {vel_h, vel_sz}}, {4, {mass_h, mass_sz}}});

    // clear_normals: @0 params, @1 normals_i32(rw atomic)
    auto bg_clear_normals = MakeBindGroup(clear_normals_pipeline_, "bg_clear_n",
        {{0, {params_h, params_sz}}, {1, {norm_i32_h, normal_i32_sz}}});

    // scatter_normals: @0 params, @1 pos, @2 faces, @3 normals_i32(rw atomic)
    auto bg_scatter = MakeBindGroup(scatter_normals_pipeline_, "bg_scatter_n",
        {{0, {params_h, params_sz}}, {1, {pos_h, pos_sz}}, {2, {face_h, face_sz}}, {3, {norm_i32_h, normal_i32_sz}}});

    // normalize_normals: @0 params, @1 normals_i32(read), @2 normals_out(rw)
    auto bg_normalize = MakeBindGroup(normalize_normals_pipeline_, "bg_norm_n",
        {{0, {params_h, params_sz}}, {1, {norm_i32_h, normal_i32_sz}}, {2, {norm_h, vec_sz}}});

    // ---- Dispatch compute passes ----

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"cloth_compute", 13};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    auto dispatch_pass = [&](WGPUComputePipeline pipeline, WGPUBindGroup bg, uint32 wg_count) {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(pipeline);
        enc.SetBindGroup(0, bg);
        enc.Dispatch(wg_count);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    };

    // ---- Newton Init: save x_old, zero dv_total ----
    dispatch_pass(newton_init_pipeline_, bg_newton_init, node_wg);

    // ---- Newton Outer Loop ----
    for (uint32 nit = 0; nit < newton_iters; ++nit) {
        // Predict positions: x_temp = x_old + dt*(v + dv_total)
        dispatch_pass(newton_predict_pos_pipeline_, bg_predict_pos, node_wg);

        // Clear forces
        dispatch_pass(clear_forces_pipeline_, bg_clear_forces, node_wg);

        // Clear diagonal Hessian buffer
        wgpuCommandEncoderClearBuffer(encoder, diag_h, 0, diag_sz);

        // Gravity
        dispatch_pass(accumulate_gravity_pipeline_, bg_gravity, node_wg);

        // Springs (forces + Hessian assembly at predicted positions)
        dispatch_pass(accumulate_springs_pipeline_, bg_springs, edge_wg);

        // Assemble RHS: b = dt*F - M*dv_total
        dispatch_pass(assemble_rhs_pipeline_, bg_rhs, node_wg);

        // Clear scalar buffer
        wgpuCommandEncoderClearBuffer(encoder, scalar_h, 0, scalar_sz);

        // CG Init: x = 0, p = r
        dispatch_pass(cg_init_pipeline_, bg_cg_init, node_wg);

        // Initial rr = dot(r, r) → scalars[0]
        dispatch_pass(cg_dot_pipeline_, bg_dot_rr, node_wg);
        dispatch_pass(cg_dot_final_pipeline_, bg_dot_final_rr, 1);

        // ---- CG Inner Loop (fixed iteration count) ----
        for (uint32 cit = 0; cit < cg_iters; ++cit) {
            // Ap = A * p
            dispatch_pass(cg_spmv_pipeline_, bg_spmv, node_wg);

            // pAp = dot(p, Ap) → scalars[1]
            dispatch_pass(cg_dot_pipeline_, bg_dot_pap, node_wg);
            dispatch_pass(cg_dot_final_pipeline_, bg_dot_final_pap, 1);

            // alpha = rr / pAp → scalars[3]
            dispatch_pass(cg_compute_scalars_pipeline_, bg_scalars_alpha, 1);

            // x += alpha*p, r -= alpha*Ap
            dispatch_pass(cg_update_xr_pipeline_, bg_update_xr, node_wg);

            // rr_new = dot(r, r) → scalars[2]
            dispatch_pass(cg_dot_pipeline_, bg_dot_rr, node_wg);
            dispatch_pass(cg_dot_final_pipeline_, bg_dot_final_rr_new, 1);

            // beta = rr_new / rr, advance rr = rr_new
            dispatch_pass(cg_compute_scalars_pipeline_, bg_scalars_beta, 1);

            // p = r + beta * p
            dispatch_pass(cg_update_p_pipeline_, bg_update_p, node_wg);
        }

        // Accumulate CG solution into Newton delta: dv_total += cg_x
        dispatch_pass(newton_accumulate_dv_pipeline_, bg_accumulate_dv, node_wg);
    }

    // ---- Final velocity and position update ----

    // Update velocity: v = (v + dv_total) * damping
    dispatch_pass(update_velocity_pipeline_, bg_vel, node_wg);

    // Update position: pos = x_old + vel * dt
    dispatch_pass(update_position_pipeline_, bg_pos, node_wg);

    // Compute normals
    dispatch_pass(clear_normals_pipeline_, bg_clear_normals, node_wg);
    dispatch_pass(scatter_normals_pipeline_, bg_scatter, face_wg);
    dispatch_pass(normalize_normals_pipeline_, bg_normalize, node_wg);

    // Submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Release all per-frame bind groups
    wgpuBindGroupRelease(bg_newton_init);
    wgpuBindGroupRelease(bg_predict_pos);
    wgpuBindGroupRelease(bg_accumulate_dv);
    wgpuBindGroupRelease(bg_clear_forces);
    wgpuBindGroupRelease(bg_gravity);
    wgpuBindGroupRelease(bg_springs);
    wgpuBindGroupRelease(bg_rhs);
    wgpuBindGroupRelease(bg_cg_init);
    wgpuBindGroupRelease(bg_spmv);
    wgpuBindGroupRelease(bg_dot_rr);
    wgpuBindGroupRelease(bg_dot_pap);
    wgpuBindGroupRelease(bg_dot_final_rr);
    wgpuBindGroupRelease(bg_dot_final_pap);
    wgpuBindGroupRelease(bg_dot_final_rr_new);
    wgpuBindGroupRelease(bg_scalars_alpha);
    wgpuBindGroupRelease(bg_scalars_beta);
    wgpuBindGroupRelease(bg_update_xr);
    wgpuBindGroupRelease(bg_update_p);
    wgpuBindGroupRelease(bg_vel);
    wgpuBindGroupRelease(bg_pos);
    wgpuBindGroupRelease(bg_clear_normals);
    wgpuBindGroupRelease(bg_scatter);
    wgpuBindGroupRelease(bg_normalize);

    // Readback
    ReadbackPositionsVelocities(db);
}

void ClothSimulator::ReadbackPositionsVelocities(Database& db) {
    WGPUBuffer pos_buf = system_.GetDeviceBuffer<ClothPosition>();
    WGPUBuffer vel_buf = system_.GetDeviceBuffer<ClothVelocity>();
    if (!pos_buf || !vel_buf) return;

    uint32 N = static_cast<uint32>(mesh_data_.positions.size());
    auto& gpu_core = GPUCore::GetInstance();

    uint64 pos_size = uint64(N) * sizeof(ClothPosition);
    uint64 vel_size = uint64(N) * sizeof(ClothVelocity);

    // Create staging buffers
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;

    staging_desc.size = pos_size;
    WGPUBuffer pos_staging = wgpuDeviceCreateBuffer(gpu_core.GetDevice(), &staging_desc);
    staging_desc.size = vel_size;
    WGPUBuffer vel_staging = wgpuDeviceCreateBuffer(gpu_core.GetDevice(), &staging_desc);

    // Copy
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu_core.GetDevice(), &enc_desc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, pos_buf, 0, pos_staging, 0, pos_size);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, vel_buf, 0, vel_staging, 0, vel_size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu_core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Map helper
    auto map_sync = [&](WGPUBuffer buffer, uint64 size) -> std::vector<uint8> {
        struct Ctx { bool done = false; bool ok = false; };
        Ctx ctx;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
        cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
        cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
            auto* c = static_cast<Ctx*>(ud1); c->done = true;
            c->ok = (status == WGPUMapAsyncStatus_Success);
        };
        cb.userdata1 = &ctx;
        WGPUFuture future = wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, static_cast<size_t>(size), cb);
#ifndef __EMSCRIPTEN__
        WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
        wait.future = future;
        wgpuInstanceWaitAny(gpu_core.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
        while (!ctx.done) gpu_core.ProcessEvents();
#endif
        std::vector<uint8> result;
        if (ctx.ok) {
            auto* mapped = wgpuBufferGetConstMappedRange(buffer, 0, static_cast<size_t>(size));
            result.resize(static_cast<size_t>(size));
            std::memcpy(result.data(), mapped, result.size());
            wgpuBufferUnmap(buffer);
        }
        return result;
    };

    auto pos_data = map_sync(pos_staging, pos_size);
    auto vel_data = map_sync(vel_staging, vel_size);
    wgpuBufferRelease(pos_staging);
    wgpuBufferRelease(vel_staging);

    if (pos_data.empty() || vel_data.empty()) return;

    auto* pos_storage = db.GetStorageById(GetComponentTypeId<ClothPosition>());
    if (!pos_storage) return;
    auto* typed_pos = static_cast<ComponentStorage<ClothPosition>*>(pos_storage);
    const auto& entities = typed_pos->GetEntities();

    auto* positions = reinterpret_cast<const ClothPosition*>(pos_data.data());
    auto* velocities = reinterpret_cast<const ClothVelocity*>(vel_data.data());

    for (uint32 i = 0; i < N && i < static_cast<uint32>(entities.size()); ++i) {
        db.SetComponent<ClothPosition>(entities[i], positions[i]);
        db.SetComponent<ClothVelocity>(entities[i], velocities[i]);
    }
}

// ============================================================================
// Accessors
// ============================================================================

WGPUBuffer ClothSimulator::GetNormalBuffer() const {
    return normal_buffer_ ? normal_buffer_->GetHandle() : nullptr;
}

WGPUBuffer ClothSimulator::GetIndexBuffer() const {
    return face_index_buffer_ ? face_index_buffer_->GetHandle() : nullptr;
}

uint32 ClothSimulator::GetFaceCount() const {
    return static_cast<uint32>(mesh_data_.faces.size());
}

uint32 ClothSimulator::GetNodeCount() const {
    return static_cast<uint32>(mesh_data_.positions.size());
}

// ============================================================================
// Shutdown
// ============================================================================

void ClothSimulator::Shutdown() {
    auto rel = [](WGPUComputePipeline& p) { if (p) { wgpuComputePipelineRelease(p); p = nullptr; } };
    rel(newton_init_pipeline_);
    rel(newton_predict_pos_pipeline_);
    rel(newton_accumulate_dv_pipeline_);
    rel(clear_forces_pipeline_);
    rel(accumulate_gravity_pipeline_);
    rel(accumulate_springs_pipeline_);
    rel(assemble_rhs_pipeline_);
    rel(cg_init_pipeline_);
    rel(cg_spmv_pipeline_);
    rel(cg_dot_pipeline_);
    rel(cg_dot_final_pipeline_);
    rel(cg_compute_scalars_pipeline_);
    rel(cg_update_xr_pipeline_);
    rel(cg_update_p_pipeline_);
    rel(update_velocity_pipeline_);
    rel(update_position_pipeline_);
    rel(clear_normals_pipeline_);
    rel(scatter_normals_pipeline_);
    rel(normalize_normals_pipeline_);
    initialized_ = false;
    LogInfo("ClothSimulator: shutdown");
}

}  // namespace ext_cloth
