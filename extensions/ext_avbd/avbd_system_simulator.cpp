#include "ext_avbd/avbd_system_simulator.h"
#include "ext_avbd/avbd_system_config.h"
#include "ext_avbd/avbd_spring_term.h"
#include "ext_avbd/avbd_area_term.h"
#include "ext_dynamics/spring_types.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_types.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_mesh/mesh_types.h"
#include "ext_mesh/mesh_component.h"
#include "ext_dynamics/global_physics_params.h"
#include "core_simulate/simulate_config.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_database/database.h"
#include "core_database/component_storage.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/gpu_buffer.h"
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
using namespace mps::database;

namespace ext_avbd {

const std::string AVBDSystemSimulator::kName = "AVBDSystemSimulator";

// ============================================================================
// Static helpers
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

// ============================================================================
// Constructor / Destructor
// ============================================================================

AVBDSystemSimulator::AVBDSystemSimulator(system::System& system)
    : system_(system) {}

AVBDSystemSimulator::~AVBDSystemSimulator() = default;

const std::string& AVBDSystemSimulator::GetName() const {
    return kName;
}

// ============================================================================
// Initialize
// ============================================================================

void AVBDSystemSimulator::Initialize() {
    const auto& db = system_.GetDatabase();

    // Find AVBDSystemConfig entities
    auto config_type_id = GetComponentTypeId<AVBDSystemConfig>();
    const auto* config_storage_base = db.GetStorageById(config_type_id);
    if (!config_storage_base || config_storage_base->GetDenseCount() == 0) {
        LogInfo("AVBDSystemSimulator: no AVBDSystemConfig entities found, skipping");
        return;
    }

    const auto* config_storage =
        static_cast<const ComponentStorage<AVBDSystemConfig>*>(config_storage_base);
    const auto& config_entities = config_storage->GetEntities();

    Entity config_entity = config_entities[0];
    const auto* config = db.GetComponent<AVBDSystemConfig>(config_entity);
    if (!config) return;

    if (config->mesh_entity == database::kInvalidEntity) {
        LogError("AVBDSystemSimulator: mesh_entity not set in config");
        return;
    }

    // Get mesh metadata
    const auto* mesh_comp = db.GetComponent<ext_mesh::MeshComponent>(config->mesh_entity);
    if (!mesh_comp) {
        LogError("AVBDSystemSimulator: no MeshComponent on mesh entity");
        return;
    }

    node_count_ = mesh_comp->vertex_count;
    edge_count_ = mesh_comp->edge_count;
    face_count_ = mesh_comp->face_count;

    if (node_count_ == 0 || edge_count_ == 0) {
        LogError("AVBDSystemSimulator: mesh has 0 nodes or 0 edges");
        return;
    }

    auto& gpu = GPUCore::GetInstance();

    // -------------------------------------------------------------------
    // Scoped mode: create local buffers sized to this mesh
    // -------------------------------------------------------------------
    auto* pos_entry = system_.GetArrayEntryById(GetComponentTypeId<SimPosition>());
    if (!pos_entry) {
        LogError("AVBDSystemSimulator: no SimPosition array entry");
        return;
    }
    node_offset_ = pos_entry->GetEntityOffset(config->mesh_entity);
    if (node_offset_ == UINT32_MAX) {
        LogError("AVBDSystemSimulator: mesh entity not in SimPosition");
        return;
    }

    auto create_buf = [&](uint64 size) -> WGPUBuffer {
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        bd.size = size;
        return wgpuDeviceCreateBuffer(gpu.GetDevice(), &bd);
    };

    uint64 pos_bytes = uint64(node_count_) * sizeof(SimPosition);
    uint64 vel_bytes = uint64(node_count_) * sizeof(SimVelocity);
    uint64 mass_bytes = uint64(node_count_) * sizeof(SimMass);

    local_pos_ = create_buf(pos_bytes);
    local_vel_ = create_buf(vel_bytes);
    local_mass_ = create_buf(mass_bytes);

    // Cache global handles for copy-in/copy-out
    global_pos_ = system_.GetDeviceBuffer<SimPosition>();
    global_vel_ = system_.GetDeviceBuffer<SimVelocity>();
    WGPUBuffer global_mass = system_.GetDeviceBuffer<SimMass>();

    // Copy mass once (immediate) -- mass doesn't change at runtime
    uint64 mass_offset = uint64(node_offset_) * sizeof(SimMass);
    WGPUCommandEncoderDescriptor me_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder me = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &me_desc);
    wgpuCommandEncoderCopyBufferToBuffer(me, global_mass, mass_offset, local_mass_, 0, mass_bytes);
    WGPUCommandBuffer mc = wgpuCommandEncoderFinish(me, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &mc);
    wgpuCommandBufferRelease(mc);
    wgpuCommandEncoderRelease(me);

    scoped_ = true;

    // -------------------------------------------------------------------
    // Graph coloring: read MeshEdge from host DB (local 0-based indices)
    // -------------------------------------------------------------------
    const auto* mesh_edges_ptr = db.GetArray<ext_mesh::MeshEdge>(config->mesh_entity);
    if (!mesh_edges_ptr || mesh_edges_ptr->empty()) {
        LogError("AVBDSystemSimulator: no MeshEdge data on mesh entity");
        return;
    }

    local_edge_buf_ = std::make_unique<GPUBuffer<ext_mesh::MeshEdge>>(
        BufferUsage::Storage,
        std::span<const ext_mesh::MeshEdge>(*mesh_edges_ptr),
        "avbd_local_edges");

    uint64 edge_buffer_size = static_cast<uint64>(edge_count_) * sizeof(ext_mesh::MeshEdge);

    // Run GPU graph coloring
    coloring_.Build(node_count_, edge_count_,
                    local_edge_buf_->GetHandle(), edge_buffer_size);

    // Build color groups (CPU readback + counting sort)
    coloring_.BuildColorGroups();

    const auto& color_offsets = coloring_.GetColorOffsets();
    WGPUBuffer vertex_order_buf = coloring_.GetVertexOrderBuffer();
    uint64 vertex_order_sz = static_cast<uint64>(node_count_) * 4;

    // -------------------------------------------------------------------
    // Get physics buffer from DeviceDB singleton
    // -------------------------------------------------------------------
    WGPUBuffer physics_buf = system_.GetDeviceDB().GetSingletonBuffer<GlobalPhysicsParams>();
    uint64 physics_sz = sizeof(PhysicsParamsGPU);

    // -------------------------------------------------------------------
    // Create VBDDynamics and initialize
    // -------------------------------------------------------------------
    dynamics_ = std::make_unique<VBDDynamics>();

    dynamics_->Initialize(
        node_count_, edge_count_, face_count_,
        config->avbd_iterations,
        physics_buf, physics_sz,
        local_pos_, local_vel_, local_mass_,
        pos_bytes, vel_bytes, mass_bytes,
        color_offsets,
        vertex_order_buf, vertex_order_sz);

    // -------------------------------------------------------------------
    // Discover constraints and create terms
    // -------------------------------------------------------------------
    for (uint32 ci = 0; ci < config->constraint_count; ++ci) {
        Entity ce = config->constraint_entities[ci];
        if (ce == database::kInvalidEntity) continue;

        // Spring constraints
        const auto* spring_data = db.GetComponent<ext_dynamics::SpringConstraintData>(ce);
        if (spring_data) {
            const auto* springs = db.GetArray<ext_dynamics::SpringEdge>(ce);
            if (springs && !springs->empty()) {
                // Build CSR adjacency with edge_index for AL dual variables
                std::vector<uint32> spring_offsets(node_count_ + 1, 0);
                std::vector<uint32> degree(node_count_, 0);
                for (const auto& e : *springs) {
                    degree[e.n0]++;
                    degree[e.n1]++;
                }
                for (uint32 i = 0; i < node_count_; ++i) {
                    spring_offsets[i + 1] = spring_offsets[i] + degree[i];
                }
                std::vector<SpringNeighbor> spring_neighbors(spring_offsets[node_count_]);
                std::vector<uint32> cursor(node_count_, 0);
                uint32 edge_idx = 0;
                for (const auto& e : *springs) {
                    uint32 s0 = spring_offsets[e.n0] + cursor[e.n0]++;
                    spring_neighbors[s0] = {e.n1, e.rest_length, edge_idx};
                    uint32 s1 = spring_offsets[e.n1] + cursor[e.n1]++;
                    spring_neighbors[s1] = {e.n0, e.rest_length, edge_idx};
                    edge_idx++;
                }

                auto spring_term = std::make_unique<AVBDSpringTerm>();
                spring_term->SetSpringData(spring_offsets, spring_neighbors,
                                           *springs, spring_data->stiffness,
                                           config->al_gamma, config->al_beta);
                dynamics_->AddTerm(std::move(spring_term));

                LogInfo("AVBDSystemSimulator: created spring term (", springs->size(),
                        " edges, k=", spring_data->stiffness, ")");
            }
        }

        // Area constraints
        const auto* area_data = db.GetComponent<ext_dynamics::AreaConstraintData>(ce);
        if (area_data) {
            const auto* triangles = db.GetArray<ext_dynamics::AreaTriangle>(ce);
            if (triangles && !triangles->empty()) {
                auto area_term = std::make_unique<AVBDAreaTerm>();
                area_term->SetAreaData(*triangles, node_count_,
                                       area_data->stretch_stiffness,
                                       area_data->shear_stiffness);
                dynamics_->AddTerm(std::move(area_term));

                LogInfo("AVBDSystemSimulator: created area term (", triangles->size(),
                        " triangles, k=", area_data->stretch_stiffness,
                        ", mu=", area_data->shear_stiffness, ")");
            }
        }
    }

    // -------------------------------------------------------------------
    // Create velocity/position update pipelines + bind groups
    // -------------------------------------------------------------------
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

    update_velocity_pipeline_ = make_pipeline("avbd_update_velocity.wgsl", "avbd_update_velocity");
    update_position_pipeline_ = make_pipeline("avbd_update_position.wgsl", "avbd_update_position");

    // update_velocity bindings: 0=physics, 1=solver_params, 2=vel(rw), 3=q(read), 4=x_old(read), 5=mass(read)
    uint64 solver_params_sz = dynamics_->GetSolverParamsSize();
    WGPUBuffer solver_params_buf = dynamics_->GetSolverParamsBuffer();
    WGPUBuffer q_buf = dynamics_->GetQBuffer();
    WGPUBuffer x_old_buf = dynamics_->GetXOldBuffer();

    bg_update_velocity_ = MakeBindGroup(update_velocity_pipeline_, "bg_avbd_update_vel",
        {{0, {physics_buf, physics_sz}},
         {1, {solver_params_buf, solver_params_sz}},
         {2, {local_vel_, vel_bytes}},
         {3, {q_buf, pos_bytes}},
         {4, {x_old_buf, pos_bytes}},
         {5, {local_mass_, mass_bytes}}});

    // update_position bindings: 0=solver_params, 1=pos(rw), 2=q(read), 3=x_old(read), 4=mass(read)
    bg_update_position_ = MakeBindGroup(update_position_pipeline_, "bg_avbd_update_pos",
        {{0, {solver_params_buf, solver_params_sz}},
         {1, {local_pos_, pos_bytes}},
         {2, {q_buf, pos_bytes}},
         {3, {x_old_buf, pos_bytes}},
         {4, {local_mass_, mass_bytes}}});

    topology_sig_ = ComputeTopologySignature();
    initialized_ = true;

    LogInfo("AVBDSystemSimulator: initialized (", node_count_, " nodes, ",
            edge_count_, " edges, ", coloring_.GetColorCount(), " colors, ",
            config->avbd_iterations, " iterations)");
}

// ============================================================================
// Update (per frame)
// ============================================================================

void AVBDSystemSimulator::Update() {
    if (!initialized_ || !dynamics_) return;

    Timer profile_timer;
    if constexpr (kEnableSimulationProfiling) {
        WaitForGPU();
        profile_timer.Start();
    }

    auto& gpu = GPUCore::GetInstance();
    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;

    // Create command encoder
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"avbd_compute", 12};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    // Copy-in: global -> local (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_pos_, pos_off, local_pos_, 0, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_vel_, vel_off, local_vel_, 0, vel_sz);
    }

    // VBD solve (init -> predict -> copy q -> iteration loop)
    dynamics_->Solve(encoder);

    // Update velocity: v = (q - x_old) / dt * damping
    {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(update_velocity_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_update_velocity_.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Update position: pos = q (free nodes), pos = x_old (pinned nodes)
    {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(update_position_pipeline_.GetHandle());
        enc.SetBindGroup(0, bg_update_position_.GetHandle());
        enc.Dispatch(node_wg);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Copy-out: local -> global (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_pos_, 0, global_pos_, pos_off, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_vel_, 0, global_vel_, vel_off, vel_sz);
    }

    // Submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    if constexpr (kEnableSimulationProfiling) {
        WaitForGPU();
        profile_timer.Stop();
        LogInfo("[Profile] ", kName, "::Update: ",
                profile_timer.GetElapsedMilliseconds(), " ms");
    }
}

// ============================================================================
// Topology change detection
// ============================================================================

AVBDSystemSimulator::TopologySignature
AVBDSystemSimulator::ComputeTopologySignature() const {
    TopologySignature sig;

    const auto& db = system_.GetDatabase();
    auto* storage = db.GetStorageById(GetComponentTypeId<AVBDSystemConfig>());
    if (!storage || storage->GetDenseCount() == 0) return sig;

    auto* typed = static_cast<const ComponentStorage<AVBDSystemConfig>*>(storage);
    Entity config_entity = typed->GetEntities()[0];
    const auto* config = db.GetComponent<AVBDSystemConfig>(config_entity);
    if (!config || config->mesh_entity == database::kInvalidEntity) return sig;

    const auto* mesh_comp = db.GetComponent<ext_mesh::MeshComponent>(config->mesh_entity);
    if (mesh_comp) {
        sig.node_count = mesh_comp->vertex_count;
        sig.edge_count = mesh_comp->edge_count;
        sig.face_count = mesh_comp->face_count;
    }
    return sig;
}

void AVBDSystemSimulator::OnDatabaseChanged() {
    auto new_sig = ComputeTopologySignature();

    if (!initialized_) {
        if (new_sig.node_count > 0 && new_sig.edge_count > 0) {
            topology_sig_ = new_sig;
            Initialize();
        }
        return;
    }

    if (new_sig.node_count == topology_sig_.node_count &&
        new_sig.edge_count == topology_sig_.edge_count &&
        new_sig.face_count == topology_sig_.face_count) {
        return;
    }

    LogInfo("AVBDSystemSimulator: topology changed, reinitializing...");
    Shutdown();
    topology_sig_ = new_sig;
    Initialize();
}

// ============================================================================
// Shutdown
// ============================================================================

void AVBDSystemSimulator::Shutdown() {
    if (dynamics_) dynamics_->Shutdown();
    dynamics_.reset();

    coloring_.Shutdown();
    local_edge_buf_.reset();

    bg_update_velocity_ = {};
    bg_update_position_ = {};
    update_velocity_pipeline_ = {};
    update_position_pipeline_ = {};

    // Release scoped local buffers
    if (local_pos_) { wgpuBufferRelease(local_pos_); local_pos_ = nullptr; }
    if (local_vel_) { wgpuBufferRelease(local_vel_); local_vel_ = nullptr; }
    if (local_mass_) { wgpuBufferRelease(local_mass_); local_mass_ = nullptr; }
    global_pos_ = nullptr;
    global_vel_ = nullptr;
    scoped_ = false;
    node_offset_ = 0;

    node_count_ = 0;
    edge_count_ = 0;
    face_count_ = 0;
    initialized_ = false;
    LogInfo("AVBDSystemSimulator: shutdown");
}

}  // namespace ext_avbd
