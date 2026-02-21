#include "ext_newton/newton_system_simulator.h"
#include "ext_newton/newton_system_config.h"
#include "ext_newton/gravity_constraint.h"
#include "ext_newton/newton_dynamics.h"
#include "ext_dynamics/inertial_term.h"
#include "core_simulate/dynamics_term_provider.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_database/component_storage.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/compute_encoder.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;
using namespace mps::database;

namespace ext_newton {

const std::string NewtonSystemSimulator::kName = "NewtonSystemSimulator";

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
// Constructor
// ============================================================================

NewtonSystemSimulator::NewtonSystemSimulator(system::System& system)
    : system_(system) {}

NewtonSystemSimulator::~NewtonSystemSimulator() = default;

const std::string& NewtonSystemSimulator::GetName() const {
    return kName;
}

// ============================================================================
// Initialize
// ============================================================================

void NewtonSystemSimulator::Initialize() {
    const auto& db = system_.GetDatabase();

    // Find NewtonSystemConfig entities
    auto config_type_id = GetComponentTypeId<NewtonSystemConfig>();
    const auto* config_storage_base = db.GetStorageById(config_type_id);
    if (!config_storage_base || config_storage_base->GetDenseCount() == 0) {
        LogInfo("NewtonSystemSimulator: no NewtonSystemConfig entities found, skipping");
        return;
    }

    const auto* config_storage =
        static_cast<const ComponentStorage<NewtonSystemConfig>*>(config_storage_base);
    const auto& config_entities = config_storage->GetEntities();

    // Process the first config entity (single Newton system for now)
    Entity config_entity = config_entities[0];
    const auto* config = db.GetComponent<NewtonSystemConfig>(config_entity);
    if (!config) return;

    newton_iterations_ = config->newton_iterations;
    cg_max_iterations_ = config->cg_max_iterations;

    // Count nodes from DeviceDB array buffer (SimPosition is stored as per-mesh arrays)
    node_count_ = system_.GetArrayTotalCount<SimPosition>();
    if (node_count_ == 0) {
        LogError("NewtonSystemSimulator: no SimPosition entities found");
        return;
    }

    // Create dynamics solver
    dynamics_ = std::make_unique<NewtonDynamics>();

    // Always add InertialTerm (fundamental to Newton method)
    dynamics_->AddTerm(std::make_unique<ext_dynamics::InertialTerm>());

    // Discover terms from constraint entity references
    uint32 total_edge_count = 0;
    uint32 total_face_count = 0;
    float32 gravity_x = 0.0f, gravity_y = -9.81f, gravity_z = 0.0f;

    for (uint32 i = 0; i < config->constraint_count; ++i) {
        Entity constraint_entity = config->constraint_entities[i];

        // Read gravity config if present
        const auto* gravity_data = db.GetComponent<GravityConstraintData>(constraint_entity);
        if (gravity_data) {
            gravity_x = gravity_data->gx;
            gravity_y = gravity_data->gy;
            gravity_z = gravity_data->gz;
        }

        // Find matching term provider
        auto* provider = system_.FindTermProvider(constraint_entity);
        if (!provider) {
            LogWarning("NewtonSystemSimulator: no provider found for constraint entity ",
                       constraint_entity);
            continue;
        }

        // Create term (may read topology from DB, populating provider cache)
        auto term = provider->CreateTerm(db, constraint_entity, node_count_);

        // Query topology (after CreateTerm, so DB-backed providers have data)
        uint32 edges = 0, faces = 0;
        provider->DeclareTopology(edges, faces);
        total_edge_count += edges;
        total_face_count += faces;

        if (term) {
            LogInfo("NewtonSystemSimulator: added term '", term->GetName(),
                    "' (edges=", edges, ", faces=", faces, ")");
            dynamics_->AddTerm(std::move(term));
        }
    }

    // Get external buffer handles for bind group caching
    WGPUBuffer pos_h = system_.GetDeviceBuffer<SimPosition>();
    WGPUBuffer vel_h = system_.GetDeviceBuffer<SimVelocity>();
    WGPUBuffer mass_h = system_.GetDeviceBuffer<SimMass>();

    // Initialize dynamics solver with external buffer handles
    dynamics_->Initialize(node_count_, total_edge_count, total_face_count,
                          pos_h, vel_h, mass_h);

    // Configure gravity from entity data
    dynamics_->SetGravity(gravity_x, gravity_y, gravity_z);

    // Create velocity/position update pipelines
    auto make_pipeline = [](const std::string& shader_path, const std::string& label) -> GPUComputePipeline {
        auto shader = ShaderLoader::CreateModule("ext_newton/" + shader_path, label);
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
    };

    update_velocity_pipeline_ = make_pipeline("update_velocity.wgsl", "newton_update_velocity");
    update_position_pipeline_ = make_pipeline("update_position.wgsl", "newton_update_position");

    // Cache velocity/position update bind groups
    uint64 params_sz = dynamics_->GetParamsSize();
    uint64 vec_sz = dynamics_->GetVec4BufferSize();
    uint64 mass_sz_bg = uint64(node_count_) * sizeof(SimMass);
    uint64 vel_sz = uint64(node_count_) * sizeof(SimVelocity);
    uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
    WGPUBuffer params_h_bg = dynamics_->GetParamsBuffer();
    WGPUBuffer dv_total_h = dynamics_->GetDVTotalBuffer();
    WGPUBuffer x_old_h = dynamics_->GetXOldBuffer();

    bg_vel_ = MakeBindGroup(update_velocity_pipeline_, "bg_vel",
        {{0, {params_h_bg, params_sz}}, {1, {vel_h, vel_sz}}, {2, {dv_total_h, vec_sz}}, {3, {mass_h, mass_sz_bg}}});
    bg_pos_ = MakeBindGroup(update_position_pipeline_, "bg_pos",
        {{0, {params_h_bg, params_sz}}, {1, {pos_h, pos_sz}}, {2, {x_old_h, vec_sz}},
         {3, {vel_h, vel_sz}}, {4, {mass_h, mass_sz_bg}}});

    topology_sig_ = ComputeTopologySignature();
    initialized_ = true;
    LogInfo("NewtonSystemSimulator: initialized (", node_count_, " nodes, ",
            total_edge_count, " edges, ", dynamics_ ? "solver ready" : "no solver", ")");
}

// ============================================================================
// Update (per frame)
// ============================================================================

void NewtonSystemSimulator::Update(float32 dt) {
    if (!initialized_ || !dynamics_) return;

    auto& gpu = GPUCore::GetInstance();
    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;

    // Create command encoder
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"newton_compute", 14};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    // Solve dynamics (computes dv_total, uses cached bind groups)
    dynamics_->Solve(encoder, dt, newton_iterations_, cg_max_iterations_);

    auto dispatch = [&](const GPUComputePipeline& pipeline, const GPUBindGroup& bg, uint32 wg_count) {
        WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pd);
        ComputeEncoder enc(pass);
        enc.SetPipeline(pipeline.GetHandle());
        enc.SetBindGroup(0, bg.GetHandle());
        enc.Dispatch(wg_count);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    };

    // Update velocity: v = (v + dv_total) * damping
    dispatch(update_velocity_pipeline_, bg_vel_, node_wg);

    // Update position: pos = x_old + vel * dt
    dispatch(update_position_pipeline_, bg_pos_, node_wg);

    // Submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}

// ============================================================================
// Topology change detection
// ============================================================================

NewtonSystemSimulator::TopologySignature
NewtonSystemSimulator::ComputeTopologySignature() const {
    TopologySignature sig;
    sig.node_count = system_.GetArrayTotalCount<SimPosition>();

    const auto& db = system_.GetDatabase();
    auto* storage = db.GetStorageById(GetComponentTypeId<NewtonSystemConfig>());
    if (!storage || storage->GetDenseCount() == 0) return sig;

    auto* typed = static_cast<const ComponentStorage<NewtonSystemConfig>*>(storage);
    Entity config_entity = typed->GetEntities()[0];
    const auto* config = db.GetComponent<NewtonSystemConfig>(config_entity);
    if (!config) return sig;

    sig.constraint_count = config->constraint_count;
    for (uint32 i = 0; i < config->constraint_count; ++i) {
        Entity ce = config->constraint_entities[i];
        auto* provider = system_.FindTermProvider(ce);
        if (!provider) continue;
        uint32 e = 0, f = 0;
        provider->QueryTopology(db, ce, e, f);
        sig.total_edges += e;
        sig.total_faces += f;
    }
    return sig;
}

void NewtonSystemSimulator::OnDatabaseChanged() {
    auto new_sig = ComputeTopologySignature();

    if (!initialized_) {
        if (new_sig.node_count > 0) {
            topology_sig_ = new_sig;
            Initialize();
        }
        return;
    }

    if (new_sig.node_count == topology_sig_.node_count &&
        new_sig.total_edges == topology_sig_.total_edges &&
        new_sig.total_faces == topology_sig_.total_faces &&
        new_sig.constraint_count == topology_sig_.constraint_count) {
        return;
    }

    LogInfo("NewtonSystemSimulator: topology changed, reinitializing...");
    Shutdown();
    topology_sig_ = new_sig;
    Initialize();
}

// ============================================================================
// Shutdown
// ============================================================================

void NewtonSystemSimulator::Shutdown() {
    if (dynamics_) dynamics_->Shutdown();
    dynamics_.reset();

    bg_vel_ = {};
    bg_pos_ = {};
    update_velocity_pipeline_ = {};
    update_position_pipeline_ = {};

    initialized_ = false;
    LogInfo("NewtonSystemSimulator: shutdown");
}

}  // namespace ext_newton
