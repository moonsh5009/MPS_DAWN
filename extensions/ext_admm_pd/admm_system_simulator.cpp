#include "ext_admm_pd/admm_system_simulator.h"
#include "ext_admm_pd/admm_system_config.h"
#include "ext_admm_pd/admm_dynamics.h"
#include "core_simulate/simulate_config.h"
#include "core_simulate/projective_term_provider.h"
#include "ext_dynamics/global_physics_params.h"
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

namespace ext_admm_pd {

const std::string ADMMSystemSimulator::kName = "ADMMSystemSimulator";

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

ADMMSystemSimulator::ADMMSystemSimulator(system::System& system)
    : system_(system) {}

ADMMSystemSimulator::~ADMMSystemSimulator() = default;

const std::string& ADMMSystemSimulator::GetName() const {
    return kName;
}

void ADMMSystemSimulator::Initialize() {
    const auto& db = system_.GetDatabase();

    auto config_type_id = GetComponentTypeId<ADMMSystemConfig>();
    const auto* config_storage_base = db.GetStorageById(config_type_id);
    if (!config_storage_base || config_storage_base->GetDenseCount() == 0) {
        LogInfo("ADMMSystemSimulator: no ADMMSystemConfig entities found, skipping");
        return;
    }

    const auto* config_storage =
        static_cast<const ComponentStorage<ADMMSystemConfig>*>(config_storage_base);
    const auto& config_entities = config_storage->GetEntities();

    Entity config_entity = config_entities[0];
    const auto* config = db.GetComponent<ADMMSystemConfig>(config_entity);
    if (!config) return;

    WGPUBuffer pos_h, vel_h, mass_h;

    if (config->mesh_entity != database::kInvalidEntity) {
        mesh_entity_ = config->mesh_entity;

        auto* pos_entry = system_.GetArrayEntryById(GetComponentTypeId<SimPosition>());
        if (!pos_entry) {
            LogError("ADMMSystemSimulator: no SimPosition array entry");
            return;
        }
        node_offset_ = pos_entry->GetEntityOffset(mesh_entity_);
        if (node_offset_ == UINT32_MAX) {
            LogError("ADMMSystemSimulator: mesh entity ", mesh_entity_, " not in SimPosition");
            return;
        }

        auto* pos_arr = db.GetArrayStorageById(GetComponentTypeId<SimPosition>());
        node_count_ = pos_arr ? pos_arr->GetArrayCount(mesh_entity_) : 0;
        if (node_count_ == 0) {
            LogError("ADMMSystemSimulator: mesh entity has 0 SimPosition nodes");
            return;
        }

        auto& gpu_local = GPUCore::GetInstance();
        auto create_buf = [&](uint64 size) -> WGPUBuffer {
            WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
            bd.size = size;
            return wgpuDeviceCreateBuffer(gpu_local.GetDevice(), &bd);
        };

        uint64 pos_bytes = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_bytes = uint64(node_count_) * sizeof(SimVelocity);
        uint64 mass_bytes = uint64(node_count_) * sizeof(SimMass);

        local_pos_ = create_buf(pos_bytes);
        local_vel_ = create_buf(vel_bytes);
        local_mass_ = create_buf(mass_bytes);

        global_pos_ = system_.GetDeviceBuffer<SimPosition>();
        global_vel_ = system_.GetDeviceBuffer<SimVelocity>();
        WGPUBuffer global_mass = system_.GetDeviceBuffer<SimMass>();

        uint64 mass_offset = uint64(node_offset_) * sizeof(SimMass);
        WGPUCommandEncoderDescriptor me_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder me = wgpuDeviceCreateCommandEncoder(gpu_local.GetDevice(), &me_desc);
        wgpuCommandEncoderCopyBufferToBuffer(me, global_mass, mass_offset, local_mass_, 0, mass_bytes);
        WGPUCommandBuffer mc = wgpuCommandEncoderFinish(me, nullptr);
        wgpuQueueSubmit(gpu_local.GetQueue(), 1, &mc);
        wgpuCommandBufferRelease(mc);
        wgpuCommandEncoderRelease(me);

        scoped_ = true;
        pos_h = local_pos_;
        vel_h = local_vel_;
        mass_h = local_mass_;
    } else {
        node_count_ = system_.GetArrayTotalCount<SimPosition>();
        if (node_count_ == 0) {
            LogError("ADMMSystemSimulator: no SimPosition entities found");
            return;
        }
        pos_h = system_.GetDeviceBuffer<SimPosition>();
        vel_h = system_.GetDeviceBuffer<SimVelocity>();
        mass_h = system_.GetDeviceBuffer<SimMass>();
    }

    dynamics_ = std::make_unique<ADMMDynamics>();

    uint32 total_edge_count = 0;
    uint32 total_face_count = 0;

    for (uint32 i = 0; i < config->constraint_count; ++i) {
        Entity constraint_entity = config->constraint_entities[i];

        auto providers = system_.FindAllPDTermProviders(constraint_entity);
        for (auto* provider : providers) {
            auto term = provider->CreateTerm(db, constraint_entity, node_count_);

            uint32 edges = 0, faces = 0;
            provider->DeclareTopology(edges, faces);
            total_edge_count += edges;
            total_face_count += faces;

            if (term) {
                LogInfo("ADMMSystemSimulator: added term '", term->GetName(),
                        "' (edges=", edges, ", faces=", faces, ")");
                dynamics_->AddTerm(std::move(term));
            }
        }
    }

    WGPUBuffer physics_h = system_.GetDeviceDB().GetSingletonBuffer<GlobalPhysicsParams>();
    uint64 physics_sz = sizeof(PhysicsParamsGPU);

    dynamics_->SetADMMIterations(config->admm_iterations);
    dynamics_->SetCGIterations(config->cg_iterations);

    dynamics_->Initialize(node_count_, total_edge_count, total_face_count,
                          physics_h, physics_sz, pos_h, vel_h, mass_h);

    // Velocity/position update pipelines (shared PD common shaders)
    auto make_pipeline = [](const std::string& shader_path, const std::string& label) -> GPUComputePipeline {
        auto shader = ShaderLoader::CreateModule("ext_pd_common/" + shader_path, label);
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
    };

    update_velocity_pipeline_ = make_pipeline("pd_update_velocity.wgsl", "admm_update_velocity");
    update_position_pipeline_ = make_pipeline("pd_update_position.wgsl", "admm_update_position");

    uint64 params_sz = dynamics_->GetParamsSize();
    uint64 vec_sz = dynamics_->GetVec4BufferSize();
    uint64 mass_sz_bg = uint64(node_count_) * sizeof(SimMass);
    uint64 vel_sz = uint64(node_count_) * sizeof(SimVelocity);
    uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
    WGPUBuffer params_h_bg = dynamics_->GetParamsBuffer();
    WGPUBuffer q_curr_h = dynamics_->GetQCurrBuffer();
    WGPUBuffer x_old_h = dynamics_->GetXOldBuffer();

    bg_vel_ = MakeBindGroup(update_velocity_pipeline_, "bg_admm_vel",
        {{0, {physics_h, physics_sz}}, {1, {params_h_bg, params_sz}},
         {2, {vel_h, vel_sz}},
         {3, {q_curr_h, vec_sz}}, {4, {x_old_h, vec_sz}},
         {5, {mass_h, mass_sz_bg}}});

    bg_pos_ = MakeBindGroup(update_position_pipeline_, "bg_admm_pos",
        {{0, {physics_h, physics_sz}}, {1, {params_h_bg, params_sz}},
         {2, {pos_h, pos_sz}}, {3, {x_old_h, vec_sz}},
         {4, {vel_h, vel_sz}}, {5, {mass_h, mass_sz_bg}}});

    topology_sig_ = ComputeTopologySignature();
    initialized_ = true;
    LogInfo("ADMMSystemSimulator: initialized (", node_count_, " nodes)");
}

void ADMMSystemSimulator::Update() {
    if (!initialized_ || !dynamics_) return;

    Timer profile_timer;
    if constexpr (kEnableSimulationProfiling) {
        WaitForGPU();
        profile_timer.Start();
    }

    auto& gpu = GPUCore::GetInstance();
    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"admm_pd_compute", 15};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    // Copy-in: global -> local (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz_copy = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_pos_, pos_off, local_pos_, 0, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_vel_, vel_off, local_vel_, 0, vel_sz_copy);
    }

    // Solve ADMM PD (computes q_curr)
    dynamics_->Solve(encoder);

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

    // Update velocity: v = (q - x_old) / dt * damping
    dispatch(update_velocity_pipeline_, bg_vel_, node_wg);

    // Update position: pos = x_old + v * dt
    dispatch(update_position_pipeline_, bg_pos_, node_wg);

    // Copy-out: local -> global (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz_copy = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_pos_, 0, global_pos_, pos_off, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_vel_, 0, global_vel_, vel_off, vel_sz_copy);
    }

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

ADMMSystemSimulator::TopologySignature
ADMMSystemSimulator::ComputeTopologySignature() const {
    TopologySignature sig;
    sig.node_count = system_.GetArrayTotalCount<SimPosition>();

    const auto& db = system_.GetDatabase();
    auto* storage = db.GetStorageById(GetComponentTypeId<ADMMSystemConfig>());
    if (!storage || storage->GetDenseCount() == 0) return sig;

    auto* typed = static_cast<const ComponentStorage<ADMMSystemConfig>*>(storage);
    Entity config_entity = typed->GetEntities()[0];
    const auto* config = db.GetComponent<ADMMSystemConfig>(config_entity);
    if (!config) return sig;

    sig.constraint_count = config->constraint_count;
    for (uint32 i = 0; i < config->constraint_count; ++i) {
        Entity ce = config->constraint_entities[i];
        auto providers = system_.FindAllPDTermProviders(ce);
        for (auto* provider : providers) {
            uint32 e = 0, f = 0;
            provider->QueryTopology(db, ce, e, f);
            sig.total_edges += e;
            sig.total_faces += f;
        }
    }
    return sig;
}

void ADMMSystemSimulator::OnDatabaseChanged() {
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

    LogInfo("ADMMSystemSimulator: topology changed, reinitializing...");
    Shutdown();
    topology_sig_ = new_sig;
    Initialize();
}

void ADMMSystemSimulator::Shutdown() {
    if (dynamics_) dynamics_->Shutdown();
    dynamics_.reset();

    bg_vel_ = {};
    bg_pos_ = {};
    update_velocity_pipeline_ = {};
    update_position_pipeline_ = {};

    if (local_pos_) { wgpuBufferRelease(local_pos_); local_pos_ = nullptr; }
    if (local_vel_) { wgpuBufferRelease(local_vel_); local_vel_ = nullptr; }
    if (local_mass_) { wgpuBufferRelease(local_mass_); local_mass_ = nullptr; }
    global_pos_ = nullptr;
    global_vel_ = nullptr;
    scoped_ = false;
    mesh_entity_ = database::kInvalidEntity;
    node_offset_ = 0;

    initialized_ = false;
    LogInfo("ADMMSystemSimulator: shutdown");
}

}  // namespace ext_admm_pd
