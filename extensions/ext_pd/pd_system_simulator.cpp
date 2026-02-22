#include "ext_pd/pd_system_simulator.h"
#include "ext_pd/pd_system_config.h"
#include "ext_pd/pd_dynamics.h"
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

namespace ext_pd {

const std::string PDSystemSimulator::kName = "PDSystemSimulator";

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

PDSystemSimulator::PDSystemSimulator(system::System& system)
    : system_(system) {}

PDSystemSimulator::~PDSystemSimulator() = default;

const std::string& PDSystemSimulator::GetName() const {
    return kName;
}

void PDSystemSimulator::Initialize() {
    const auto& db = system_.GetDatabase();

    // Find PDSystemConfig entities
    auto config_type_id = GetComponentTypeId<PDSystemConfig>();
    const auto* config_storage_base = db.GetStorageById(config_type_id);
    if (!config_storage_base || config_storage_base->GetDenseCount() == 0) {
        LogInfo("PDSystemSimulator: no PDSystemConfig entities found, skipping");
        return;
    }

    const auto* config_storage =
        static_cast<const ComponentStorage<PDSystemConfig>*>(config_storage_base);
    const auto& config_entities = config_storage->GetEntities();

    Entity config_entity = config_entities[0];
    const auto* config = db.GetComponent<PDSystemConfig>(config_entity);
    if (!config) return;

    // Determine node count and buffer handles (scoped vs global mode)
    WGPUBuffer pos_h, vel_h, mass_h;

    if (config->mesh_entity != database::kInvalidEntity) {
        mesh_entity_ = config->mesh_entity;

        // Get entity offset from DeviceDB
        auto* pos_entry = system_.GetArrayEntryById(GetComponentTypeId<SimPosition>());
        if (!pos_entry) {
            LogError("PDSystemSimulator: no SimPosition array entry");
            return;
        }
        node_offset_ = pos_entry->GetEntityOffset(mesh_entity_);
        if (node_offset_ == UINT32_MAX) {
            LogError("PDSystemSimulator: mesh entity ", mesh_entity_, " not in SimPosition");
            return;
        }

        // Get per-entity node count from host
        auto* pos_arr = db.GetArrayStorageById(GetComponentTypeId<SimPosition>());
        node_count_ = pos_arr ? pos_arr->GetArrayCount(mesh_entity_) : 0;
        if (node_count_ == 0) {
            LogError("PDSystemSimulator: mesh entity has 0 SimPosition nodes");
            return;
        }

        // Create local GPU buffers
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

        // Cache global handles for copy-in/copy-out
        global_pos_ = system_.GetDeviceBuffer<SimPosition>();
        global_vel_ = system_.GetDeviceBuffer<SimVelocity>();
        WGPUBuffer global_mass = system_.GetDeviceBuffer<SimMass>();

        // Copy mass once (immediate) — mass doesn't change at runtime
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
        // Global mode (all nodes)
        node_count_ = system_.GetArrayTotalCount<SimPosition>();
        if (node_count_ == 0) {
            LogError("PDSystemSimulator: no SimPosition entities found");
            return;
        }
        pos_h = system_.GetDeviceBuffer<SimPosition>();
        vel_h = system_.GetDeviceBuffer<SimVelocity>();
        mass_h = system_.GetDeviceBuffer<SimMass>();
    }

    // Create PD solver
    dynamics_ = std::make_unique<PDDynamics>();

    // Discover PD terms from constraint entity references
    uint32 total_edge_count = 0;
    uint32 total_face_count = 0;

    for (uint32 i = 0; i < config->constraint_count; ++i) {
        Entity constraint_entity = config->constraint_entities[i];

        // Find ALL matching PD term providers for this entity
        auto providers = system_.FindAllPDTermProviders(constraint_entity);
        for (auto* provider : providers) {
            auto term = provider->CreateTerm(db, constraint_entity, node_count_);

            uint32 edges = 0, faces = 0;
            provider->DeclareTopology(edges, faces);
            total_edge_count += edges;
            total_face_count += faces;

            if (term) {
                LogInfo("PDSystemSimulator: added term '", term->GetName(),
                        "' (edges=", edges, ", faces=", faces, ")");
                dynamics_->AddTerm(std::move(term));
            }
        }
    }

    // Get physics buffer from DeviceDB singleton
    WGPUBuffer physics_h = system_.GetDeviceDB().GetSingletonBuffer<GlobalPhysicsParams>();
    uint64 physics_sz = sizeof(PhysicsParamsGPU);

    // Store PD config iterations
    dynamics_->SetIterations(config->iterations);
    dynamics_->SetChebyshevRho(config->chebyshev_rho);

    // Initialize PD solver with physics + external buffer handles
    dynamics_->Initialize(node_count_, total_edge_count, total_face_count,
                          physics_h, physics_sz, pos_h, vel_h, mass_h);

    // Create velocity/position update pipelines (reuse Newton's shaders)
    auto make_pipeline = [](const std::string& shader_path, const std::string& label) -> GPUComputePipeline {
        auto shader = ShaderLoader::CreateModule("ext_pd/" + shader_path, label);
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.label = {label.data(), label.size()};
        desc.layout = nullptr;
        desc.compute.module = shader.GetHandle();
        std::string entry = "cs_main";
        desc.compute.entryPoint = {entry.data(), entry.size()};
        return GPUComputePipeline(wgpuDeviceCreateComputePipeline(GPUCore::GetInstance().GetDevice(), &desc));
    };

    update_velocity_pipeline_ = make_pipeline("pd_update_velocity.wgsl", "pd_update_velocity");
    update_position_pipeline_ = make_pipeline("pd_update_position.wgsl", "pd_update_position");

    // Cache velocity/position update bind groups
    uint64 params_sz = dynamics_->GetParamsSize();
    uint64 vec_sz = dynamics_->GetVec4BufferSize();
    uint64 mass_sz_bg = uint64(node_count_) * sizeof(SimMass);
    uint64 vel_sz = uint64(node_count_) * sizeof(SimVelocity);
    uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
    WGPUBuffer params_h_bg = dynamics_->GetParamsBuffer();
    WGPUBuffer q_curr_h = dynamics_->GetQCurrBuffer();
    WGPUBuffer x_old_h = dynamics_->GetXOldBuffer();

    // Velocity: v = (q - x_old) / dt * damping
    bg_vel_ = MakeBindGroup(update_velocity_pipeline_, "bg_pd_vel",
        {{0, {physics_h, physics_sz}}, {1, {params_h_bg, params_sz}},
         {2, {vel_h, vel_sz}},
         {3, {q_curr_h, vec_sz}}, {4, {x_old_h, vec_sz}},
         {5, {mass_h, mass_sz_bg}}});

    // Position: pos = x_old + v * dt (consistent with damped velocity)
    bg_pos_ = MakeBindGroup(update_position_pipeline_, "bg_pd_pos",
        {{0, {physics_h, physics_sz}}, {1, {params_h_bg, params_sz}},
         {2, {pos_h, pos_sz}}, {3, {x_old_h, vec_sz}},
         {4, {vel_h, vel_sz}}, {5, {mass_h, mass_sz_bg}}});

    topology_sig_ = ComputeTopologySignature();
    initialized_ = true;
    LogInfo("PDSystemSimulator: initialized (", node_count_, " nodes)");
}

void PDSystemSimulator::Update() {
    if (!initialized_ || !dynamics_) return;

    Timer profile_timer;
    if constexpr (kEnableSimulationProfiling) {
        WaitForGPU();
        profile_timer.Start();
    }

    auto& gpu = GPUCore::GetInstance();
    uint32 node_wg = (node_count_ + kWorkgroupSize - 1) / kWorkgroupSize;

    // First frame: calibrate ρ adaptively (requires separate GPU submissions)
    if (!rho_calibrated_ && !dynamics_->IsRhoCalibrated()) {
        // Copy-in first (separate submission for scoped mode)
        if (scoped_) {
            WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &ed);
            uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
            uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
            uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
            uint64 vel_sz_copy = uint64(node_count_) * sizeof(SimVelocity);
            wgpuCommandEncoderCopyBufferToBuffer(enc, global_pos_, pos_off, local_pos_, 0, pos_sz);
            wgpuCommandEncoderCopyBufferToBuffer(enc, global_vel_, vel_off, local_vel_, 0, vel_sz_copy);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
            WaitForGPU();
        }

        dynamics_->CalibrateRho();
        rho_calibrated_ = true;

        // Fall through to normal solve for this frame
        // (calibration changed q_curr state, but we'll do a fresh solve)
    }

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"pd_compute", 10};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    // Copy-in: global → local (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz_copy = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_pos_, pos_off, local_pos_, 0, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, global_vel_, vel_off, local_vel_, 0, vel_sz_copy);
    }

    // Solve PD (computes q_curr)
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

    // Update position: pos = x_old + v * dt (consistent with damped velocity)
    dispatch(update_position_pipeline_, bg_pos_, node_wg);

    // Copy-out: local → global (scoped mode)
    if (scoped_) {
        uint64 pos_off = uint64(node_offset_) * sizeof(SimPosition);
        uint64 vel_off = uint64(node_offset_) * sizeof(SimVelocity);
        uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
        uint64 vel_sz_copy = uint64(node_count_) * sizeof(SimVelocity);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_pos_, 0, global_pos_, pos_off, pos_sz);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, local_vel_, 0, global_vel_, vel_off, vel_sz_copy);
    }

    // Submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Debug: log sample node positions for first 20 frames
    if (debug_frame_ < 20) {
        WaitForGPU();
        WGPUBuffer pos_buf = scoped_ ? local_pos_ : system_.GetDeviceBuffer<SimPosition>();
        uint32 sample_node = std::min(uint32(2048), node_count_ - 1);
        uint64 read_offset = uint64(sample_node) * sizeof(SimPosition);
        uint64 read_size = sizeof(SimPosition);

        auto& gpu_rb = GPUCore::GetInstance();
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        bd.size = read_size;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(gpu_rb.GetDevice(), &bd);

        WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder enc_rb = wgpuDeviceCreateCommandEncoder(gpu_rb.GetDevice(), &ed);
        wgpuCommandEncoderCopyBufferToBuffer(enc_rb, pos_buf, read_offset, staging, 0, read_size);
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc_rb, nullptr);
        wgpuQueueSubmit(gpu_rb.GetQueue(), 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc_rb);

        WaitForGPU();
        struct Ctx { bool done = false; };
        Ctx map_ctx;
        WGPUBufferMapCallbackInfo mi = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        mi.mode = WGPUCallbackMode_WaitAnyOnly;
        mi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* ud, void*) {
            static_cast<Ctx*>(ud)->done = true;
        };
        mi.userdata1 = &map_ctx;
        WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, read_size, mi);
        WGPUFutureWaitInfo wi = WGPU_FUTURE_WAIT_INFO_INIT;
        wi.future = future;
        wgpuInstanceWaitAny(gpu_rb.GetWGPUInstance(), 1, &wi, UINT64_MAX);

        const float32* p = static_cast<const float32*>(
            wgpuBufferGetConstMappedRange(staging, 0, read_size));
        LogInfo("[PD] frame=", debug_frame_, " node=", sample_node,
                " pos=(", p[0], ", ", p[1], ", ", p[2], ")");
        wgpuBufferUnmap(staging);
        wgpuBufferRelease(staging);
        debug_frame_++;
    }

    if constexpr (kEnableSimulationProfiling) {
        WaitForGPU();
        profile_timer.Stop();
        LogInfo("[Profile] ", kName, "::Update: ",
                profile_timer.GetElapsedMilliseconds(), " ms");
    }
}

PDSystemSimulator::TopologySignature
PDSystemSimulator::ComputeTopologySignature() const {
    TopologySignature sig;
    sig.node_count = system_.GetArrayTotalCount<SimPosition>();

    const auto& db = system_.GetDatabase();
    auto* storage = db.GetStorageById(GetComponentTypeId<PDSystemConfig>());
    if (!storage || storage->GetDenseCount() == 0) return sig;

    auto* typed = static_cast<const ComponentStorage<PDSystemConfig>*>(storage);
    Entity config_entity = typed->GetEntities()[0];
    const auto* config = db.GetComponent<PDSystemConfig>(config_entity);
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

void PDSystemSimulator::OnDatabaseChanged() {
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

    LogInfo("PDSystemSimulator: topology changed, reinitializing...");
    Shutdown();
    topology_sig_ = new_sig;
    Initialize();
}

void PDSystemSimulator::Shutdown() {
    if (dynamics_) dynamics_->Shutdown();
    dynamics_.reset();

    bg_vel_ = {};
    bg_pos_ = {};
    update_velocity_pipeline_ = {};
    update_position_pipeline_ = {};

    // Release scoped local buffers
    if (local_pos_) { wgpuBufferRelease(local_pos_); local_pos_ = nullptr; }
    if (local_vel_) { wgpuBufferRelease(local_vel_); local_vel_ = nullptr; }
    if (local_mass_) { wgpuBufferRelease(local_mass_); local_mass_ = nullptr; }
    global_pos_ = nullptr;
    global_vel_ = nullptr;
    scoped_ = false;
    mesh_entity_ = database::kInvalidEntity;
    node_offset_ = 0;

    initialized_ = false;
    rho_calibrated_ = false;
    LogInfo("PDSystemSimulator: shutdown");
}

}  // namespace ext_pd
