#include "core_system/system.h"
#include "core_system/extension.h"
#include "core_simulate/simulator.h"
#include "core_simulate/dynamics_term_provider.h"
#include "core_simulate/projective_term_provider.h"
#include "core_render/object_renderer.h"
#include "core_render/render_engine.h"
#include "core_render/pass/render_pass_builder.h"
#include "core_platform/window.h"
#include "core_platform/input.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cstring>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace mps;
using namespace mps::system;
using namespace mps::database;
using namespace mps::util;

System::System()
    : device_db_(db_) {}

System::~System() {
    ShutdownExtensions();
    if (engine_) {
        engine_->Shutdown();
        engine_.reset();
    }
    auto& gpu = gpu::GPUCore::GetInstance();
    if (gpu.IsInitialized()) {
        gpu.Shutdown();
    }
    if (window_) {
        window_->Shutdown();
        window_.reset();
    }
    LogInfo("MPS_DAWN finished.");
}

// ============================================================================
// Lifecycle
// ============================================================================

bool System::Initialize() {
    LogInfo("MPS_DAWN starting...");

    // --- Create window ---
    window_ = platform::IWindow::Create();
    platform::WindowConfig win_config;
    win_config.title = "MPS_DAWN";
    win_config.width = 1280;
    win_config.height = 720;
    if (!window_->Initialize(win_config)) {
        LogError("Failed to initialize window");
        return false;
    }

    // --- Initialize GPU ---
    auto& gpu = gpu::GPUCore::GetInstance();
    WGPUSurface surface = gpu.CreateSurface(
        window_->GetNativeWindowHandle(), window_->GetNativeDisplayHandle());
    if (!gpu.Initialize({}, surface)) {
        LogError("Failed to initialize GPU");
        return false;
    }

#ifndef __EMSCRIPTEN__
    // Native: synchronous wait — callbacks fire via WaitAny
    while (!gpu.IsInitialized()) {
        gpu.ProcessEvents();
    }
    pending_surface_ = surface;
    FinishGPUInit();
#else
    // WASM: async — adapter/device callbacks need browser event loop.
    // Deferred to Run() via emscripten_set_main_loop.
    pending_surface_ = surface;
#endif

    return true;
}

void System::FinishGPUInit() {
    auto& gpu = gpu::GPUCore::GetInstance();
    LogInfo("GPU initialized: ", gpu.GetAdapterName());
    LogInfo("Backend: ", gpu.GetBackendType());

    // Create RenderEngine (needs device)
    engine_ = std::make_unique<render::RenderEngine>();
    render::RenderEngineConfig render_config;
    render_config.clear_color = {0.1, 0.1, 0.15, 1.0};
    engine_->Initialize(pending_surface_, window_->GetWidth(), window_->GetHeight(), render_config);
    pending_surface_ = nullptr;

    // Sync any data transacted before GPU was ready
    device_db_.Sync();

    gpu_ready_ = true;
}

void System::Run() {
#ifndef __EMSCRIPTEN__
    // Native: synchronous extensions init + main loop
    InitializeExtensions();

    LogInfo("Entering main loop... (simulation paused, press Space to start)");

    while (!window_->ShouldClose()) {
        RunFrame();
    }
#else
    // WASM: async main loop — yields to browser event loop each frame
    emscripten_set_main_loop_arg(EmscriptenMainLoop, this, 0, true);
    // Does not return (simulates infinite loop)
#endif
}

#ifdef __EMSCRIPTEN__
void System::EmscriptenMainLoop(void* arg) {
    auto* self = static_cast<System*>(arg);
    auto& gpu = gpu::GPUCore::GetInstance();

    // Wait for async GPU initialization
    if (!self->gpu_ready_) {
        if (!gpu.IsInitialized()) {
            gpu.ProcessEvents();
            return;
        }
        self->FinishGPUInit();
        self->InitializeExtensions();
        LogInfo("Entering main loop... (simulation paused, press Space to start)");
        return;
    }

    self->RunFrame();
}
#endif

void System::RunFrame() {
    constexpr float32 dt = 1.0f / 60.0f;
    auto& input = platform::InputManager::GetInstance();
    window_->PollEvents();

    // Space: toggle simulation
    if (platform::IsKeyPressed(platform::Key::Space)) {
        simulation_running_ = !simulation_running_;
        LogInfo("Simulation ", simulation_running_ ? "running" : "paused");
    }

    // R: reset simulation (restore GPU from Database)
    if (platform::IsKeyPressed(platform::Key::R)) {
        ResetSimulation();
    }

    // ESC: quit
    if (platform::IsKeyPressed(platform::Key::Escape)) {
#ifdef __EMSCRIPTEN__
        emscripten_cancel_main_loop();
#endif
        return;
    }

    // Update simulators (only when running)
    if (simulation_running_) {
        UpdateSimulators();
    }

    // Update camera and uniforms
    engine_->UpdateUniforms(dt);

    // Render
    RenderFrame();

    // Transition input states AFTER game logic reads them
    // Native: PollEvents() delivers events → game reads Pressed → Update transitions to Held
    // WASM: events arrive between frames → game reads Pressed → Update transitions to Held
    input.Update();
}

// ============================================================================
// Simulation control
// ============================================================================

bool System::IsSimulationRunning() const {
    return simulation_running_;
}

void System::SetSimulationRunning(bool running) {
    simulation_running_ = running;
}

void System::ResetSimulation() {
    device_db_.ForceSync();
    simulation_running_ = false;
    LogInfo("Simulation reset");
}

// ============================================================================
// Transactions
// ============================================================================

void System::Transact(std::function<void(Database&)> fn) {
    db_.Transact([&] { fn(db_); });
    SyncToDevice();
    NotifyDatabaseChanged();
}

void System::Undo() {
    if (db_.Undo()) {
        SyncToDevice();
        NotifyDatabaseChanged();
    }
}

void System::Redo() {
    if (db_.Redo()) {
        SyncToDevice();
        NotifyDatabaseChanged();
    }
}

bool System::CanUndo() const {
    return db_.CanUndo();
}

bool System::CanRedo() const {
    return db_.CanRedo();
}

const Database& System::GetDatabase() const {
    return db_;
}

Database& System::GetDatabase() {
    return db_;
}

simulate::DeviceDB& System::GetDeviceDB() {
    return device_db_;
}

simulate::IDeviceArrayEntry* System::GetArrayEntryById(database::ComponentTypeId id) const {
    return device_db_.GetArrayEntryById(id);
}

void System::SyncToDevice() {
    if (!gpu::GPUCore::GetInstance().IsInitialized()) return;
    device_db_.Sync();
}

void System::NotifyDatabaseChanged() {
    if (!extensions_initialized_) return;
    for (auto& sim : simulators_) {
        sim->OnDatabaseChanged();
    }
}

// ============================================================================
// Extension system
// ============================================================================

void System::AddExtension(std::unique_ptr<IExtension> extension) {
    LogInfo("Registering extension: ", extension->GetName());
    extension->Register(*this);
    extensions_.push_back(std::move(extension));
}

void System::AddSimulator(std::unique_ptr<simulate::ISimulator> simulator) {
    LogInfo("Simulator added: ", simulator->GetName());
    simulators_.push_back(std::move(simulator));
}

void System::AddRenderer(std::unique_ptr<render::IObjectRenderer> renderer) {
    LogInfo("Renderer added: ", renderer->GetName());
    renderers_.push_back(std::move(renderer));
}

void System::RegisterTermProvider(database::ComponentTypeId config_type,
                                   std::unique_ptr<simulate::IDynamicsTermProvider> provider) {
    LogInfo("Term provider registered: ", provider->GetTermName());
    term_providers_.emplace(config_type, std::move(provider));
}

simulate::IDynamicsTermProvider* System::FindTermProvider(database::Entity constraint_entity) const {
    for (const auto& [type_id, provider] : term_providers_) {
        if (provider->HasConfig(db_, constraint_entity)) {
            return provider.get();
        }
    }
    return nullptr;
}

void System::RegisterPDTermProvider(database::ComponentTypeId config_type,
                                     std::unique_ptr<simulate::IProjectiveTermProvider> provider) {
    LogInfo("PD term provider registered: ", provider->GetTermName());
    pd_term_providers_.emplace(config_type, std::move(provider));
}

simulate::IProjectiveTermProvider* System::FindPDTermProvider(database::Entity constraint_entity) const {
    for (const auto& [type_id, provider] : pd_term_providers_) {
        if (provider->HasConfig(db_, constraint_entity)) {
            return provider.get();
        }
    }
    return nullptr;
}

std::vector<simulate::IDynamicsTermProvider*> System::FindAllTermProviders(
    database::Entity constraint_entity) const {
    std::vector<simulate::IDynamicsTermProvider*> result;
    for (const auto& [type_id, provider] : term_providers_) {
        if (provider->HasConfig(db_, constraint_entity)) {
            result.push_back(provider.get());
        }
    }
    return result;
}

std::vector<simulate::IProjectiveTermProvider*> System::FindAllPDTermProviders(
    database::Entity constraint_entity) const {
    std::vector<simulate::IProjectiveTermProvider*> result;
    for (const auto& [type_id, provider] : pd_term_providers_) {
        if (provider->HasConfig(db_, constraint_entity)) {
            result.push_back(provider.get());
        }
    }
    return result;
}

void System::InitializeExtensions() {
    if (extensions_initialized_) {
        LogError("Extensions already initialized");
        return;
    }

    // 1) Initialize simulators (GPU pipeline setup)
    for (auto& sim : simulators_) {
        LogInfo("Initializing simulator: ", sim->GetName());
        sim->Initialize();
    }

    // 2) Sort renderers by order (lower = earlier)
    std::sort(renderers_.begin(), renderers_.end(),
        [](const auto& a, const auto& b) {
            return a->GetOrder() < b->GetOrder();
        });

    // 3) Initialize renderers
    for (auto& renderer : renderers_) {
        LogInfo("Initializing renderer: ", renderer->GetName());
        renderer->Initialize(*engine_);
    }

    extensions_initialized_ = true;
    LogInfo("Extensions initialized (", simulators_.size(), " simulators, ",
            renderers_.size(), " renderers)");
}

void System::ShutdownExtensions() {
    if (!extensions_initialized_) {
        return;
    }

    for (auto it = renderers_.rbegin(); it != renderers_.rend(); ++it) {
        (*it)->Shutdown();
    }

    for (auto it = simulators_.rbegin(); it != simulators_.rend(); ++it) {
        (*it)->Shutdown();
    }

    renderers_.clear();
    simulators_.clear();
    extensions_.clear();
    extensions_initialized_ = false;
}

void System::UpdateSimulators() {
    for (auto& sim : simulators_) {
        sim->Update();
    }
}

void System::RenderFrame() {
    uint32 w = window_->GetWidth();
    uint32 h = window_->GetHeight();
    if (w != engine_->GetWidth() || h != engine_->GetHeight()) {
        engine_->Resize(w, h);
    }

    if (engine_->BeginFrame()) {
        render::RenderPassBuilder("main_pass")
            .AddColorAttachment(engine_->GetFrameView(),
                render::LoadOp::Clear, render::StoreOp::Store,
                {0.1, 0.1, 0.15, 1.0})
            .SetDepthStencilAttachment(engine_->GetDepthTarget().GetView(),
                render::LoadOp::Clear, render::StoreOp::Store, 1.0f)
            .Execute(engine_->GetEncoder(), [&](WGPURenderPassEncoder pass) {
                for (auto& renderer : renderers_) {
                    renderer->Render(*engine_, pass);
                }
            });

        engine_->EndFrame();
    }
}

// ============================================================================
// GPU readback
// ============================================================================

std::vector<uint8> System::ReadbackBuffer(WGPUBuffer src, uint64 size) {
    auto& gpu = gpu::GPUCore::GetInstance();

    // Create staging buffer
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.size = size;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(gpu.GetDevice(), &staging_desc);

    // Copy src -> staging
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, staging, 0, size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Synchronous map
    struct Ctx { bool done = false; bool ok = false; };
    Ctx ctx;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* c = static_cast<Ctx*>(ud1);
        c->done = true;
        c->ok = (status == WGPUMapAsyncStatus_Success);
    };
    cb.userdata1 = &ctx;
    WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0,
                                            static_cast<size_t>(size), cb);
#ifndef __EMSCRIPTEN__
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(gpu.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
    while (!ctx.done) gpu.ProcessEvents();
#endif

    std::vector<uint8> result;
    if (ctx.ok) {
        auto* mapped = wgpuBufferGetConstMappedRange(staging, 0, static_cast<size_t>(size));
        result.resize(static_cast<size_t>(size));
        std::memcpy(result.data(), mapped, result.size());
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
    return result;
}
