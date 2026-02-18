#include "core_system/system.h"
#include "core_system/extension.h"
#include "core_simulate/simulator.h"
#include "core_render/object_renderer.h"
#include "core_render/render_engine.h"
#include "core_util/logger.h"
#include <algorithm>

using namespace mps;
using namespace mps::system;
using namespace mps::database;
using namespace mps::util;

System::System()
    : device_db_(db_) {}

System::~System() = default;

void System::Transact(std::function<void(Database&)> fn) {
    db_.Transact([&] { fn(db_); });
    SyncToDevice();
}

void System::Undo() {
    if (db_.Undo()) {
        SyncToDevice();
    }
}

void System::Redo() {
    if (db_.Redo()) {
        SyncToDevice();
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

simulate::DeviceDB& System::GetDeviceDB() {
    return device_db_;
}

void System::SyncToDevice() {
    device_db_.Sync();
}

// --- Extension system ---

void System::AddExtension(std::unique_ptr<IExtension> extension) {
    LogInfo("Extension added: ", extension->GetName());
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

void System::InitializeExtensions(render::RenderEngine& engine) {
    if (extensions_initialized_) {
        LogError("Extensions already initialized");
        return;
    }

    // 1) Let each extension register components, simulators, renderers
    for (auto& ext : extensions_) {
        LogInfo("Registering extension: ", ext->GetName());
        ext->Register(*this);
    }

    // 2) Initialize simulators (each wrapped in a transaction for GPU sync)
    for (auto& sim : simulators_) {
        LogInfo("Initializing simulator: ", sim->GetName());
        Transact([&](Database& db) {
            sim->Initialize(db);
        });
    }

    // 3) Sort renderers by order (lower = earlier)
    std::sort(renderers_.begin(), renderers_.end(),
        [](const auto& a, const auto& b) {
            return a->GetOrder() < b->GetOrder();
        });

    // 4) Initialize renderers
    for (auto& renderer : renderers_) {
        LogInfo("Initializing renderer: ", renderer->GetName());
        renderer->Initialize(engine);
    }

    extensions_initialized_ = true;
    LogInfo("Extensions initialized (", simulators_.size(), " simulators, ",
            renderers_.size(), " renderers)");
}

void System::ShutdownExtensions() {
    if (!extensions_initialized_) {
        return;
    }

    // Reverse-order shutdown: renderers first, then simulators
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

void System::UpdateSimulators(float32 dt) {
    for (auto& sim : simulators_) {
        Transact([&](Database& db) {
            sim->Update(db, dt);
        });
    }
}

void System::RenderAll(render::RenderEngine& engine, WGPURenderPassEncoder pass) {
    for (auto& renderer : renderers_) {
        renderer->Render(engine, pass);
    }
}
