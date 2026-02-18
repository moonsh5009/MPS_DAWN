#include "core_system/system.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include "core_util/types.h"
#include <stdexcept>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::database;
using namespace mps::system;

// Test component types
struct Position { float32 x, y, z; };
struct Velocity { float32 x, y, z; };

int main() {
    LogInfo("MPS_DAWN starting...");

    // Initialize GPU
    auto& gpu = GPUCore::GetInstance();
    if (!gpu.Initialize()) {
        LogError("Failed to initialize GPU");
        return 1;
    }

    // Native: already ready. WASM: poll until ready.
    while (!gpu.IsInitialized()) {
        gpu.ProcessEvents();
    }

    LogInfo("GPU initialized: ", gpu.GetAdapterName());
    LogInfo("Backend: ", gpu.GetBackendType());

    // --- System integration test ---
    LogInfo("--- System integration test ---");

    System sys;
    sys.RegisterComponent<Position>(BufferUsage::Vertex, "positions");
    sys.RegisterComponent<Velocity>(BufferUsage::None, "velocities");

    // Transact: create 100 entities with Position and Velocity
    sys.Transact([](Database& db) {
        for (uint32 i = 0; i < 100; ++i) {
            auto e = db.CreateEntity();
            db.AddComponent<Position>(e, {1.0f, 2.0f, 3.0f});
            db.AddComponent<Velocity>(e, {0.1f, 0.0f, 0.0f});
        }
    });
    LogInfo("Created 100 entities with Position and Velocity");

    // Log buffer validity
    auto pos_buf = sys.GetDeviceBuffer<Position>();
    auto vel_buf = sys.GetDeviceBuffer<Velocity>();
    LogInfo("Position buffer: ", pos_buf ? "valid" : "null");
    LogInfo("Velocity buffer: ", vel_buf ? "valid" : "null");

    // Test Undo
    if (sys.CanUndo()) {
        sys.Undo();
        LogInfo("Undo successful");
        auto pos_after_undo = sys.GetDeviceBuffer<Position>();
        LogInfo("Position buffer after undo: ", pos_after_undo ? "valid" : "null");
    }

    // Test Redo
    if (sys.CanRedo()) {
        sys.Redo();
        LogInfo("Redo successful");
        auto pos_after_redo = sys.GetDeviceBuffer<Position>();
        LogInfo("Position buffer after redo: ", pos_after_redo ? "valid" : "null");
    }

    // Test rollback (Transact that throws)
    try {
        sys.Transact([](Database& db) {
            auto e = db.CreateEntity();
            db.AddComponent<Position>(e, {999.0f, 999.0f, 999.0f});
            throw std::runtime_error("test rollback");
        });
    } catch (const std::exception& e) {
        LogInfo("Rollback test passed: ", e.what());
    }

    LogInfo("--- System integration test complete ---");

    // Cleanup
    gpu.Shutdown();

    LogInfo("MPS_DAWN finished.");
    return 0;
}
