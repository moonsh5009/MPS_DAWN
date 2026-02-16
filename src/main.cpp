#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include "core_util/types.h"

using namespace mps::util;
using namespace mps::gpu;

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

    // Cleanup
    gpu.Shutdown();

    LogInfo("MPS_DAWN finished.");
    return 0;
}
