#pragma once

#include "core_gpu/gpu_core.h"
#include "core_util/timer.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>

namespace mps {
namespace simulate {

// Toggle simulation profiling at compile time.
// When true, ISimulator::Update() implementations log GPU execution time via if constexpr.
// Only included by .cpp files — changing this value only rebuilds those .cpp files.
inline constexpr bool kEnableSimulationProfiling = true;

// Wait for all previously submitted GPU work to complete.
// Native: uses wgpuQueueOnSubmittedWorkDone + WaitAny (synchronous).
// WASM: no-op — synchronous GPU waits deadlock because ProcessEvents()
// busy-wait prevents JS promises from resolving. Callers must not rely
// on this for correctness on WASM; use asynchronous patterns instead.
inline void WaitForGPU() {
#ifdef __EMSCRIPTEN__
    // Cannot synchronously wait on WASM — JS event loop must run for
    // GPU callbacks to fire. This is intentionally a no-op.
    return;
#else
    auto& gpu = gpu::GPUCore::GetInstance();

    struct Ctx { bool done = false; };
    Ctx ctx;

    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* ud1, void*) {
        static_cast<Ctx*>(ud1)->done = true;
    };
    cb.userdata1 = &ctx;

    WGPUFuture future = wgpuQueueOnSubmittedWorkDone(gpu.GetQueue(), cb);

    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(gpu.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#endif
}

}  // namespace simulate
}  // namespace mps
