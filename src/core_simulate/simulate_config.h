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
// Uses wgpuQueueOnSubmittedWorkDone + WaitAny (native) or ProcessEvents (WASM).
inline void WaitForGPU() {
    auto& gpu = gpu::GPUCore::GetInstance();

    struct Ctx { bool done = false; };
    Ctx ctx;

    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* ud1, void*) {
        static_cast<Ctx*>(ud1)->done = true;
    };
    cb.userdata1 = &ctx;

    WGPUFuture future = wgpuQueueOnSubmittedWorkDone(gpu.GetQueue(), cb);

#ifndef __EMSCRIPTEN__
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(gpu.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
    while (!ctx.done) gpu.ProcessEvents();
#endif
}

}  // namespace simulate
}  // namespace mps
