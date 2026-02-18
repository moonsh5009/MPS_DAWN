#include "core_platform/window.h"
#include "core_platform/input.h"
#include "core_gpu/gpu_core.h"
#include "core_render/render_engine.h"
#include "core_render/pass/render_pass_builder.h"
#include "core_render/pass/render_encoder.h"
#include "core_system/system.h"
#include "ext_cloth/cloth_extension.h"
#include "core_util/logger.h"
#include "core_util/types.h"
#include "core_util/timer.h"
#include <webgpu/webgpu.h>
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::render;
using namespace mps::platform;
using namespace mps::system;

int main() {
    LogInfo("MPS_DAWN starting...");

    // --- Create window ---
    auto window = IWindow::Create();
    WindowConfig win_config;
    win_config.title = "MPS_DAWN";
    win_config.width = 1280;
    win_config.height = 720;
    if (!window->Initialize(win_config)) {
        LogError("Failed to initialize window");
        return 1;
    }

    // --- Initialize GPU with surface ---
    auto& gpu = GPUCore::GetInstance();
    WGPUSurface surface = gpu.CreateSurface(window->GetNativeWindowHandle(), window->GetNativeDisplayHandle());
    if (!gpu.Initialize({}, surface)) {
        LogError("Failed to initialize GPU");
        return 1;
    }
    while (!gpu.IsInitialized()) {
        gpu.ProcessEvents();
    }
    LogInfo("GPU initialized: ", gpu.GetAdapterName());
    LogInfo("Backend: ", gpu.GetBackendType());

    // --- Initialize RenderEngine ---
    RenderEngine engine;
    RenderEngineConfig render_config;
    render_config.clear_color = {0.1, 0.1, 0.15, 1.0};
    engine.Initialize(surface, window->GetWidth(), window->GetHeight(), render_config);

    // --- Initialize extension system ---
    System system;
    system.AddExtension(std::make_unique<ext_cloth::ClothExtension>(system));
    system.InitializeExtensions(engine);

    // --- Main loop ---
    auto& input = InputManager::GetInstance();
    Timer timer;
    timer.Start();

    LogInfo("Entering main loop...");

    while (!window->ShouldClose()) {
        input.Update();
        window->PollEvents();

        float32 dt = static_cast<float32>(timer.GetElapsedSeconds());
        timer.Reset();
        timer.Start();

        // Handle resize
        uint32 w = window->GetWidth();
        uint32 h = window->GetHeight();
        if (w != engine.GetWidth() || h != engine.GetHeight()) {
            engine.Resize(w, h);
        }

        // Update simulators
        system.UpdateSimulators(dt);

        // Update camera and uniforms
        engine.UpdateUniforms(dt);

        // Render
        if (engine.BeginFrame()) {
            RenderPassBuilder("main_pass")
                .AddColorAttachment(engine.GetFrameView(),
                    LoadOp::Clear, StoreOp::Store,
                    {0.1, 0.1, 0.15, 1.0})
                .SetDepthStencilAttachment(engine.GetDepthTarget().GetView(),
                    LoadOp::Clear, StoreOp::Store, 1.0f)
                .Execute(engine.GetEncoder(), [&](WGPURenderPassEncoder pass) {
                    system.RenderAll(engine, pass);
                });

            engine.EndFrame();
        }

        // ESC to quit
        if (IsKeyPressed(Key::Escape)) {
            break;
        }
    }

    // Cleanup
    system.ShutdownExtensions();
    engine.Shutdown();
    gpu.Shutdown();
    window->Shutdown();

    LogInfo("MPS_DAWN finished.");
    return 0;
}
