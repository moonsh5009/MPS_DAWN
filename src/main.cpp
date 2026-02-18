#include "core_platform/window.h"
#include "core_platform/input.h"
#include "core_gpu/gpu_core.h"
#include "core_gpu/gpu_buffer.h"
#include "core_render/render_engine.h"
#include "core_gpu/shader_loader.h"
#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/bind_group_builder.h"
#include "core_gpu/pipeline_layout_builder.h"
#include "core_render/pipeline/render_pipeline_builder.h"
#include "core_render/pass/render_pass_builder.h"
#include "core_render/pass/render_encoder.h"
#include "core_render/uniform/camera_uniform.h"
#include "core_system/system.h"
#include "ext_sample/sample_extension.h"
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

// Vertex structure for colored cube
struct Vertex {
    float32 position[3];
    float32 color[4];
};

int main() {
    LogInfo("MPS_DAWN starting...");

    // --- Create window ---
    auto window = IWindow::Create();
    WindowConfig win_config;
    win_config.title = "MPS_DAWN - Render Engine";
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
    system.AddExtension(std::make_unique<ext_sample::SampleExtension>(system));
    system.InitializeExtensions(engine);

    // --- Create cube geometry ---
    // Cube vertices: position (xyz) + color (rgba)
    Vertex cube_vertices[] = {
        // Front face (red)
        {{-0.5f, -0.5f,  0.5f}, {0.9f, 0.2f, 0.2f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, {0.9f, 0.2f, 0.2f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {0.9f, 0.3f, 0.3f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, {0.9f, 0.3f, 0.3f, 1.0f}},
        // Back face (green)
        {{ 0.5f, -0.5f, -0.5f}, {0.2f, 0.9f, 0.2f, 1.0f}},
        {{-0.5f, -0.5f, -0.5f}, {0.2f, 0.9f, 0.2f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {0.3f, 0.9f, 0.3f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {0.3f, 0.9f, 0.3f, 1.0f}},
        // Top face (blue)
        {{-0.5f,  0.5f,  0.5f}, {0.2f, 0.2f, 0.9f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {0.2f, 0.2f, 0.9f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {0.3f, 0.3f, 0.9f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {0.3f, 0.3f, 0.9f, 1.0f}},
        // Bottom face (yellow)
        {{-0.5f, -0.5f, -0.5f}, {0.9f, 0.9f, 0.2f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, {0.9f, 0.9f, 0.2f, 1.0f}},
        {{ 0.5f, -0.5f,  0.5f}, {0.9f, 0.9f, 0.3f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, {0.9f, 0.9f, 0.3f, 1.0f}},
        // Right face (magenta)
        {{ 0.5f, -0.5f,  0.5f}, {0.9f, 0.2f, 0.9f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, {0.9f, 0.2f, 0.9f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {0.9f, 0.3f, 0.9f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {0.9f, 0.3f, 0.9f, 1.0f}},
        // Left face (cyan)
        {{-0.5f, -0.5f, -0.5f}, {0.2f, 0.9f, 0.9f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, {0.2f, 0.9f, 0.9f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, {0.3f, 0.9f, 0.9f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {0.3f, 0.9f, 0.9f, 1.0f}},
    };

    uint32 cube_indices[] = {
         0,  1,  2,   0,  2,  3,   // front
         4,  5,  6,   4,  6,  7,   // back
         8,  9, 10,   8, 10, 11,   // top
        12, 13, 14,  12, 14, 15,   // bottom
        16, 17, 18,  16, 18, 19,   // right
        20, 21, 22,  20, 22, 23,   // left
    };

    GPUBuffer<Vertex> vertex_buffer(BufferUsage::Vertex,
        std::span<const Vertex>(cube_vertices, 24), "cube_vertices");
    GPUBuffer<uint32> index_buffer(BufferUsage::Index,
        std::span<const uint32>(cube_indices, 36), "cube_indices");

    // --- Load shaders ---
    auto shader = ShaderLoader::CreateModule("basic/mesh_vert.wgsl", "mesh_vert");
    auto frag_shader = ShaderLoader::CreateModule("basic/mesh_frag.wgsl", "mesh_frag");

    // --- Create camera bind group layout and bind group ---
    auto camera_bgl = BindGroupLayoutBuilder("camera_bgl")
        .AddUniformBinding(0, ShaderStage::Vertex)
        .Build();

    auto camera_bg = BindGroupBuilder("camera_bg")
        .AddBuffer(0, engine.GetCameraUniform().GetBuffer(), sizeof(CameraUBOData))
        .Build(camera_bgl);

    // --- Create pipeline ---
    auto pipeline_layout = PipelineLayoutBuilder("main_layout")
        .AddBindGroupLayout(camera_bgl)
        .Build();

    auto pipeline = RenderPipelineBuilder("cube_pipeline")
        .SetPipelineLayout(pipeline_layout)
        .SetVertexShader(shader.GetHandle())
        .SetFragmentShader(frag_shader.GetHandle())
        .AddVertexBufferLayout(VertexStepMode::Vertex, sizeof(Vertex), {
            {0, VertexFormat::Float32x3, 0},                           // position
            {1, VertexFormat::Float32x4, sizeof(float32) * 3},         // color
        })
        .AddColorTarget(engine.GetColorFormat())
        .SetDepthStencil(engine.GetDepthFormat(), true, CompareFunction::Less)
        .SetPrimitive(PrimitiveTopology::TriangleList, CullMode::Back, FrontFace::CCW)
        .Build();

    // Release intermediate objects (pipeline holds refs)
    wgpuPipelineLayoutRelease(pipeline_layout);

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
            // Recreate camera bind group (buffer handle may change)
            camera_bg = BindGroupBuilder("camera_bg")
                .AddBuffer(0, engine.GetCameraUniform().GetBuffer(), sizeof(CameraUBOData))
                .Build(camera_bgl);
        }

        // Update simulators
        system.UpdateSimulators(dt);

        // Update camera
        engine.GetCameraController().Update(dt);
        engine.GetCameraUniform().Update(engine.GetCamera(), w, h);
        engine.GetCamera().ClearDirty();

        // Render
        if (engine.BeginFrame()) {
            RenderPassBuilder("main_pass")
                .AddColorAttachment(engine.GetFrameView(),
                    LoadOp::Clear, StoreOp::Store,
                    {0.1, 0.1, 0.15, 1.0})
                .SetDepthStencilAttachment(engine.GetDepthTarget().GetView(),
                    LoadOp::Clear, StoreOp::Store, 1.0f)
                .Execute(engine.GetEncoder(), [&](WGPURenderPassEncoder pass) {
                    RenderEncoder enc(pass);
                    enc.SetPipeline(pipeline);
                    enc.SetBindGroup(0, camera_bg);
                    enc.SetVertexBuffer(0, vertex_buffer.GetHandle());
                    enc.SetIndexBuffer(index_buffer.GetHandle());
                    enc.DrawIndexed(36);

                    // Extension renderers
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

    wgpuRenderPipelineRelease(pipeline);
    wgpuBindGroupRelease(camera_bg);
    wgpuBindGroupLayoutRelease(camera_bgl);

    engine.Shutdown();
    gpu.Shutdown();
    window->Shutdown();

    LogInfo("MPS_DAWN finished.");
    return 0;
}
