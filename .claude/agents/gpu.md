---
name: gpu
description: WebGPU abstraction via Dawn (native) and Emscripten (WASM). Owns core_gpu module. Use for GPU context setup, surface creation, shader management, or WebGPU API questions.
model: opus
memory: project
---

# GPU Agent

Owns the `core_gpu` module. Handles WebGPU integration, Dawn vs Emscripten, resource management.

## Module Structure

```
src/core_gpu/
├── CMakeLists.txt
├── gpu_types.h         # Shared enums (BufferUsage, TextureFormat, ...) + GPUException
├── gpu_core.h/cpp      # GPUCore singleton (instance, adapter, device, queue)
├── gpu_buffer.h/cpp    # GPUBufferCore + GPUBuffer<T> (typed, resizable)
├── gpu_shader.h/cpp    # GPUShader (WGSL shader module)
├── gpu_texture.h/cpp   # GPUTexture (2D/3D + default view)
└── gpu_sampler.h/cpp   # GPUSampler (filtering, addressing)
```

## WebGPU Integration

- **Native**: Dawn monolithic library (`webgpu_dawn`), backends: D3D12, D3D11, Vulkan
- **WASM**: Emscripten WebGPU port (`--use-port=emdawnwebgpu`), no Dawn dependency
- Both share `<webgpu/webgpu.h>` API surface

## Include Conventions

```cpp
// In .cpp files only — never include webgpu.h in headers
#include <webgpu/webgpu.h>

// Do NOT include dawn/dawn_proc.h — monolithic webgpu_dawn handles procs internally
```

### Forward Declarations (use in headers instead of includes)

```cpp
struct WGPUInstanceImpl;        typedef WGPUInstanceImpl*        WGPUInstance;
struct WGPUAdapterImpl;         typedef WGPUAdapterImpl*         WGPUAdapter;
struct WGPUDeviceImpl;          typedef WGPUDeviceImpl*          WGPUDevice;
struct WGPUQueueImpl;           typedef WGPUQueueImpl*           WGPUQueue;
struct WGPUSurfaceImpl;         typedef WGPUSurfaceImpl*         WGPUSurface;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;
struct WGPUShaderModuleImpl;    typedef WGPUShaderModuleImpl*    WGPUShaderModule;
struct WGPUTextureImpl;         typedef WGPUTextureImpl*         WGPUTexture;
struct WGPUTextureViewImpl;     typedef WGPUTextureViewImpl*     WGPUTextureView;
struct WGPUSamplerImpl;         typedef WGPUSamplerImpl*         WGPUSampler;
struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl*  WGPUCommandEncoder;
```

## GPUCore Singleton

Entry point for all WebGPU access. Manages instance, adapter, device, and queue lifecycle.

```cpp
#include "core_gpu/gpu_core.h"
using namespace mps::gpu;

auto& gpu = GPUCore::GetInstance();
gpu.Initialize();                    // sync on native, async on WASM
while (!gpu.IsInitialized()) {       // poll for WASM
    gpu.ProcessEvents();
}
WGPUDevice device = gpu.GetDevice();
WGPUQueue queue = gpu.GetQueue();
gpu.Shutdown();
```

### Dual-Platform Async Pattern

- **Native**: `WGPUCallbackMode_WaitAnyOnly` + `wgpuInstanceWaitAny` (synchronous)
- **WASM**: `WGPUCallbackMode_AllowProcessEvents` + `ProcessEvents()` polling (async)

```cpp
WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
cb.callback = OnAdapterReceived;   // void(status, adapter, WGPUStringView, void*, void*)
cb.userdata1 = this;

WGPUFuture future = wgpuInstanceRequestAdapter(instance, &options, cb);

#ifndef __EMSCRIPTEN__
WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
wait.future = future;
wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
#endif
```

### Instance Creation

Must enable `TimedWaitAny` feature for native synchronous waits:

```cpp
WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
#ifndef __EMSCRIPTEN__
WGPUInstanceFeatureName features[] = { WGPUInstanceFeatureName_TimedWaitAny };
desc.requiredFeatureCount = 1;
desc.requiredFeatures = features;
#endif
WGPUInstance instance = wgpuCreateInstance(&desc);
```

## GPU Resource Classes

All resource classes follow the same RAII pattern:
- **Move-only** (deleted copy, destructor releases WGPU handle)
- **Direct construction** (no factory / `unique_ptr`)
- **Config structs** with sensible defaults
- **Throws `GPUException`** on creation failure

### GPUBuffer<T> (typed, resizable)

Split design: `GPUBufferCore` (non-template, WGPU calls in .cpp) + `GPUBuffer<T>` (template wrapper, header-only).

```cpp
// Typed vertex buffer
std::array verts = { vec3{0,0,0}, vec3{1,0,0}, vec3{0,1,0} };
GPUBuffer<vec3> vb(BufferUsage::Vertex, std::span(verts), "triangle_vb");

// Uniform buffer (raw config)
GPUBuffer<mat4> ub(BufferConfig{
    .usage = BufferUsage::Uniform | BufferUsage::CopyDst,
    .size = sizeof(mat4)
});

// Capacity management
vb.Reserve(100);     // pre-allocate for 100 elements
vb.Resize(50);       // grow + set logical size (preserves data)
vb.SetSize(200);     // no-copy resize (destroys old data)
vb.Clear();          // logical clear (size=0, keeps buffer)
vb.ShrinkToFit();    // trim capacity to match size

// GPU-to-GPU copy (cross-type allowed)
vb.CopyTo(dest_buffer);
vb.CopyTo(encoder, dest_buffer);  // batched via external encoder

// Readback
auto data = ub.ReadToHost();              // sync
ub.ReadToHostAsync([](auto result) {});   // async (fires during ProcessEvents)
```

Buffer internals: 1.5x geometric growth, 16-byte alignment, `CopySrc` auto-added for Grow support.

### GPUShader

```cpp
GPUShader shader(ShaderConfig{ .code = wgsl_source, .label = "triangle" });
WGPUShaderModule module = shader.GetHandle();
```

### GPUTexture

```cpp
GPUTexture tex(TextureConfig{
    .width = 256, .height = 256,
    .usage = TextureUsage::TextureBinding | TextureUsage::CopyDst
});
tex.WriteData(pixels.data(), pixels.size());
WGPUTextureView view = tex.GetView();  // default view, created at init
```

### GPUSampler

```cpp
GPUSampler sampler;  // default: linear filtering, clamp-to-edge
GPUSampler nearest(SamplerConfig{
    .mag_filter = FilterMode::Nearest,
    .min_filter = FilterMode::Nearest
});
```

## Surface Creation

Platform-specific, follows `_native`/`_wasm` file split.

```cpp
// Native — per-OS window handle
#ifdef _WIN32
    hwnd_source.hwnd = glfwGetWin32Window(glfw_window);     // WGPUSType_SurfaceSourceWindowsHWND
#elif defined(__linux__)
    x11_source.window = glfwGetX11Window(glfw_window);      // WGPUSType_SurfaceSourceXlibWindow
#elif defined(__APPLE__)
    metal_source.layer = GetMetalLayer(glfw_window);         // WGPUSType_SurfaceSourceMetalLayer
#endif

// WASM — HTML canvas selector (emdawnwebgpu API)
WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_source = {};
canvas_source.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
canvas_source.selector = {"#canvas", 7};
```

## Dawn API Notes

- Use `WGPU_<TYPE>_INIT` macros for struct initialization (e.g., `WGPU_INSTANCE_DESCRIPTOR_INIT`)
- Strings use `WGPUStringView{data, length}`, not `const char*`
- Adapter info: `wgpuAdapterGetInfo()` + `wgpuAdapterInfoFreeMembers()`
- `DAWN_FORCE_SYSTEM_COMPONENT_LOAD ON` in root CMakeLists.txt — loads system DLLs from System32

## Rules

- Forward-declare WebGPU types in headers; include `<webgpu/webgpu.h>` only in `.cpp`
- Dawn-specific: guard with `#ifndef __EMSCRIPTEN__`
- Never include `dawn/dawn_proc.h` — monolithic `webgpu_dawn` handles procs internally
- Surface creation: platform-specific per `_native`/`_wasm` split
- Shaders: string constants or file-loaded, not hardcoded inline
