# core_gpu

> WebGPU abstraction — device lifecycle, buffers, shaders, textures, builders, and RAII handles.

## Module Structure

```
src/core_gpu/
├── CMakeLists.txt
├── gpu_types.h                     # Shared enums (BufferUsage, TextureFormat, ShaderStage, BindingType, ...) + GPUException
├── gpu_core.h / gpu_core.cpp       # GPUCore singleton (instance, adapter, device, queue, surface creation)
├── gpu_buffer.h / gpu_buffer.cpp   # GPUBufferCore + GPUBuffer<T> (typed, resizable)
├── gpu_shader.h / gpu_shader.cpp   # GPUShader (WGSL shader module)
├── gpu_texture.h / gpu_texture.cpp # GPUTexture (2D/3D + default view)
├── gpu_sampler.h / gpu_sampler.cpp # GPUSampler (filtering, addressing)
├── gpu_handle.h / gpu_handle.cpp   # GPUHandle<Tag> RAII template (5 WebGPU handle types)
├── bind_group_layout_builder.h/cpp # Fluent WGPUBindGroupLayout builder
├── bind_group_builder.h/cpp        # Fluent WGPUBindGroup builder (buffer/texture/sampler)
├── pipeline_layout_builder.h/cpp   # Fluent WGPUPipelineLayout builder
├── compute_pipeline_builder.h/cpp  # Fluent WGPUComputePipeline builder
├── compute_encoder.h/cpp           # Compute pass encoding utilities
├── surface_manager.h/cpp           # Surface configure/present/acquire/resize
├── shader_loader.h/cpp             # WGSL file loader with #import directive support
└── asset_path.h/cpp                # General asset path resolver (assets/ directory lookup)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `GPUCore` | `gpu_core.h` | Singleton: instance, adapter, device, queue lifecycle |
| `GPUBuffer<T>` | `gpu_buffer.h` | Typed, resizable GPU buffer with RAII |
| `GPUBufferCore` | `gpu_buffer.h` | Non-template base (WGPU calls in .cpp) |
| `GPUShader` | `gpu_shader.h` | WGSL shader module wrapper |
| `GPUTexture` | `gpu_texture.h` | 2D/3D texture with default view |
| `GPUSampler` | `gpu_sampler.h` | Sampler with filtering/addressing config |
| `GPUHandle<Tag>` | `gpu_handle.h` | RAII wrapper for WebGPU handle types (see below) |
| `GPUException` | `gpu_types.h` | Exception for GPU resource creation failures |
| `BindGroupLayoutBuilder` | `bind_group_layout_builder.h` | Fluent bind group layout builder |
| `BindGroupBuilder` | `bind_group_builder.h` | Fluent bind group builder |
| `PipelineLayoutBuilder` | `pipeline_layout_builder.h` | Fluent pipeline layout builder |
| `ComputePipelineBuilder` | `compute_pipeline_builder.h` | Fluent compute pipeline builder |
| `ComputeEncoder` | `compute_encoder.h` | Compute pass dispatch wrapper |
| `SurfaceManager` | `surface_manager.h` | Swap chain surface config/present/acquire/resize |
| `ShaderLoader` | `shader_loader.h` | WGSL file loader with `#import` support |
| `ResolveAssetPath` | `asset_path.h` | Free function: resolves paths relative to `assets/` directory |

### GPUHandle<Tag> Type Aliases

| Type Alias | Tag | Wrapped Handle |
|------------|-----|---------------|
| `GPUComputePipeline` | `ComputePipelineTag` | `WGPUComputePipeline` |
| `GPURenderPipeline` | `RenderPipelineTag` | `WGPURenderPipeline` |
| `GPUBindGroup` | `BindGroupTag` | `WGPUBindGroup` |
| `GPUBindGroupLayout` | `BindGroupLayoutTag` | `WGPUBindGroupLayout` |
| `GPUPipelineLayout` | `PipelineLayoutTag` | `WGPUPipelineLayout` |

API: `GetHandle()`, `IsValid()`, `explicit operator bool()`, `Detach()`. All builders return wrapped types.

### gpu_types.h Enums

- `BufferUsage` — flags: `Vertex`, `Index`, `Uniform`, `Storage`, `CopySrc`, `CopyDst`, `MapRead`, `MapWrite`, `None`
- `TextureUsage` — `RenderAttachment`, `TextureBinding`, `CopyDst`, `CopySrc`
- `TextureFormat` — `RGBA8Unorm`, `BGRA8Unorm`, `Depth24Plus`, `Depth32Float`, etc.
- `ShaderStage` — `Vertex`, `Fragment`, `Compute`
- `BindingType` — `UniformBuffer`, `StorageBuffer`, `ReadOnlyStorageBuffer`, `Texture`, `Sampler`
- `VertexFormat` — `Float32`, `Float32x2`, `Float32x3`, `Float32x4`, `Uint32`, `Sint32`, etc.
- `VertexStepMode` — `Vertex`, `Instance`
- `PrimitiveTopology` — `PointList`, `LineList`, `LineStrip`, `TriangleList`, `TriangleStrip`
- `FilterMode` — `Nearest`, `Linear`
- `AddressMode` — `ClampToEdge`, `Repeat`, `MirrorRepeat`

## API

### GPUCore

```cpp
static GPUCore& GetInstance();

WGPUSurface CreateSurface(void* native_window, void* native_display = nullptr);
bool Initialize(const GPUConfig& config = {}, WGPUSurface compatible_surface = nullptr);
bool IsInitialized() const;
GPUState GetState() const;
void ProcessEvents();
void Shutdown();

WGPUInstance GetWGPUInstance() const;
WGPUAdapter GetAdapter() const;
WGPUDevice GetDevice() const;
WGPUQueue GetQueue() const;
std::string GetAdapterName() const;
std::string GetBackendType() const;
```

### GPUBuffer<T>

```cpp
GPUBuffer(BufferUsage usage, std::span<const T> data, const std::string& label = "");
GPUBuffer(const BufferConfig& config);

WGPUBuffer GetHandle() const;
uint64 GetSize() const;           // Element count (current)
uint64 GetByteLength() const;     // Total bytes (current size * sizeof(T))
uint64 GetCount() const;          // Element count (alias for GetSize)
uint64 GetCapacity() const;       // Element capacity (allocated)

void WriteData(std::span<const T> data, uint64 offset = 0);
void Reserve(uint64 count);
void Resize(uint64 count);
void SetSize(uint64 count);
void Clear();
void ShrinkToFit();
void CopyTo(GPUBufferCore& dest);
void CopyTo(WGPUCommandEncoder encoder, GPUBufferCore& dest);
std::vector<T> ReadToHost();
void ReadToHostAsync(std::function<void(std::vector<T>)> callback);
```

### Builders (fluent rvalue-reference API)

```cpp
// BindGroupLayoutBuilder
BindGroupLayoutBuilder(const std::string& label = "")
    .AddBinding(binding, stage, type)
    .AddUniformBinding(binding, stage)
    .AddStorageBinding(binding, stage)
    .AddReadOnlyStorageBinding(binding, stage)
    .AddTextureBinding(binding, stage)
    .AddSamplerBinding(binding, stage)
    .Build()  // → GPUBindGroupLayout

// BindGroupBuilder
BindGroupBuilder(const std::string& label = "")
    .AddBuffer(binding, buffer, size, offset)
    .AddTextureView(binding, view)
    .AddSampler(binding, sampler)
    .Build(layout)  // → GPUBindGroup

// PipelineLayoutBuilder
PipelineLayoutBuilder(const std::string& label = "")
    .AddBindGroupLayout(layout)
    .Build()  // → GPUPipelineLayout

// ComputePipelineBuilder
ComputePipelineBuilder(const std::string& label = "")
    .SetPipelineLayout(layout)
    .SetComputeShader(module, entry)
    .Build()  // → GPUComputePipeline
```

### ComputeEncoder

```cpp
ComputeEncoder(WGPUComputePassEncoder pass);
void SetPipeline(pipeline);
void SetBindGroup(index, bind_group);
void Dispatch(workgroup_count);
void DispatchIndirect(indirect_buffer);
```

### SurfaceManager

```cpp
void Initialize(WGPUSurface surface, const SurfaceConfig& config);
WGPUTextureView AcquireNextFrameView();
void Present();
void Resize(uint32 width, uint32 height);
```

### ResolveAssetPath

```cpp
// Resolve a path relative to the assets/ directory.
// Searches: CWD/assets/, CWD/../assets/, exe_dir/assets/ (Windows).
// Example: ResolveAssetPath("objs/cube.obj") → "<base>/assets/objs/cube.obj"
std::string ResolveAssetPath(const std::string& relative_path);
```

### ShaderLoader

Uses `ResolveAssetPath("shaders/")` internally for base path resolution.

```cpp
static GPUShader CreateModule(const std::string& path, const std::string& label = "");
```
