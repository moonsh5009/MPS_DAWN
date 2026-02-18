# Render Agent Memory

## Moved to core_gpu (no longer in core_render)
- BindGroupLayoutBuilder, BindGroupBuilder, PipelineLayoutBuilder -> core_gpu (namespace `mps::gpu`)
- SurfaceManager, ShaderLoader -> core_gpu (namespace `mps::gpu`)
- Enums ShaderStage, BindingType, VertexFormat, VertexStepMode, PrimitiveTopology -> core_gpu/gpu_types.h
- When using these from render code, qualify with `gpu::` prefix (e.g., `gpu::ShaderStage::Fragment`)

## Dawn WebGPU API Gotchas (Render-Specific)
- `depthWriteEnabled` is `WGPUOptionalBool`, not bool. Use `WGPUOptionalBool_True`/`WGPUOptionalBool_False` (NOT `WGPU_TRUE`/`WGPU_FALSE`)
- Use `WGPU_*_INIT` macros for all struct initialization (e.g., `WGPU_VERTEX_ATTRIBUTE_INIT`, `WGPU_VERTEX_BUFFER_LAYOUT_INIT`)
- `WGPUStringView` for labels: `{str.data(), str.size()}`
- Pipeline builders use rvalue-ref chaining: `Builder().Method1().Method2().Build()`
- Keep attribute/blend vectors alive (on stack) until wgpuDeviceCreateRenderPipeline returns

## Extension Interface
- `object_renderer.h` defines `IObjectRenderer` — extension rendering interface
- `IObjectRenderer` has NO dependency on database/simulate (core_render isolation preserved)
- Concrete renderers in extensions hold `System&` reference for `GetDeviceBuffer<T>()` access
- `GetOrder()` sorts renderers (lower = earlier, default 1000)
