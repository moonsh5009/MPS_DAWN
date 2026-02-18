# Render Agent Memory

## Module Structure
- `core_render` is a STATIC library with alias `mps::core_render`
- Namespace: `mps::render`
- Dependencies: `mps::core_util`, `mps::core_gpu`, `mps::core_platform`
- Source files organized in subdirectories: pipeline/, camera/, uniform/, pass/, target/, geometry/, post/

## Moved to core_gpu (no longer in core_render)
- BindGroupLayoutBuilder, BindGroupBuilder, PipelineLayoutBuilder -> core_gpu (namespace `mps::gpu`)
- SurfaceManager, ShaderLoader -> core_gpu (namespace `mps::gpu`)
- Enums ShaderStage, BindingType, VertexFormat, VertexStepMode, PrimitiveTopology -> core_gpu/gpu_types.h
- When using these from render code, qualify with `gpu::` prefix (e.g., `gpu::ShaderStage::Fragment`)

## Implemented Files (core_render)
- render_types.h: Forward-declared WebGPU handles, render-specific enums (CullMode, FrontFace, LoadOp, StoreOp, BlendFactor, BlendOp), BlendState/ClearColor structs. Includes core_gpu/gpu_types.h for moved enums.
- render_engine.h/.cpp: Frame lifecycle, sub-system orchestration. Uses gpu::SurfaceManager (not render::).
- pipeline/render_pipeline_builder.h/.cpp: Full render pipeline builder. Uses gpu::VertexFormat, gpu::VertexStepMode, gpu::PrimitiveTopology.
- camera/camera.h/.cpp: Orbit camera (yaw/pitch/distance model)
- camera/camera_controller.h/.cpp: Mouse input -> camera orbit/pan/zoom
- uniform/camera_uniform.h/.cpp: Camera UBO (view/proj matrices, position, viewport, frustum)
- uniform/light_uniform.h/.cpp: Directional light UBO (direction, ambient, diffuse, specular)
- post/fxaa_pass.h/.cpp: FXAA post-processing. Uses gpu:: builders and ShaderLoader.
- post/wboit_pass.h/.cpp: WBOIT transparency compositing. Uses gpu:: builders and ShaderLoader.

## Stub Files (Not Yet Implemented)
- pass/render_pass_builder.cpp, pass/render_encoder.cpp
- target/render_target.cpp
- geometry/draw_command.cpp
- post/fullscreen_quad.cpp

## Dawn WebGPU API Gotchas (Render-Specific)
- `depthWriteEnabled` is `WGPUOptionalBool`, not bool. Use `WGPUOptionalBool_True`/`WGPUOptionalBool_False` (NOT `WGPU_TRUE`/`WGPU_FALSE`)
- Use `WGPU_*_INIT` macros for all struct initialization (e.g., `WGPU_VERTEX_ATTRIBUTE_INIT`, `WGPU_VERTEX_BUFFER_LAYOUT_INIT`)
- `WGPUStringView` for labels: `{str.data(), str.size()}`
- Pipeline builders use rvalue-ref chaining: `Builder().Method1().Method2().Build()`
- Keep attribute/blend vectors alive (on stack) until wgpuDeviceCreateRenderPipeline returns

## Conventions
- Forward-declare WebGPU types in headers (no webgpu.h includes)
- Include webgpu.h only in .cpp files
- `using namespace mps::util;` OK in .cpp files
- Builder pattern uses `&&`-qualified methods returning `std::move(*this)`
- Types from core_gpu must be qualified with `gpu::` in render code (e.g., gpu::PrimitiveTopology)
