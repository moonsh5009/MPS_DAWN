# core_gpu - WebGPU abstraction layer

# Create core_gpu library
add_library(core_gpu STATIC
    gpu_core.cpp
    gpu_buffer.cpp
    gpu_shader.cpp
    gpu_texture.cpp
    gpu_sampler.cpp
    gpu_handle.cpp
    bind_group_layout_builder.cpp
    bind_group_builder.cpp
    pipeline_layout_builder.cpp
    surface_manager.cpp
    shader_loader.cpp
    asset_path.cpp
    compute_pipeline_builder.cpp
    compute_encoder.cpp
)

# Set target properties
set_target_properties(core_gpu PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_gpu PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(core_gpu PUBLIC
    mps::core_util
)

# Platform-specific WebGPU linkage
if(NOT EMSCRIPTEN)
    target_link_libraries(core_gpu PUBLIC
        webgpu_dawn
    )
else()
    target_compile_options(core_gpu PUBLIC --use-port=emdawnwebgpu)
    target_link_options(core_gpu PUBLIC --use-port=emdawnwebgpu)
endif()

# Create alias for namespace-style usage
add_library(mps::core_gpu ALIAS core_gpu)
