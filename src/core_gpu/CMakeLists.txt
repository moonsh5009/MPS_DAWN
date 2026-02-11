# core_gpu - WebGPU abstraction layer

# Create core_gpu library
add_library(core_gpu STATIC
    # Source files will be added here
    # device.cpp
    # buffer.cpp
    # texture.cpp
    # pipeline.cpp
    # shader.cpp
    # command.cpp
    # sync.cpp
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

# Link Dawn WebGPU (only for native builds)
if(NOT EMSCRIPTEN)
    target_link_libraries(core_gpu PUBLIC
        webgpu_dawn
    )
endif()

# Create alias for namespace-style usage
add_library(mps::core_gpu ALIAS core_gpu)
