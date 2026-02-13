# core_platform - Platform abstraction layer

# Source files (common)
set(CORE_PLATFORM_SOURCES
    window.cpp
    input.cpp
)

# Add platform-specific sources
if(EMSCRIPTEN)
    list(APPEND CORE_PLATFORM_SOURCES
        window_wasm.cpp
    )
else()
    list(APPEND CORE_PLATFORM_SOURCES
        window_native.cpp
    )
endif()

# Create core_platform library
add_library(core_platform STATIC
    ${CORE_PLATFORM_SOURCES}
)

# Set target properties
set_target_properties(core_platform PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_platform PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(core_platform PUBLIC
    mps::core_util
)

# Link platform-specific dependencies
if(EMSCRIPTEN)
    target_compile_options(core_platform PUBLIC --use-port=emdawnwebgpu)
    target_link_options(core_platform PUBLIC --use-port=emdawnwebgpu)
else()
    target_link_libraries(core_platform PUBLIC
        glfw
        webgpu_dawn
    )
endif()

# Create alias for namespace-style usage
add_library(mps::core_platform ALIAS core_platform)
