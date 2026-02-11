# core_render - Rendering engine core

# Create core_render library
add_library(core_render STATIC
    # Source files will be added here
    # renderer.cpp
    # camera.cpp
    # material.cpp
)

# Set target properties
set_target_properties(core_render PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_render PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(core_render PUBLIC
    mps::core_util
    mps::core_gpu
)

# Create alias for namespace-style usage
add_library(mps::core_render ALIAS core_render)
