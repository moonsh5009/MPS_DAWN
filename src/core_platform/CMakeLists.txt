# core_platform - Platform abstraction layer

# Create core_platform library
add_library(core_platform STATIC
    # Source files will be added here
    # window_native.cpp (for native builds)
    # window_wasm.cpp (for WASM builds)
    # input.cpp
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

# Create alias for namespace-style usage
add_library(mps::core_platform ALIAS core_platform)
