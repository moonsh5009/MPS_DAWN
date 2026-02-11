# core_util - Foundation layer utilities

# Create core_util library
add_library(core_util STATIC
    logger.cpp
    timer.cpp
)

# Set target properties
set_target_properties(core_util PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_util PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/glm
)

# Create alias for namespace-style usage
add_library(mps::core_util ALIAS core_util)
