# core_simulate - Simulation management and scheduling

# Create core_simulate library
add_library(core_simulate STATIC
    # Source files will be added here
    # manager.cpp
    # scheduler.cpp
)

# Set target properties
set_target_properties(core_simulate PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_simulate PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(core_simulate PUBLIC
    mps::core_util
    mps::core_gpu
    mps::core_database
)

# Create alias for namespace-style usage
add_library(mps::core_simulate ALIAS core_simulate)
