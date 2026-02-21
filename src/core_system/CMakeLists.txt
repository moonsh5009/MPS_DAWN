# core_system - System controller orchestrating database, simulate, render

add_library(core_system STATIC
    system.cpp
)

set_target_properties(core_system PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(core_system PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(core_system PUBLIC
    mps::core_util
    mps::core_platform
    mps::core_gpu
    mps::core_database
    mps::core_simulate
    mps::core_render
)

add_library(mps::core_system ALIAS core_system)
