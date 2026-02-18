# core_render - Rendering engine core
# Currently a stub (no source files yet) — uses INTERFACE library

add_library(core_render INTERFACE)

target_include_directories(core_render INTERFACE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(core_render INTERFACE
    mps::core_util
    mps::core_gpu
)

add_library(mps::core_render ALIAS core_render)
