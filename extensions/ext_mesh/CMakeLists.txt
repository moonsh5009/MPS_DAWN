add_library(ext_mesh STATIC
    mesh_extension.cpp
    mesh_generator.cpp
    mesh_post_processor.cpp
    mesh_renderer.cpp
    normal_computer.cpp
)

set_target_properties(ext_mesh PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_mesh PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_mesh PUBLIC
    mps::core_system
)

add_library(mps::ext_mesh ALIAS ext_mesh)
