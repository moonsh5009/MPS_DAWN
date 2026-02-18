add_library(ext_cloth STATIC
    cloth_mesh.cpp
    cloth_simulator.cpp
    cloth_renderer.cpp
    cloth_extension.cpp
)

set_target_properties(ext_cloth PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_cloth PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_cloth PUBLIC
    mps::core_system
)

add_library(mps::ext_cloth ALIAS ext_cloth)
