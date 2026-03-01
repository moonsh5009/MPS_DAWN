add_library(ext_avbd STATIC
    avbd_extension.cpp
    avbd_system_simulator.cpp
    vbd_dynamics.cpp
    avbd_spring_term.cpp
    avbd_area_term.cpp
    graph_coloring.cpp
)

set_target_properties(ext_avbd PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_avbd PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_avbd PUBLIC
    mps::core_system
    mps::ext_dynamics
    mps::ext_mesh
)

add_library(mps::ext_avbd ALIAS ext_avbd)
