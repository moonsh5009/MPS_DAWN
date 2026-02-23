add_library(ext_jgs2 STATIC
    jgs2_extension.cpp
    jgs2_system_simulator.cpp
    jgs2_dynamics.cpp
    jgs2_spring_term.cpp
    jgs2_precompute.cpp
)

set_target_properties(ext_jgs2 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_jgs2 PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_jgs2 PUBLIC
    mps::core_system
    mps::ext_dynamics
)

add_library(mps::ext_jgs2 ALIAS ext_jgs2)
