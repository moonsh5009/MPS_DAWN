add_library(ext_pd STATIC
    pd_extension.cpp
    pd_system_simulator.cpp
    pd_dynamics.cpp
    pd_spring_term.cpp
    pd_spring_term_provider.cpp
    pd_area_term.cpp
    pd_area_term_provider.cpp
)

set_target_properties(ext_pd PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_pd PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_pd PUBLIC
    mps::core_system
    mps::ext_dynamics
    mps::ext_newton
)

add_library(mps::ext_pd ALIAS ext_pd)
