add_library(ext_admm_pd STATIC
    admm_extension.cpp
    admm_system_simulator.cpp
    admm_dynamics.cpp
)

set_target_properties(ext_admm_pd PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_admm_pd PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_admm_pd PUBLIC
    mps::core_system
    mps::ext_dynamics
    mps::ext_newton
    mps::ext_pd_term
)

add_library(mps::ext_admm_pd ALIAS ext_admm_pd)
