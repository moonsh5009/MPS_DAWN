add_library(ext_chebyshev_pd STATIC
    pd_extension.cpp
    pd_system_simulator.cpp
    pd_dynamics.cpp
)

set_target_properties(ext_chebyshev_pd PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_chebyshev_pd PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_chebyshev_pd PUBLIC
    mps::core_system
    mps::ext_dynamics
    mps::ext_newton
    mps::ext_pd_term
)

add_library(mps::ext_chebyshev_pd ALIAS ext_chebyshev_pd)
