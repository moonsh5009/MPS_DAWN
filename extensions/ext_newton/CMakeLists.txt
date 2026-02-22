add_library(ext_newton STATIC
    newton_extension.cpp
    newton_system_simulator.cpp
    newton_dynamics.cpp
    spring_term.cpp
    spring_term_provider.cpp
    area_term.cpp
    area_term_provider.cpp
)

set_target_properties(ext_newton PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_newton PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_newton PUBLIC
    mps::core_system
    mps::ext_dynamics
)

add_library(mps::ext_newton ALIAS ext_newton)
