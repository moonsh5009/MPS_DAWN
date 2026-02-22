add_library(ext_dynamics STATIC
    dynamics_extension.cpp
    constraint_builder.cpp
)

set_target_properties(ext_dynamics PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_dynamics PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_dynamics PUBLIC
    mps::core_system
)

add_library(mps::ext_dynamics ALIAS ext_dynamics)
