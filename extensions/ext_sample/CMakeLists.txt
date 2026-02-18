add_library(ext_sample STATIC
    sample_simulator.cpp
    sample_renderer.cpp
    sample_extension.cpp
)

set_target_properties(ext_sample PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ext_sample PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/extensions
)

target_link_libraries(ext_sample PUBLIC
    mps::core_system
)

add_library(mps::ext_sample ALIAS ext_sample)
