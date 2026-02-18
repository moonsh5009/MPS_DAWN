# Add core modules
add_subdirectory(core_util)
add_subdirectory(core_platform)
add_subdirectory(core_gpu)
add_subdirectory(core_database)
add_subdirectory(core_simulate)
add_subdirectory(core_render)
add_subdirectory(core_system)

# MPS_DAWN executable

# Source files
set(SOURCES
    main.cpp
)

# Create executable
add_executable(mps_dawn ${SOURCES})

# Set output directories based on platform
if(NOT EMSCRIPTEN)
    # Native build: bin/x64/Debug or bin/x64/Release
    set_target_properties(mps_dawn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/x64/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/x64/Release"
    )
else()
    # WASM build: bin/wasm/Debug or bin/wasm/Release
    set_target_properties(mps_dawn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/wasm/${CMAKE_BUILD_TYPE}"
    )
endif()

# Link libraries (webgpu_dawn propagates transitively through core_gpu)
target_link_libraries(mps_dawn PRIVATE
    mps::core_util
    mps::core_platform
    mps::core_gpu
    mps::core_render
    mps::core_system
    mps::ext_cloth
)

# Copy shaders to output directory (next to executable)
add_custom_command(TARGET mps_dawn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:mps_dawn>/shaders
    COMMENT "Copying shaders to output directory"
)
