# MPS_DAWN executable

# Source files
set(SOURCES
    main.cpp
    # Add more source files here as the project grows
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

# Link libraries
if(NOT EMSCRIPTEN)
    # Native build: Link Dawn WebGPU library
    target_link_libraries(mps_dawn PRIVATE
        webgpu_dawn
    )
else()
    # WASM build: Use Emscripten's WebGPU port
    target_link_options(mps_dawn PRIVATE
        --use-port=emdawnwebgpu
    )
endif()

# Include directories (if needed in the future)
# target_include_directories(mps_dawn PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
