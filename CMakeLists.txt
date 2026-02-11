cmake_minimum_required(VERSION 3.20)
project(MPS_DAWN CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Organize build outputs
# Static libraries (.lib, .a) -> lib/x64/Debug or lib/x64/Release
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/x64/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/x64/Release)
# DLLs (.dll) -> bin/x64/Debug or bin/x64/Release (same as executables)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/x64/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/x64/Release)
# Note: Object files (*.obj) in .dir folders are controlled by Visual Studio generator

# Compiler-specific settings
if(MSVC)
    # Fix PDB file conflicts during parallel builds
    add_compile_options(/FS)
endif()

# WASM-specific configuration
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    set(DAWN_ENABLE_D3D12 OFF)
    set(DAWN_ENABLE_METAL OFF)
    set(DAWN_ENABLE_VULKAN OFF)
    set(DAWN_ENABLE_DESKTOP_GL OFF)
    set(DAWN_ENABLE_OPENGLES OFF)
endif()

# Dawn library configuration
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "Fetch Dawn dependencies" FORCE)
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "Build Dawn samples" FORCE)
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "Build Tint command-line tools" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "Build Tint tests" FORCE)

# Add Dawn library (skip for WASM - uses browser WebGPU)
if(NOT EMSCRIPTEN)
    add_subdirectory(third_party/dawn EXCLUDE_FROM_ALL)
endif()

# Add GLM library (header-only math library)
add_subdirectory(third_party/glm)

# Add source directory
add_subdirectory(src)
