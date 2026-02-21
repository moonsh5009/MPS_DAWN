cmake_minimum_required(VERSION 3.20)
project(MPS_DAWN CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Organize build outputs (native only — WASM uses per-target properties)
if(NOT EMSCRIPTEN)
    # Static libraries (.lib) -> lib/x64/Debug or lib/x64/Release
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/x64/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/x64/Release)
    # Executables (.exe) -> bin/x64/Debug or bin/x64/Release
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/x64/Debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/x64/Release)
endif()

# Compiler-specific settings
if(MSVC)
    add_compile_options(/FS)      # Fix PDB file conflicts during parallel builds
    add_compile_options(/utf-8)   # Source and execution charset UTF-8 (suppresses C4819)
    add_compile_options(/wd4100)  # Suppress unused parameter warnings
else()
    add_compile_options(-Wno-unused-parameter)
endif()

# WASM-specific configuration
if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    set(DAWN_ENABLE_D3D12 OFF)
    set(DAWN_ENABLE_METAL OFF)
    set(DAWN_ENABLE_VULKAN OFF)
    set(DAWN_ENABLE_DESKTOP_GL OFF)
    set(DAWN_ENABLE_OPENGLES OFF)

    # 64-bit WASM (Memory64): 8-byte pointers and size_t
    add_compile_options(-sMEMORY64)
    add_link_options(-sMEMORY64)

    # Memory: allow growth, 128MB initial, 4GB max
    add_link_options(-sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=134217728 -sMAXIMUM_MEMORY=4294967296)
endif()

# Dawn library configuration
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "Fetch Dawn dependencies" FORCE)
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "Build Dawn samples" FORCE)
set(DAWN_FORCE_SYSTEM_COMPONENT_LOAD ON CACHE BOOL "Load system DLLs from System32" FORCE)
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "Build Tint command-line tools" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "Build Tint tests" FORCE)

# Add Dawn library (skip for WASM - uses browser WebGPU)
if(NOT EMSCRIPTEN)
    add_subdirectory(third_party/dawn EXCLUDE_FROM_ALL)
endif()

# Add GLM library (header-only math library)
add_subdirectory(third_party/glm)

# GLFW is provided by Dawn (third_party/dawn/third_party/glfw)
# No separate add_subdirectory needed for native builds

# Add source directory
add_subdirectory(src)

# Add extensions
add_subdirectory(extensions)
