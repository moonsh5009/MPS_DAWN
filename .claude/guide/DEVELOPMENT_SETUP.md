# Development Environment Setup

This guide provides step-by-step instructions for setting up the development environment for the MPS_DAWN project.

## Prerequisites

- CMake 3.20 or higher
- C++20 compatible compiler
  - Windows: MSVC 2019/2022 or Clang
  - WASM: Emscripten SDK
- Git
- Python 3.x (for Dawn build scripts)

**Note**: Build tools like Ninja, depot_tools, and Emscripten SDK will be installed locally in the project directory. No system PATH modifications required.

---

## 1. C++20 Build Environment Setup

### 1.1 Create Hello World Test

#### Step 1: Create project structure
```bash
mkdir src
```

#### Step 2: Create main.cpp
Create `src/main.cpp`:
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

#### Step 3: Create CMakeLists.txt
Create root `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(MPS_DAWN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
```

Create `src/CMakeLists.txt`:
```cmake
# MPS_DAWN executable

# Source files
set(SOURCES
    main.cpp
)

# Create executable
add_executable(mps_dawn ${SOURCES})
```

#### Step 4: Build and test
```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Run
./bin/x64/Debug/mps_dawn  # Linux/Mac
# or
.\bin\x64\Debug\mps_dawn.exe  # Windows
```

Expected output: `Hello, World!`

**Note**: With the hierarchical structure and custom output directories, executables are placed in `build/bin/x64/Debug/` or `build/bin/x64/Release/` for better organization.

---

## 2. Dawn Library Installation

Dawn is Google's open-source implementation of WebGPU. We'll set it up for both Windows native and WASM builds.

### 2.1 Install Dawn Library for Windows

#### Step 1: Clone Dawn as a submodule
```bash
# From project root
git init  # If not already a git repository
git submodule add https://dawn.googlesource.com/dawn third_party/dawn
```

#### Step 2: Clone depot_tools (optional, for advanced builds)
```bash
# Clone depot_tools into third_party (from project root)
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git third_party/depot_tools

# depot_tools includes ninja and other build tools
# No PATH modification needed - we'll use relative paths
```

**Note**: For most cases, you can skip manual dependency syncing with gclient. CMake's `DAWN_FETCH_DEPENDENCIES=ON` option will automatically handle dependency management.

#### Step 3: Update CMakeLists.txt files for hierarchical structure
Update root `CMakeLists.txt` to handle project-wide configuration:
```cmake
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

# Add source directory
add_subdirectory(src)
```

Update `src/CMakeLists.txt` to handle executable, linking, and output directories:
```cmake
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
```

**Benefits of hierarchical structure**:
- Root CMakeLists.txt manages project-wide settings (C++ standard, compiler flags, third-party libraries)
- src/CMakeLists.txt focuses on source files and executable configuration
- Easier to maintain and extend as the project grows
- Clear separation of concerns
- Organized output directories:
  - Native builds:
    - Libraries (.lib): `build/lib/x64/Debug` and `build/lib/x64/Release`
    - Executables (.exe) and DLLs (.dll): `build/bin/x64/Debug` and `build/bin/x64/Release`
  - WASM builds:
    - Libraries (.a): `build/lib/wasm/Debug` and `build/lib/wasm/Release`
    - Executables (.wasm, .js, .html): `build/bin/wasm/Debug` and `build/bin/wasm/Release`
  - Clear platform and configuration separation

#### Step 4: Update main.cpp to test Dawn linkage
Update `src/main.cpp`:
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << "Dawn library linked successfully!" << std::endl;
    return 0;
}
```

### 2.2 Setup VS Code Debugging Environment

#### Step 1: Create VS Code settings
Create `.vscode/settings.json`:
```json
{
    "git.ignoreLimitWarning": true,
    "cmake.configureSettings": {
        "CMAKE_GENERATOR_PLATFORM": "x64"
    },
    "cmake.buildDirectory": "${workspaceFolder}/build/${buildType}",
    "cmake.preferredGenerators": [
        "Visual Studio 17 2022"
    ],
    "cmake.configureOnOpen": false,
    "files.associations": {
        "*.h": "cpp",
        "*.hpp": "cpp"
    }
}
```

#### Step 2: Create launch configurations
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(Windows) Debug",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/src/Debug/mps_dawn.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "console": "integratedTerminal",
            "preLaunchTask": "CMake: Build (Debug)"
        },
        {
            "name": "(Windows) Release",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/src/Release/mps_dawn.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "console": "integratedTerminal",
            "preLaunchTask": "CMake: Build (Release)"
        }
    ]
}
```

#### Step 3: Create build tasks
Create `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "CMake: Configure",
            "type": "shell",
            "command": "cmake",
            "args": [
                "..",
                "-A",
                "x64"
            ],
            "options": {
                "cwd": "${workspaceFolder}/build"
            },
            "problemMatcher": []
        },
        {
            "label": "CMake: Build (Debug)",
            "type": "shell",
            "command": "cmake",
            "args": [
                "--build",
                ".",
                "--config",
                "Debug"
            ],
            "options": {
                "cwd": "${workspaceFolder}/build"
            },
            "dependsOn": "CMake: Configure",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$msCompile"]
        },
        {
            "label": "CMake: Build (Release)",
            "type": "shell",
            "command": "cmake",
            "args": [
                "--build",
                ".",
                "--config",
                "Release"
            ],
            "options": {
                "cwd": "${workspaceFolder}/build"
            },
            "dependsOn": "CMake: Configure",
            "group": "build",
            "problemMatcher": ["$msCompile"]
        },
        {
            "label": "CMake: Clean",
            "type": "shell",
            "command": "cmake",
            "args": [
                "--build",
                ".",
                "--target",
                "clean"
            ],
            "options": {
                "cwd": "${workspaceFolder}/build"
            },
            "problemMatcher": []
        }
    ]
}
```

#### Step 4: Create IntelliSense configuration
Create `.vscode/c_cpp_properties.json`:
```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/third_party/dawn/include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.22621.0",
            "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++20",
            "intelliSenseMode": "windows-msvc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```

### 2.3 Verify Windows x64 Build

#### Step 1: Create build directory
```bash
# From project root
mkdir build
cd build
```

#### Step 2: Configure with CMake (x64 architecture)
```bash
# Windows - Visual Studio generator with x64 platform
cmake .. -A x64
```

**Important**: The `-A x64` flag specifies x64 architecture for Visual Studio generators.

#### Step 3: Build Debug configuration
```bash
cmake --build . --config Debug
```

This will:
- Download Dawn dependencies automatically (first time only, may take 10-15 minutes)
- Build Dawn libraries
- Build the mps_dawn executable

#### Step 4: Test Debug build
```bash
.\bin\x64\Debug\mps_dawn.exe
```

Expected output:
```
Hello, World!
Dawn library linked successfully!
```

#### Step 5: Build Release configuration
```bash
cmake --build . --config Release
```

#### Step 6: Test Release build
```bash
.\bin\x64\Release\mps_dawn.exe
```

Expected output: Same as Debug build.

---

### 2.4 Install Emscripten for WASM Build (Optional)

**Note**: Complete Windows build verification before attempting WASM build.

#### Step 1: Install Emscripten
```bash
# Clone Emscripten SDK into third_party (from project root)
git clone https://github.com/emscripten-core/emsdk.git third_party/emsdk
cd third_party/emsdk

# Install and activate latest using relative path
# Windows
.\emsdk.bat install latest
.\emsdk.bat activate latest

# Linux/Mac
./emsdk install latest
./emsdk activate latest

# Return to project root
cd ../..
```

**Note**: No need to modify PATH. We'll use relative paths to access Emscripten tools.

**WASM Configuration**: The WASM-specific configuration is already included in the hierarchical CMakeLists.txt structure created earlier:
- `CMAKE_EXECUTABLE_SUFFIX` set to ".html" for Emscripten
- Dawn backend flags (D3D12, Metal, Vulkan) disabled for WASM
- `--use-port=emdawnwebgpu` link option in `src/CMakeLists.txt` for browser WebGPU

### 2.5 Verify WASM Build (Optional)

#### Step 1: Create WASM build directory
```bash
# From project root
mkdir build-wasm
cd build-wasm
```

#### Step 2: Configure with Emscripten
```bash
# Windows - using default generator
..\third_party\emsdk\emcmake.bat cmake .. -DCMAKE_BUILD_TYPE=Release

# Linux/Mac - using default generator
../third_party/emsdk/emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Note**: Emscripten includes its own build tools, so depot_tools is not required for WASM builds.

#### Step 3: Build
```bash
cmake --build . --config Release
```

#### Step 4: Test in browser
```bash
# Serve the built files (run from build-wasm directory)
python -m http.server 8000

# Open browser to http://localhost:8000/mps_dawn.html
```

Expected result: HTML page loads and shows console output with "Hello, World!" and "Dawn library linked successfully!"

---

## Troubleshooting

### Common Issues

1. **CMake version too old**
   - Update CMake to 3.20 or higher

2. **Dawn submodule not initialized**
   ```bash
   git submodule update --init --recursive
   ```

3. **PDB file conflict errors during MSVC build**
   ```
   프로그램 데이터베이스 'xxx.pdb'을(를) 열 수 없습니다. 여러 CL.EXE에서 동일한 .PDB 파일에 쓰는 경우 /FS를 사용하십시오
   ```
   - **Solution**: Add `/FS` compiler flag to CMakeLists.txt:
   ```cmake
   if(MSVC)
       add_compile_options(/FS)
   endif()
   ```

4. **gclient sync fails on Windows**
   - Permission denied or file locking errors
   - "couldn't set 'HEAD'" errors
   - **Recommended solution**: Use `DAWN_FETCH_DEPENDENCIES=ON` in CMakeLists.txt instead of manual gclient sync
   - CMake will automatically download dependencies during configuration
   - More reliable on Windows and requires no manual intervention

5. **First CMake configure is very slow**
   - This is normal when `DAWN_FETCH_DEPENDENCIES=ON` is used
   - Dawn is downloading its dependencies (may take 10-15 minutes)
   - Subsequent configures will be much faster

6. **depot_tools gclient fails** (if manually syncing)
   - Ensure you're using the correct relative path: `..\depot_tools\gclient.bat` (Windows) or `../depot_tools/gclient` (Linux/Mac)
   - Check that depot_tools was cloned into `third_party/depot_tools`
   - Consider using `DAWN_FETCH_DEPENDENCIES=ON` instead

7. **Emscripten commands not found**
   - Use relative paths: `..\third_party\emsdk\emcmake.bat` (Windows) or `../third_party/emsdk/emcmake` (Linux/Mac)
   - Verify emsdk was cloned into `third_party/emsdk`
   - Run install and activate commands again from `third_party/emsdk` directory

8. **Build fails with C++20 errors**
   - Verify compiler supports C++20
   - Check `CMAKE_CXX_STANDARD` is set to 20

9. **Python not found during Dawn sync**
   - Install Python 3.x and ensure it's available in system PATH
   - depot_tools includes its own Python, but system Python may be needed for some scripts

10. **Ninja not found**
    - **For WASM builds**: Download Ninja separately
      ```bash
      # Download and extract to third_party/ninja/
      # Windows: ninja-win.zip from https://github.com/ninja-build/ninja/releases
      # Linux/Mac: ninja-linux.zip or ninja-mac.zip
      ```
    - Specify explicitly in CMake: `-DCMAKE_MAKE_PROGRAM=../third_party/ninja/ninja.exe` (Windows) or `-DCMAKE_MAKE_PROGRAM=../third_party/ninja/ninja` (Linux/Mac)
    - **For native Windows builds**: Use Visual Studio generator (default, no Ninja needed): `cmake .. -A x64`
    - depot_tools also includes ninja, but requires full gclient setup

11. **Wrong architecture (x86 instead of x64)**
    - Always specify `-A x64` when using Visual Studio generator
    - Verify in CMake output: `CMAKE_GENERATOR_PLATFORM` should be `x64`

### Debugging in VS Code

- Press `F5` to launch Debug configuration
- Use `Ctrl+Shift+B` to build without debugging
- Select configuration from Run and Debug panel:
  - `(Windows) Debug` - Debug build with debugging symbols
  - `(Windows) Release` - Optimized Release build

---

## 3. Project Organization

### 3.1 Directory Structure

The project follows a hierarchical structure:

```
MPS_DAWN/
├── .claude/                    # Claude Code prompts and guides
│   ├── guide/                  # Development guides
│   │   └── DEVELOPMENT_SETUP.md
│   └── prompts/                # Git workflow prompts
│       ├── git-branch.md
│       ├── git-commit.md
│       ├── git-pr.md
│       ├── git-push.md
│       ├── git-status.md
│       └── git-summary.md
├── .vscode/                    # VS Code configuration
│   ├── c_cpp_properties.json
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
├── build/                      # Native build output (gitignored)
│   ├── lib/                    # Library output directory
│   │   └── x64/                # x64 architecture libraries
│   │       ├── Debug/          # .lib files (Debug)
│   │       └── Release/        # .lib files (Release)
│   └── bin/                    # Executable output directory
│       └── x64/                # x64 architecture binaries
│           ├── Debug/          # .exe, .dll files (Debug)
│           └── Release/        # .exe, .dll files (Release)
├── build-wasm/                 # WASM build output (gitignored)
│   ├── lib/                    # WASM library output
│   │   └── wasm/
│   │       ├── Debug/          # .a files (Debug)
│   │       └── Release/        # .a files (Release)
│   └── bin/                    # WASM executable output directory
│       └── wasm/               # WASM binaries
│           ├── Debug/          # .wasm, .js, .html files (Debug)
│           └── Release/        # .wasm, .js, .html files (Release)
├── src/                        # Source code
│   ├── CMakeLists.txt         # Executable configuration
│   └── main.cpp               # Entry point
├── third_party/                # External dependencies
│   ├── dawn/                  # Dawn WebGPU library (submodule)
│   ├── depot_tools/           # Google build tools (optional, gitignored)
│   ├── emsdk/                 # Emscripten SDK (optional, gitignored)
│   └── ninja/                 # Ninja build tool (optional, gitignored)
├── .gitignore
├── .gitmodules
└── CMakeLists.txt             # Root project configuration
```

### 3.2 Git Workflow

The project is set up with git and includes helpful workflow prompts in `.claude/prompts/`:
- **git-commit.md**: Guidelines for creating commits
- **git-push.md**: Safe push practices with force push warnings
- **git-status.md**: Repository status checking
- **git-pr.md**: Pull request creation templates
- **git-branch.md**: Branch management and naming conventions
- **git-summary.md**: Changes summary for review/documentation

**Repository**: https://github.com/moonsh5009/MPS_DAWN

---

## Next Steps

After completing this setup:
1. **Verify all build configurations**
   - Windows x64 Debug ✓
   - Windows x64 Release ✓
   - WASM build ✓

2. **Create Architecture Design Guide**
   - Define module structure within src/
   - Establish code organization patterns
   - Plan compute shader integration
   - Design physics simulation components

3. **Implement WebGPU functionality**
   - Initialize WebGPU instance and device
   - Create rendering pipeline
   - Implement compute shaders for physics simulation
   - Add particle/fluid simulation logic

4. **Additional development**
   - Set up testing framework
   - Create cross-platform window management
   - Add performance profiling tools
   - Implement visualization for simulation results

---

## References

- [Dawn Documentation](https://dawn.googlesource.com/dawn)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [Emscripten Documentation](https://emscripten.org/docs/getting_started/index.html)
- [CMake Documentation](https://cmake.org/documentation/)
