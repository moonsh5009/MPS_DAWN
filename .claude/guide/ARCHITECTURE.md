# MPS_DAWN Architecture Design Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Principles](#architecture-principles)
3. [Module Structure](#module-structure)
4. [Layer Architecture](#layer-architecture)
5. [Directory Organization](#directory-organization)
6. [WebGPU Integration Strategy](#webgpu-integration-strategy)
7. [Physics Simulation Components](#physics-simulation-components)
8. [Cross-Platform Strategy](#cross-platform-strategy)
9. [Code Organization Patterns](#code-organization-patterns)
10. [Build System Design](#build-system-design)
11. [Performance Considerations](#performance-considerations)
12. [Future Extensibility](#future-extensibility)

---

## Project Overview

### Purpose
MPS_DAWN is a high-performance **My(Moon) Physics Simulation** framework built on **WebGPU (Dawn)**.

### Key Features
- GPU-Accelerated Physics
- Cross-Platform (Native + WASM)
- Modern C++20
- WebGPU-First

### Target Use Cases
- Real-time particle simulations
- Fluid dynamics
- Soft body physics
- Custom physics simulations

---

## Architecture Principles

### 1. Separation of Concerns
(내용 추가 예정)

### 2. GPU-First Design
(내용 추가 예정)

### 3. Cross-Platform from Ground Up
(내용 추가 예정)

### 4. Performance by Default
(내용 추가 예정)

### 5. Testability and Maintainability
(내용 추가 예정)

---

## Module Structure

### Core Modules Overview

```
src/
├── core_util/          # 기본 유틸리티 (최하위 레이어, 의존성 없음)
├── core_gpu/           # WebGPU 추상화 레이어
├── core_platform/      # 플랫폼 추상화 (Window, Input)
├── core_database/      # DB 관리 시스템 + Transaction 처리
├── core_render/        # Rendering engine 핵심
├── core_simulate/      # Simulator 관리
├── shaders/            # WGSL compute and render shaders
└── main.cpp            # Application entry point
```

### 1. core_util (Foundation Layer)
**책임:** 기초 유틸리티 제공, 다른 모듈의 공통 의존성
**의존성:** 없음 (최하위 레이어)

```
core_util/
├── logger.h/cpp        # 로깅 시스템
├── timer.h/cpp         # 타이밍 및 프로파일링
├── math.h              # 수학 유틸리티 (vec2/3/4, mat3/4, 등)
├── types.h             # 공통 타입 정의 (uint32, float32, 등)
├── memory.h/cpp        # 메모리 풀 및 커스텀 할당자
└── hash.h              # 해싱 유틸리티
```

### 2. core_gpu (WebGPU Abstraction)
**책임:** WebGPU API를 추상화하고 사용하기 쉬운 인터페이스 제공
**의존성:** core_util

```
core_gpu/
├── device.h/cpp        # WebGPU 디바이스 관리 및 초기화
├── buffer.h/cpp        # 버퍼 래퍼 (Uniform, Storage, Vertex, Index)
├── texture.h/cpp       # 텍스처 및 샘플러 관리
├── pipeline.h/cpp      # Render/Compute 파이프라인 래퍼
├── shader.h/cpp        # 셰이더 모듈 로딩 및 컴파일
├── command.h/cpp       # 커맨드 인코더 유틸리티
└── sync.h/cpp          # 동기화 프리미티브 (fence, semaphore)
```

### 3. core_platform (Platform Abstraction)
**책임:** OS/브라우저 간 플랫폼 차이 추상화
**의존성:** core_util

```
core_platform/
├── window.h            # Window 인터페이스 (추상 클래스)
├── window_native.h/cpp # Native 구현 (GLFW, SDL, 또는 커스텀)
├── window_wasm.h/cpp   # WASM 구현 (HTML5 Canvas)
└── input.h/cpp         # 입력 처리 (키보드, 마우스)
```

### 4. core_database (Database & Transaction)
**책임:** 데이터 관리, 트랜잭션 처리, 쿼리 시스템
**의존성:** core_util

```
core_database/
├── database.h/cpp      # 데이터베이스 매니저
├── transaction.h/cpp   # 트랜잭션 처리 시스템
├── query.h/cpp         # 쿼리 인터페이스
├── storage.h/cpp       # 저장소 관리
└── cache.h/cpp         # 캐싱 시스템
```

### 5. core_render (Rendering Engine Core)
**책임:** 렌더링 파이프라인 및 시각화
**의존성:** core_util, core_gpu

```
core_render/
├── renderer.h/cpp      # 메인 렌더러
├── camera.h/cpp        # 카메라 시스템
├── material.h/cpp      # 머티리얼 시스템
└── (추가 예정)
```

### 6. core_simulate (Simulation Core)
**책임:** 물리 시뮬레이션 관리 및 스케줄링
**의존성:** core_util, core_gpu, core_database

```
core_simulate/
├── simulator.h         # Simulator 인터페이스 (추상 클래스)
├── manager.h/cpp       # Simulator 매니저
├── scheduler.h/cpp     # 시뮬레이션 스케줄링
└── (추가 예정)
```

### Module Dependency Graph

```
                     ┌──────────────┐
                     │  core_util   │  (최하위 - 의존성 없음)
                     └──────┬───────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │  core_gpu   │   │core_platform│   │core_database│
  └──────┬──────┘   └─────────────┘   └──────┬──────┘
         │                                    │
         ├────────────────┬───────────────────┤
         ▼                ▼                   ▼
  ┌─────────────┐   ┌─────────────────────────────┐
  │core_render  │   │      core_simulate          │
  └─────────────┘   └─────────────────────────────┘
```

**의존성 규칙:**
- 하위 레이어만 참조 가능 (상향 의존성 금지)
- 순환 의존성 절대 금지
- core_util은 완전히 독립적

---

## Layer Architecture

### Layering Diagram

```
┌─────────────────────────────────────────────────┐
│         Application Layer (main.cpp)            │  ← 최상위
│  - 애플리케이션 진입점 및 초기화                    │
│  - 전체 시스템 조율                                │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│      High-Level Systems Layer                   │
│  ┌──────────────────┐  ┌──────────────────┐    │
│  │ core_simulate    │  │  core_render     │    │
│  │ (시뮬레이션 관리)  │  │  (렌더링)         │    │
│  └──────────────────┘  └──────────────────┘    │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│         Core Services Layer                     │
│  ┌──────────────────┐  ┌──────────────────┐    │
│  │   core_gpu       │  │ core_database    │    │
│  │ (WebGPU 추상화)   │  │  (데이터 관리)    │    │
│  └──────────────────┘  └──────────────────┘    │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│         Platform Layer                          │
│              core_platform                      │
│    (Window, Input - OS/Browser 추상화)          │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│         Foundation Layer                        │
│              core_util                          │
│  (Logger, Timer, Math, Types - 기초 유틸리티)    │
└─────────────────────────────────────────────────┘
```

### Dependency Rules

#### 계층별 의존성 규칙
1. **상위 레이어 → 하위 레이어만 참조 가능**
   - Application Layer는 모든 레이어 사용 가능
   - High-Level Systems는 Core Services + Platform + Foundation 사용
   - Core Services는 Platform + Foundation 사용
   - Platform은 Foundation만 사용
   - Foundation은 외부 의존성 없음

2. **동일 레이어 내 모듈 간 참조**
   - 원칙적으로 금지 (결합도 최소화)
   - 필요시 명확한 인터페이스를 통해서만 참조
   - 예: core_simulate → core_database (허용, 명시적 의존성)

3. **순환 의존성 금지**
   - 어떤 경우에도 순환 참조 불가
   - 빌드 시스템에서 강제 검증

#### 레이어별 책임

| Layer | Modules | 책임 | 의존성 |
|-------|---------|------|--------|
| **Application** | main.cpp | 프로그램 진입점, 전체 초기화 | 모든 Core 모듈 |
| **High-Level Systems** | core_simulate<br>core_render | 시뮬레이션 로직<br>렌더링 파이프라인 | core_gpu, core_database,<br>core_platform, core_util |
| **Core Services** | core_gpu<br>core_database | WebGPU 추상화<br>데이터 관리 | core_util |
| **Platform** | core_platform | OS/브라우저 추상화 | core_util |
| **Foundation** | core_util | 기초 유틸리티 | 없음 (독립적) |

---

## Directory Organization

### Current Project Structure

```
MPS_DAWN/
├── .claude/                    # Claude Code configuration
│   ├── guide/                  # Architecture and setup guides
│   │   ├── ARCHITECTURE.md     # This file
│   │   └── DEVELOPMENT_SETUP.md
│   ├── instructions.md         # Project-wide instructions
│   └── memory/                 # Auto memory (persistent context)
│
├── .vscode/                    # VS Code configuration
│   ├── c_cpp_properties.json
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
│
├── src/                        # Source code
│   ├── core_util/              # ✓ Foundation utilities (완료)
│   │   ├── CMakeLists.txt
│   │   ├── types.h
│   │   ├── logger.h/cpp
│   │   ├── timer.h/cpp
│   │   └── math.h
│   │
│   ├── core_gpu/               # WebGPU abstraction (예정)
│   │   └── CMakeLists.txt
│   │
│   ├── core_platform/          # Platform abstraction (예정)
│   │   └── CMakeLists.txt
│   │
│   ├── core_database/          # Database & transaction (예정)
│   │   └── CMakeLists.txt
│   │
│   ├── core_render/            # Rendering engine (예정)
│   │   └── CMakeLists.txt
│   │
│   ├── core_simulate/          # Simulation management (예정)
│   │   └── CMakeLists.txt
│   │
│   ├── shaders/                # WGSL shaders (예정)
│   │
│   ├── CMakeLists.txt          # Source build configuration
│   └── main.cpp                # Application entry point
│
├── third_party/                # External dependencies
│   ├── dawn/                   # Dawn WebGPU (submodule)
│   ├── glm/                    # GLM math library (submodule)
│   ├── depot_tools/            # Google build tools (optional, gitignored)
│   └── emsdk/                  # Emscripten SDK (optional, gitignored)
│
├── build/                      # Native build output (gitignored)
│   ├── lib/x64/{Debug,Release}/    # Static libraries
│   └── bin/x64/{Debug,Release}/    # Executables
│
├── build-wasm/                 # WASM build output (gitignored)
│   ├── lib/wasm/{Debug,Release}/   # WASM libraries
│   └── bin/wasm/{Debug,Release}/   # WASM binaries
│
├── .gitignore
├── .gitmodules
└── CMakeLists.txt              # Root build configuration
```

### File Organization Rules

#### Header Files
- Use `#pragma once` for header guards
- Place in the same directory as implementation
- Include order: own header → C/C++ std → third-party → project headers

#### Source Files
- One class per file (generally)
- Match header file name exactly
- Keep implementation details private

#### CMake Files
- Each module has its own `CMakeLists.txt`
- Use `add_library()` with ALIAS for all modules
- Explicitly list source files (no GLOB)

---

## WebGPU Integration Strategy

### Initialization Flow
(내용 추가 예정)

### Resource Management Pattern
(내용 추가 예정)

### Compute Pipeline Pattern
(내용 추가 예정)

---

## Physics Simulation Components

### Particle System Design
(내용 추가 예정)

### Solver Interface
(내용 추가 예정)

### Spatial Hashing
(내용 추가 예정)

---

## Cross-Platform Strategy

### Platform Abstraction
(내용 추가 예정)

### WASM-Specific Considerations
(내용 추가 예정)

---

## Code Organization Patterns

### Naming Conventions

#### Files
- **Headers**: `snake_case.h`
- **Source**: `snake_case.cpp`
- **Shaders**: `snake_case.wgsl`

#### Code Elements
- **Classes/Structs**: `PascalCase`
- **Functions/Methods**: `PascalCase`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Member variables**: `snake_case_` (trailing underscore)
- **Namespaces**: `lowercase`

#### Prefixes
- **Interfaces**: `I` prefix (e.g., `IWindow`)
- **GPU types**: `GPU` prefix (e.g., `GPUBuffer`)

### Namespace Organization
```cpp
namespace mps {

// Foundation layer
namespace util {
    // logger, timer, math, types, memory, hash
}

// Platform layer
namespace platform {
    // window, input
}

// Core services
namespace gpu {
    // device, buffer, texture, pipeline, shader, command, sync
}

namespace database {
    // database, transaction, query, storage, cache
}

// High-level systems
namespace render {
    // renderer, camera, material
}

namespace simulate {
    // simulator, manager, scheduler
}

}  // namespace mps
```

**네임스페이스 규칙:**
- 모든 코드는 `mps` 네임스페이스 안에 위치
- 모듈명에서 `core_` 접두사 제거 (네임스페이스에서는 중복)
- 네임스페이스는 소문자 사용
- 중첩 네임스페이스 사용 가능 (예: `mps::gpu::internal`)

(추가 패턴은 개발 진행에 따라 작성)

---

## Build System Design

(내용 추가 예정)

---

## Performance Considerations

### GPU Optimization Strategies
(내용 추가 예정)

### CPU Optimization Strategies
(내용 추가 예정)

---

## Future Extensibility

### Planned Features
(내용 추가 예정)

### Extension Points
(내용 추가 예정)

---

## References

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Dawn Documentation](https://dawn.googlesource.com/dawn)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Status**: Work in Progress - 개발과 함께 내용을 채워나갈 예정
