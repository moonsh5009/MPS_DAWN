# core_util

> Primitive types, math wrappers, logging, and timing utilities.

## Module Structure

```
src/core_util/
├── CMakeLists.txt    # STATIC library → mps::core_util (no dependencies)
├── types.h           # Primitive type aliases (uint32, float32, etc.) in mps namespace
├── math.h            # GLM wrappers (vec3, mat4, quat, constants, utility functions) in mps::util
├── logger.h / .cpp   # Logger singleton + LogDebug/Info/Warning/Error convenience functions
└── timer.h / .cpp    # Timer (start/stop/elapsed) + ScopedTimer (RAII profiler)
```

## Key Types

| Type | Header | Namespace | Description |
|------|--------|-----------|-------------|
| `uint8..uint64` | `types.h` | `mps` | Aliases for `std::uintN_t` |
| `int8..int64` | `types.h` | `mps` | Aliases for `std::intN_t` |
| `float32`, `float64` | `types.h` | `mps` | Aliases for `float` / `double` |
| `size_t` | `types.h` | `mps` | Alias for `std::size_t` |
| `byte` | `types.h` | `mps` | Alias for `std::byte` |
| `vec2/3/4` | `math.h` | `mps::util` | GLM float vector types |
| `ivec2/3/4` | `math.h` | `mps::util` | GLM signed int vector types |
| `uvec2/3/4` | `math.h` | `mps::util` | GLM unsigned int vector types |
| `mat2/3/4` | `math.h` | `mps::util` | GLM matrix types |
| `quat` | `math.h` | `mps::util` | GLM quaternion type |
| `PI`, `TWO_PI`, `HALF_PI`, `DEG_TO_RAD`, `RAD_TO_DEG` | `math.h` | `mps::util` | `constexpr float32` constants |
| `Logger` | `logger.h` | `mps::util` | Singleton, variadic `Debug/Info/Warning/Error(args...)` (fold expression) |
| `Timer` | `timer.h` | `mps::util` | `Start/Stop/Reset`, `GetElapsedSeconds/Milliseconds/Microseconds` |
| `ScopedTimer` | `timer.h` | `mps::util` | RAII — logs elapsed time on destruction |

## Free Functions

### math.h (`mps::util`)

```cpp
Radians(deg), Degrees(rad), Clamp(val, min, max), Lerp(a, b, t)
Length(vec3), Normalize(vec3), Dot(a, b), Cross(a, b)
Translate(m, v), Rotate(m, angle, axis), Scale(m, v)
LookAt(eye, center, up), Perspective(fovy, aspect, near, far), Ortho(l, r, b, t, n, f)
```

### logger.h (`mps::util`)

```cpp
LogDebug(...), LogInfo(...), LogWarning(...), LogError(...)
```

Convenience functions that delegate to Logger singleton. Variadic — concatenates all args.
