---
name: type-ref
description: Type system and math reference for MPS_DAWN — primitives, math types, constants, utility functions
---

# Type System Reference

Always use types from `core_util/types.h` and `core_util/math.h`. Never use raw `int`/`float`/`size_t`.

## Primitives (`core_util/types.h`)

| Category | Types |
|----------|-------|
| Unsigned int | `util::uint8`, `uint16`, `uint32`, `uint64` |
| Signed int | `util::int8`, `int16`, `int32`, `int64` |
| Float | `util::float32`, `float64` |
| Other | `util::size_t`, `util::byte` |

## Math Types (`core_util/math.h`, GLM-based)

| Category | Types |
|----------|-------|
| Vectors | `util::vec2/3/4`, `ivec2/3/4`, `uvec2/3/4` |
| Matrices | `util::mat2/3/4` |
| Quaternion | `util::quat` |
| Constants | `PI`, `TWO_PI`, `HALF_PI`, `DEG_TO_RAD`, `RAD_TO_DEG` |

## Math Utility Functions (`core_util/math.h`)

| Category | Functions |
|----------|-----------|
| Conversion | `Radians()`, `Degrees()` |
| Basic | `Clamp()`, `Lerp()` |
| Vector | `Length()`, `Normalize()`, `Dot()`, `Cross()` |
| Matrix | `Translate()`, `Rotate()`, `Scale()`, `LookAt()`, `Perspective()`, `Ortho()` |

## Usage Examples

```cpp
#include "core_util/types.h"
#include "core_util/math.h"

using namespace mps::util;  // OK in .cpp files

uint32 count = 0;
float32 delta = 0.016f;
vec3 position{0.0f, 1.0f, 0.0f};
mat4 view = LookAt(vec3{0, 0, 5}, vec3{0, 0, 0}, vec3{0, 1, 0});
float32 angle = Radians(90.0f);
```
