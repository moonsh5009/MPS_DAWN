#pragma once

#include <cstdint>
#include <cstddef>

namespace mps {

// Unsigned integer types
using uint8  = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

// Signed integer types
using int8  = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

// Floating-point types
using float32 = float;
using float64 = double;

// Size types
using size_t = std::size_t;

// Byte type
using byte = std::byte;

namespace util {

// Re-export into util for backward compatibility
using mps::uint8;
using mps::uint16;
using mps::uint32;
using mps::uint64;
using mps::int8;
using mps::int16;
using mps::int32;
using mps::int64;
using mps::float32;
using mps::float64;
using mps::size_t;
using mps::byte;

}  // namespace util
}  // namespace mps
