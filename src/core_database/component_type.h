#pragma once

#include "core_util/types.h"
#include <atomic>
#include <concepts>
#include <type_traits>

namespace mps {
namespace database {

// Unique identifier for each component type
using ComponentTypeId = uint32;

// Invalid component type sentinel
inline constexpr ComponentTypeId kInvalidComponentTypeId = UINT32_MAX;

// Concept: a valid ECS component must be trivially copyable and standard layout
template<typename T>
concept Component = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

// Returns a unique ComponentTypeId for each type T (thread-safe, monotonic counter)
template<Component T>
ComponentTypeId GetComponentTypeId() {
    static const ComponentTypeId id = [] {
        static std::atomic<ComponentTypeId> counter{0};
        return counter.fetch_add(1);
    }();
    return id;
}

}  // namespace database
}  // namespace mps
