#pragma once

#include "core_util/types.h"
#include <vector>

namespace mps {
namespace database {

// Entity is a lightweight identifier (index into arrays)
using Entity = uint32;

// Sentinel value for invalid/null entities
inline constexpr Entity kInvalidEntity = UINT32_MAX;

// Manages entity creation, destruction, and recycling via free-list
class EntityManager {
public:
    EntityManager() = default;

    // Create a new entity (recycled from free-list if available)
    Entity Create();

    // Destroy an entity and add it to the free-list for recycling
    void Destroy(Entity entity);

    // Check if an entity is currently alive
    bool IsAlive(Entity entity) const;

    // Get the number of currently alive entities
    uint32 GetAliveCount() const;

private:
    // Tracks which entities are alive (indexed by entity id)
    std::vector<bool> alive_;

    // Free-list of recycled entity ids
    std::vector<Entity> free_list_;

    // Next fresh entity id (used when free-list is empty)
    Entity next_id_ = 0;
};

}  // namespace database
}  // namespace mps
