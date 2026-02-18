#include "core_database/entity.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::database;
using namespace mps::util;

Entity EntityManager::Create() {
    Entity id;
    if (!free_list_.empty()) {
        id = free_list_.back();
        free_list_.pop_back();
    } else {
        id = next_id_++;
        alive_.push_back(false);
    }
    alive_[id] = true;
    return id;
}

void EntityManager::Destroy(Entity entity) {
    if (entity >= alive_.size() || !alive_[entity]) {
        LogError("EntityManager::Destroy — invalid or already-dead entity ", entity);
        return;
    }
    alive_[entity] = false;
    free_list_.push_back(entity);
}

bool EntityManager::IsAlive(Entity entity) const {
    if (entity >= alive_.size()) {
        return false;
    }
    return alive_[entity];
}

uint32 EntityManager::GetAliveCount() const {
    uint32 count = 0;
    for (auto alive : alive_) {
        if (alive) {
            ++count;
        }
    }
    return count;
}
