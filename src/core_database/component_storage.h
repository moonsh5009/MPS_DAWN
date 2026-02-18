#pragma once

#include "core_database/component_type.h"
#include "core_database/entity.h"
#include "core_util/logger.h"
#include <vector>

namespace mps {
namespace database {

// Type-erased interface for component storage (used by Database to iterate storages)
class IComponentStorage {
public:
    virtual ~IComponentStorage() = default;

    // Access the contiguous dense data array (type-erased)
    virtual const void* GetDenseData() const = 0;

    // Total byte size of the dense data array
    virtual uint64 GetDenseDataSizeBytes() const = 0;

    // Number of components currently stored
    virtual uint32 GetDenseCount() const = 0;

    // Whether any component was added, removed, or modified since last ClearDirty
    virtual bool IsDirty() const = 0;

    // Reset the dirty flag
    virtual void ClearDirty() = 0;

    // Remove a component by entity (type-erased, for entity destruction)
    virtual void RemoveByEntity(Entity entity) = 0;

    // Check if entity has a component in this storage
    virtual bool Contains(Entity entity) const = 0;
};

// Sparse-set based component storage for a specific component type T.
// Provides O(1) add, remove (swap-and-pop), get, and has operations.
// Dense array is contiguous for cache-friendly iteration.
template<Component T>
class ComponentStorage : public IComponentStorage {
public:
    ComponentStorage() = default;

    // Add a component for an entity. Returns false if entity already has one.
    bool Add(Entity entity, const T& component) {
        if (Contains(entity)) {
            mps::util::LogError("ComponentStorage::Add — entity ", entity, " already has component");
            return false;
        }
        EnsureSparseSize(entity);
        sparse_[entity] = static_cast<uint32>(dense_.size());
        dense_.push_back(component);
        dense_to_entity_.push_back(entity);
        dirty_ = true;
        return true;
    }

    // Remove a component from an entity using swap-and-pop. Returns false if not found.
    bool Remove(Entity entity) {
        if (!Contains(entity)) {
            return false;
        }
        uint32 index = sparse_[entity];
        uint32 last = static_cast<uint32>(dense_.size()) - 1;

        if (index != last) {
            // Swap the removed element with the last element
            Entity last_entity = dense_to_entity_[last];
            dense_[index] = dense_[last];
            dense_to_entity_[index] = last_entity;
            sparse_[last_entity] = index;
        }

        dense_.pop_back();
        dense_to_entity_.pop_back();
        sparse_[entity] = kInvalidEntity;
        dirty_ = true;
        return true;
    }

    // Set (overwrite) a component value for an entity. Returns false if not found.
    bool Set(Entity entity, const T& component) {
        if (!Contains(entity)) {
            return false;
        }
        dense_[sparse_[entity]] = component;
        dirty_ = true;
        return true;
    }

    // Get a pointer to the component for an entity. Returns nullptr if not found.
    T* Get(Entity entity) {
        if (!Contains(entity)) {
            return nullptr;
        }
        return &dense_[sparse_[entity]];
    }

    const T* Get(Entity entity) const {
        if (!Contains(entity)) {
            return nullptr;
        }
        return &dense_[sparse_[entity]];
    }

    // Check if an entity has a component in this storage
    bool Contains(Entity entity) const override {
        return entity < sparse_.size() && sparse_[entity] != kInvalidEntity;
    }

    // IComponentStorage interface
    const void* GetDenseData() const override {
        return dense_.empty() ? nullptr : dense_.data();
    }

    uint64 GetDenseDataSizeBytes() const override {
        return static_cast<uint64>(dense_.size()) * sizeof(T);
    }

    uint32 GetDenseCount() const override {
        return static_cast<uint32>(dense_.size());
    }

    bool IsDirty() const override {
        return dirty_;
    }

    void ClearDirty() override {
        dirty_ = false;
    }

    void RemoveByEntity(Entity entity) override {
        Remove(entity);
    }

    // Access the dense-to-entity mapping (useful for iteration)
    const std::vector<Entity>& GetEntities() const {
        return dense_to_entity_;
    }

private:
    void EnsureSparseSize(Entity entity) {
        if (entity >= sparse_.size()) {
            sparse_.resize(static_cast<mps::size_t>(entity) + 1, kInvalidEntity);
        }
    }

    // Sparse array: entity id -> index into dense_ (kInvalidEntity if absent)
    std::vector<uint32> sparse_;

    // Dense array: contiguous component data
    std::vector<T> dense_;

    // Maps dense index back to entity id
    std::vector<Entity> dense_to_entity_;

    // Dirty flag — set on any mutation
    bool dirty_ = false;
};

}  // namespace database
}  // namespace mps
