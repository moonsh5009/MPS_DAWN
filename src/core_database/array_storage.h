#pragma once

#include "core_database/component_type.h"
#include "core_database/entity.h"
#include <unordered_map>
#include <vector>

namespace mps {
namespace database {

// Type-erased interface for per-entity array storage
class IArrayStorage {
public:
    virtual ~IArrayStorage() = default;

    virtual bool Has(Entity entity) const = 0;
    virtual void Remove(Entity entity) = 0;
    virtual bool IsDirty() const = 0;
    virtual void ClearDirty() = 0;

    // Iteration support (for DeviceArrayBuffer concatenation)
    virtual std::vector<Entity> GetEntities() const = 0;
    virtual const void* GetArrayData(Entity entity) const = 0;
    virtual uint32 GetArrayCount(Entity entity) const = 0;
    virtual uint32 GetElementSize() const = 0;
};

// Stores variable-length arrays per entity (e.g., faces, edges).
// T must satisfy the Component concept (trivially_copyable + standard_layout).
template<Component T>
class ArrayStorage : public IArrayStorage {
public:
    ArrayStorage() = default;

    void SetArray(Entity entity, std::vector<T> data) {
        arrays_[entity] = std::move(data);
        dirty_ = true;
    }

    const std::vector<T>* GetArray(Entity entity) const {
        auto it = arrays_.find(entity);
        if (it == arrays_.end()) return nullptr;
        return &it->second;
    }

    bool Has(Entity entity) const override {
        return arrays_.contains(entity);
    }

    uint32 GetCount(Entity entity) const {
        auto it = arrays_.find(entity);
        if (it == arrays_.end()) return 0;
        return static_cast<uint32>(it->second.size());
    }

    void Remove(Entity entity) override {
        if (arrays_.erase(entity) > 0) {
            dirty_ = true;
        }
    }

    bool IsDirty() const override { return dirty_; }
    void ClearDirty() override { dirty_ = false; }

    std::vector<Entity> GetEntities() const override {
        std::vector<Entity> result;
        result.reserve(arrays_.size());
        for (const auto& [e, _] : arrays_) {
            result.push_back(e);
        }
        return result;
    }

    const void* GetArrayData(Entity entity) const override {
        auto it = arrays_.find(entity);
        if (it == arrays_.end() || it->second.empty()) return nullptr;
        return it->second.data();
    }

    uint32 GetArrayCount(Entity entity) const override {
        return GetCount(entity);
    }

    uint32 GetElementSize() const override {
        return static_cast<uint32>(sizeof(T));
    }

private:
    std::unordered_map<Entity, std::vector<T>> arrays_;
    bool dirty_ = false;
};

}  // namespace database
}  // namespace mps
