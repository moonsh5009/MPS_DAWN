#pragma once

#include "core_database/array_storage.h"
#include "core_database/array_transaction.h"
#include "core_database/component_storage.h"
#include "core_database/component_type.h"
#include "core_database/entity.h"
#include "core_database/transaction.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mps {
namespace database {

// Central ECS database facade.
// Manages entities, component storage, and undo/redo transactions.
class Database {
public:
    Database() = default;

    // --- Entity management ---
    Entity CreateEntity();
    void DestroyEntity(Entity entity);

    // --- Component operations (template, public) ---
    template<Component T>
    void AddComponent(Entity entity, const T& component);

    template<Component T>
    void RemoveComponent(Entity entity);

    template<Component T>
    void SetComponent(Entity entity, const T& component);

    template<Component T>
    T* GetComponent(Entity entity);

    template<Component T>
    const T* GetComponent(Entity entity) const;

    template<Component T>
    bool HasComponent(Entity entity) const;

    // --- Transaction / undo-redo ---
    void Transact(std::function<void()> fn);
    bool Undo();
    bool Redo();
    bool CanUndo() const;
    bool CanRedo() const;

    // --- Storage access (for renderers / systems) ---
    IComponentStorage* GetStorageById(ComponentTypeId id);
    const IComponentStorage* GetStorageById(ComponentTypeId id) const;
    std::vector<ComponentTypeId> GetDirtyTypeIds() const;
    void ClearAllDirty();

    // --- Array storage access (for DeviceArrayBuffer) ---
    IArrayStorage* GetArrayStorageById(ComponentTypeId id);
    const IArrayStorage* GetArrayStorageById(ComponentTypeId id) const;
    std::vector<ComponentTypeId> GetDirtyArrayTypeIds() const;

    // --- Array operations (per-entity variable-length arrays) ---
    template<Component T>
    void SetArray(Entity entity, std::vector<T> data);

    template<Component T>
    const std::vector<T>* GetArray(Entity entity) const;

    template<Component T>
    void RemoveArray(Entity entity);

    template<Component T>
    bool HasArray(Entity entity) const;

    // --- Direct array operations (no transaction recording) ---
    template<Component T>
    void DirectSetArray(Entity entity, std::vector<T> data);

    template<Component T>
    void DirectRemoveArray(Entity entity);

    // --- Direct storage operations (no transaction recording) ---
    // Used internally by operation Apply/Revert so undo/redo replays
    // don't double-record into the transaction manager.
    template<Component T>
    void DirectAddComponent(Entity entity, const T& component);

    template<Component T>
    void DirectRemoveComponent(Entity entity);

    template<Component T>
    void DirectSetComponent(Entity entity, const T& component);

private:
    // Begin/Commit/Rollback are private — use Transact() publicly
    void BeginTransaction();
    void Commit();
    void Rollback();

    // Get or create typed storage for component type T
    template<Component T>
    ComponentStorage<T>& GetOrCreateStorage();

    template<Component T>
    ComponentStorage<T>* GetStorage();

    template<Component T>
    const ComponentStorage<T>* GetStorage() const;

    // Get or create typed array storage for type T
    template<Component T>
    ArrayStorage<T>& GetOrCreateArrayStorage();

    template<Component T>
    ArrayStorage<T>* GetArrayStorage();

    template<Component T>
    const ArrayStorage<T>* GetArrayStorage() const;

    EntityManager entity_manager_;
    TransactionManager transaction_manager_;
    std::unordered_map<ComponentTypeId, std::unique_ptr<IComponentStorage>> storages_;
    std::unordered_map<ComponentTypeId, std::unique_ptr<IArrayStorage>> array_storages_;
};

// ============================================================================
// Template implementations
// ============================================================================

template<Component T>
ComponentStorage<T>& Database::GetOrCreateStorage() {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = storages_.find(id);
    if (it == storages_.end()) {
        auto storage = std::make_unique<ComponentStorage<T>>();
        auto* ptr = storage.get();
        storages_.emplace(id, std::move(storage));
        return *ptr;
    }
    return *static_cast<ComponentStorage<T>*>(it->second.get());
}

template<Component T>
ComponentStorage<T>* Database::GetStorage() {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = storages_.find(id);
    if (it == storages_.end()) {
        return nullptr;
    }
    return static_cast<ComponentStorage<T>*>(it->second.get());
}

template<Component T>
const ComponentStorage<T>* Database::GetStorage() const {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = storages_.find(id);
    if (it == storages_.end()) {
        return nullptr;
    }
    return static_cast<const ComponentStorage<T>*>(it->second.get());
}

// --- Public component operations (record into transaction) ---

template<Component T>
void Database::AddComponent(Entity entity, const T& component) {
    auto& storage = GetOrCreateStorage<T>();
    storage.Add(entity, component);
    transaction_manager_.Record(
        std::make_unique<AddComponentOp<T>>(entity, component));
}

template<Component T>
void Database::RemoveComponent(Entity entity) {
    auto* storage = GetStorage<T>();
    if (!storage) return;
    const T* existing = storage->Get(entity);
    if (!existing) return;
    T copy = *existing;
    storage->Remove(entity);
    transaction_manager_.Record(
        std::make_unique<RemoveComponentOp<T>>(entity, copy));
}

template<Component T>
void Database::SetComponent(Entity entity, const T& component) {
    auto* storage = GetStorage<T>();
    if (!storage) return;
    const T* existing = storage->Get(entity);
    if (!existing) return;
    T old_value = *existing;
    storage->Set(entity, component);
    transaction_manager_.Record(
        std::make_unique<SetComponentOp<T>>(entity, old_value, component));
}

template<Component T>
T* Database::GetComponent(Entity entity) {
    auto* storage = GetStorage<T>();
    if (!storage) return nullptr;
    return storage->Get(entity);
}

template<Component T>
const T* Database::GetComponent(Entity entity) const {
    auto* storage = GetStorage<T>();
    if (!storage) return nullptr;
    return storage->Get(entity);
}

template<Component T>
bool Database::HasComponent(Entity entity) const {
    auto* storage = GetStorage<T>();
    if (!storage) return false;
    return storage->Contains(entity);
}

// --- Array storage helpers ---

template<Component T>
ArrayStorage<T>& Database::GetOrCreateArrayStorage() {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = array_storages_.find(id);
    if (it == array_storages_.end()) {
        auto storage = std::make_unique<ArrayStorage<T>>();
        auto* ptr = storage.get();
        array_storages_.emplace(id, std::move(storage));
        return *ptr;
    }
    return *static_cast<ArrayStorage<T>*>(it->second.get());
}

template<Component T>
ArrayStorage<T>* Database::GetArrayStorage() {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = array_storages_.find(id);
    if (it == array_storages_.end()) return nullptr;
    return static_cast<ArrayStorage<T>*>(it->second.get());
}

template<Component T>
const ArrayStorage<T>* Database::GetArrayStorage() const {
    ComponentTypeId id = GetComponentTypeId<T>();
    auto it = array_storages_.find(id);
    if (it == array_storages_.end()) return nullptr;
    return static_cast<const ArrayStorage<T>*>(it->second.get());
}

// --- Public array operations (record into transaction) ---

template<Component T>
void Database::SetArray(Entity entity, std::vector<T> data) {
    auto& storage = GetOrCreateArrayStorage<T>();
    std::vector<T> old_data;
    const auto* existing = storage.GetArray(entity);
    if (existing) {
        old_data = *existing;
    }
    storage.SetArray(entity, data);
    transaction_manager_.Record(
        std::make_unique<SetArrayOp<T>>(entity, std::move(old_data), std::move(data)));
}

template<Component T>
const std::vector<T>* Database::GetArray(Entity entity) const {
    auto* storage = GetArrayStorage<T>();
    if (!storage) return nullptr;
    return storage->GetArray(entity);
}

template<Component T>
void Database::RemoveArray(Entity entity) {
    auto* storage = GetArrayStorage<T>();
    if (!storage) return;
    const auto* existing = storage->GetArray(entity);
    if (!existing) return;
    std::vector<T> old_data = *existing;
    storage->Remove(entity);
    transaction_manager_.Record(
        std::make_unique<RemoveArrayOp<T>>(entity, std::move(old_data)));
}

template<Component T>
bool Database::HasArray(Entity entity) const {
    auto* storage = GetArrayStorage<T>();
    if (!storage) return false;
    return storage->Has(entity);
}

// --- Direct array operations (no transaction recording) ---

template<Component T>
void Database::DirectSetArray(Entity entity, std::vector<T> data) {
    auto& storage = GetOrCreateArrayStorage<T>();
    storage.SetArray(entity, std::move(data));
}

template<Component T>
void Database::DirectRemoveArray(Entity entity) {
    auto* storage = GetArrayStorage<T>();
    if (!storage) return;
    storage->Remove(entity);
}

// --- Direct operations (no transaction recording, used by undo/redo) ---

template<Component T>
void Database::DirectAddComponent(Entity entity, const T& component) {
    auto& storage = GetOrCreateStorage<T>();
    storage.Add(entity, component);
}

template<Component T>
void Database::DirectRemoveComponent(Entity entity) {
    auto* storage = GetStorage<T>();
    if (!storage) return;
    storage->Remove(entity);
}

template<Component T>
void Database::DirectSetComponent(Entity entity, const T& component) {
    auto* storage = GetStorage<T>();
    if (!storage) return;
    storage->Set(entity, component);
}

// ============================================================================
// Operation template implementations (need full Database definition)
// ============================================================================

template<Component T>
void AddComponentOp<T>::Apply(Database& db) {
    db.DirectAddComponent<T>(entity_, component_);
}

template<Component T>
void AddComponentOp<T>::Revert(Database& db) {
    db.DirectRemoveComponent<T>(entity_);
}

template<Component T>
void RemoveComponentOp<T>::Apply(Database& db) {
    db.DirectRemoveComponent<T>(entity_);
}

template<Component T>
void RemoveComponentOp<T>::Revert(Database& db) {
    db.DirectAddComponent<T>(entity_, component_);
}

template<Component T>
void SetComponentOp<T>::Apply(Database& db) {
    db.DirectSetComponent<T>(entity_, new_value_);
}

template<Component T>
void SetComponentOp<T>::Revert(Database& db) {
    db.DirectSetComponent<T>(entity_, old_value_);
}

// ============================================================================
// Array operation template implementations (need full Database definition)
// ============================================================================

template<Component T>
void SetArrayOp<T>::Apply(Database& db) {
    db.DirectSetArray<T>(entity_, new_data_);
}

template<Component T>
void SetArrayOp<T>::Revert(Database& db) {
    if (old_data_.empty()) {
        db.DirectRemoveArray<T>(entity_);
    } else {
        db.DirectSetArray<T>(entity_, old_data_);
    }
}

template<Component T>
void RemoveArrayOp<T>::Apply(Database& db) {
    db.DirectRemoveArray<T>(entity_);
}

template<Component T>
void RemoveArrayOp<T>::Revert(Database& db) {
    db.DirectSetArray<T>(entity_, old_data_);
}

}  // namespace database
}  // namespace mps
