#include "core_database/database.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::database;
using namespace mps::util;

// --- Entity management ---

Entity Database::CreateEntity() {
    return entity_manager_.Create();
}

void Database::DestroyEntity(Entity entity) {
    // Remove the entity's components from all storages
    for (auto& [id, storage] : storages_) {
        if (storage->Contains(entity)) {
            storage->RemoveByEntity(entity);
        }
    }
    // Remove the entity's arrays from all array storages
    for (auto& [id, storage] : array_storages_) {
        if (storage->Has(entity)) {
            storage->Remove(entity);
        }
    }
    entity_manager_.Destroy(entity);
}

// --- Transaction / undo-redo ---

void Database::BeginTransaction() {
    transaction_manager_.Begin();
}

void Database::Commit() {
    transaction_manager_.Commit();
}

void Database::Rollback() {
    transaction_manager_.Rollback(*this);
}

void Database::Transact(std::function<void()> fn) {
    BeginTransaction();
    try {
        fn();
        Commit();
    } catch (...) {
        Rollback();
        throw;
    }
}

bool Database::Undo() {
    return transaction_manager_.Undo(*this);
}

bool Database::Redo() {
    return transaction_manager_.Redo(*this);
}

bool Database::CanUndo() const {
    return transaction_manager_.CanUndo();
}

bool Database::CanRedo() const {
    return transaction_manager_.CanRedo();
}

// --- Storage access ---

IComponentStorage* Database::GetStorageById(ComponentTypeId id) {
    auto it = storages_.find(id);
    if (it == storages_.end()) {
        return nullptr;
    }
    return it->second.get();
}

const IComponentStorage* Database::GetStorageById(ComponentTypeId id) const {
    auto it = storages_.find(id);
    if (it == storages_.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<ComponentTypeId> Database::GetDirtyTypeIds() const {
    std::vector<ComponentTypeId> result;
    for (const auto& [id, storage] : storages_) {
        if (storage->IsDirty()) {
            result.push_back(id);
        }
    }
    return result;
}

void Database::ClearAllDirty() {
    for (auto& [id, storage] : storages_) {
        storage->ClearDirty();
    }
    for (auto& [id, storage] : array_storages_) {
        storage->ClearDirty();
    }
}

// --- Array storage access ---

IArrayStorage* Database::GetArrayStorageById(ComponentTypeId id) {
    auto it = array_storages_.find(id);
    if (it == array_storages_.end()) return nullptr;
    return it->second.get();
}

const IArrayStorage* Database::GetArrayStorageById(ComponentTypeId id) const {
    auto it = array_storages_.find(id);
    if (it == array_storages_.end()) return nullptr;
    return it->second.get();
}

std::vector<ComponentTypeId> Database::GetDirtyArrayTypeIds() const {
    std::vector<ComponentTypeId> result;
    for (const auto& [id, storage] : array_storages_) {
        if (storage->IsDirty()) {
            result.push_back(id);
        }
    }
    return result;
}
