#include "core_simulate/device_db.h"

using namespace mps;
using namespace mps::simulate;
using namespace mps::database;

DeviceDB::DeviceDB(Database& host_db)
    : host_db_(host_db) {}

void DeviceDB::Sync() {
    // 1. Component sync
    auto dirty_ids = host_db_.GetDirtyTypeIds();
    for (auto id : dirty_ids) {
        auto it = entries_.find(id);
        if (it != entries_.end()) {
            auto* storage = host_db_.GetStorageById(id);
            if (storage) {
                it->second->SyncFromHost(*storage);
            }
        }
    }

    // 2. Reference array sync FIRST (these are position-like arrays that indexed arrays depend on)
    auto dirty_array_ids = host_db_.GetDirtyArrayTypeIds();
    std::unordered_set<ComponentTypeId> changed_refs;
    for (auto id : dirty_array_ids) {
        auto it = array_entries_.find(id);
        if (it != array_entries_.end()) {
            uint32 old_count = it->second->GetTotalCount();
            it->second->SyncFromHost(host_db_);
            if (it->second->GetTotalCount() != old_count) {
                changed_refs.insert(id);
            }
        }
    }

    // 3. Mark dependent indexed arrays if reference layout changed
    for (auto& [id, entry] : indexed_entries_) {
        auto ref_it = indexed_ref_map_.find(id);
        if (ref_it != indexed_ref_map_.end() && changed_refs.contains(ref_it->second)) {
            entry->MarkRefLayoutChanged();
        }
    }

    // 4. Indexed array sync (after references are up to date)
    for (auto& [id, entry] : indexed_entries_) {
        entry->SyncFromHost(host_db_);
    }

    host_db_.ClearAllDirty();
}

void DeviceDB::ForceSync() {
    // Components
    for (auto& [id, entry] : entries_) {
        auto* storage = host_db_.GetStorageById(id);
        if (storage) {
            entry->SyncFromHost(*storage);
        }
    }

    // Reference arrays FIRST
    for (auto& [id, entry] : array_entries_) {
        entry->ForceSyncFromHost(host_db_);
    }

    // Mark all indexed arrays for rebuild (reference layout may have changed)
    for (auto& [id, entry] : indexed_entries_) {
        entry->MarkRefLayoutChanged();
    }

    // Indexed arrays
    for (auto& [id, entry] : indexed_entries_) {
        entry->ForceSyncFromHost(host_db_);
    }

    host_db_.ClearAllDirty();
}

IDeviceBufferEntry* DeviceDB::GetEntryById(ComponentTypeId id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return nullptr;
    }
    return it->second.get();
}

IDeviceArrayEntry* DeviceDB::GetArrayEntryById(ComponentTypeId id) const {
    auto it = array_entries_.find(id);
    if (it != array_entries_.end()) return it->second.get();
    auto iit = indexed_entries_.find(id);
    if (iit != indexed_entries_.end()) return iit->second.get();
    return nullptr;
}

bool DeviceDB::IsRegistered(ComponentTypeId id) const {
    return entries_.contains(id);
}
