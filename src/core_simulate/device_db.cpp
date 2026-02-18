#include "core_simulate/device_db.h"

using namespace mps;
using namespace mps::simulate;
using namespace mps::database;

DeviceDB::DeviceDB(Database& host_db)
    : host_db_(host_db) {}

void DeviceDB::Sync() {
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
    host_db_.ClearAllDirty();
}

IDeviceBufferEntry* DeviceDB::GetEntryById(ComponentTypeId id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return nullptr;
    }
    return it->second.get();
}

bool DeviceDB::IsRegistered(ComponentTypeId id) const {
    return entries_.contains(id);
}
