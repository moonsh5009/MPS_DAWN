#include "core_system/system.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::system;
using namespace mps::database;
using namespace mps::util;

System::System()
    : device_db_(db_) {}

System::~System() = default;

void System::Transact(std::function<void(Database&)> fn) {
    db_.Transact([&] { fn(db_); });
    SyncToDevice();
}

void System::Undo() {
    if (db_.Undo()) {
        SyncToDevice();
    }
}

void System::Redo() {
    if (db_.Redo()) {
        SyncToDevice();
    }
}

bool System::CanUndo() const {
    return db_.CanUndo();
}

bool System::CanRedo() const {
    return db_.CanRedo();
}

const Database& System::GetDatabase() const {
    return db_;
}

void System::SyncToDevice() {
    device_db_.Sync();
}
