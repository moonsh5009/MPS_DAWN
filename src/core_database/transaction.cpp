#include "core_database/transaction.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::database;
using namespace mps::util;

bool TransactionManager::Begin() {
    if (active_) {
        LogError("TransactionManager::Begin — transaction already active");
        return false;
    }
    active_ = std::make_unique<Transaction>();
    return true;
}

bool TransactionManager::Commit() {
    if (!active_) {
        LogError("TransactionManager::Commit — no active transaction");
        return false;
    }
    if (!active_->IsEmpty()) {
        undo_stack_.push_back(std::move(active_));
        redo_stack_.clear();
    }
    active_.reset();
    return true;
}

bool TransactionManager::Rollback(Database& db) {
    if (!active_) {
        LogError("TransactionManager::Rollback — no active transaction");
        return false;
    }
    active_->Revert(db);
    active_.reset();
    return true;
}

bool TransactionManager::Undo(Database& db) {
    if (undo_stack_.empty()) {
        return false;
    }
    auto txn = std::move(undo_stack_.back());
    undo_stack_.pop_back();
    txn->Revert(db);
    redo_stack_.push_back(std::move(txn));
    return true;
}

bool TransactionManager::Redo(Database& db) {
    if (redo_stack_.empty()) {
        return false;
    }
    auto txn = std::move(redo_stack_.back());
    redo_stack_.pop_back();
    txn->Apply(db);
    undo_stack_.push_back(std::move(txn));
    return true;
}

bool TransactionManager::IsActive() const {
    return active_ != nullptr;
}

bool TransactionManager::CanUndo() const {
    return !undo_stack_.empty();
}

bool TransactionManager::CanRedo() const {
    return !redo_stack_.empty();
}

void TransactionManager::Record(std::unique_ptr<IOperation> op) {
    if (!active_) {
        return;
    }
    active_->AddOperation(std::move(op));
}
