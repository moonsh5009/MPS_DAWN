#pragma once

#include "core_database/component_type.h"
#include "core_database/entity.h"
#include <memory>
#include <vector>

namespace mps {
namespace database {

// Forward declaration — Database is needed by operations but full definition is in database.h
class Database;

// Interface for a single undoable/redoable operation
class IOperation {
public:
    virtual ~IOperation() = default;

    // Apply the operation (forward)
    virtual void Apply(Database& db) = 0;

    // Revert the operation (undo)
    virtual void Revert(Database& db) = 0;
};

// Operation: add a component to an entity
template<Component T>
class AddComponentOp : public IOperation {
public:
    AddComponentOp(Entity entity, const T& component)
        : entity_(entity), component_(component) {}

    void Apply(Database& db) override;
    void Revert(Database& db) override;

private:
    Entity entity_;
    T component_;
};

// Operation: remove a component from an entity
template<Component T>
class RemoveComponentOp : public IOperation {
public:
    RemoveComponentOp(Entity entity, const T& component)
        : entity_(entity), component_(component) {}

    void Apply(Database& db) override;
    void Revert(Database& db) override;

private:
    Entity entity_;
    T component_;
};

// Operation: set (overwrite) a component value
template<Component T>
class SetComponentOp : public IOperation {
public:
    SetComponentOp(Entity entity, const T& old_value, const T& new_value)
        : entity_(entity), old_value_(old_value), new_value_(new_value) {}

    void Apply(Database& db) override;
    void Revert(Database& db) override;

private:
    Entity entity_;
    T old_value_;
    T new_value_;
};

// A group of operations that form an atomic unit of work
class Transaction {
public:
    Transaction() = default;

    // Add an operation to this transaction
    void AddOperation(std::unique_ptr<IOperation> op) {
        operations_.push_back(std::move(op));
    }

    // Apply all operations in order
    void Apply(Database& db) {
        for (auto& op : operations_) {
            op->Apply(db);
        }
    }

    // Revert all operations in reverse order
    void Revert(Database& db) {
        for (auto it = operations_.rbegin(); it != operations_.rend(); ++it) {
            (*it)->Revert(db);
        }
    }

    // Check if this transaction has any operations
    bool IsEmpty() const {
        return operations_.empty();
    }

private:
    std::vector<std::unique_ptr<IOperation>> operations_;
};

// Manages the active transaction and undo/redo stacks
class TransactionManager {
public:
    TransactionManager() = default;

    // Begin a new transaction. Returns false if one is already active.
    bool Begin();

    // Commit the active transaction and push it to the undo stack.
    // Clears the redo stack. Returns false if no active transaction.
    bool Commit();

    // Rollback the active transaction by reverting all its operations.
    // Returns false if no active transaction.
    bool Rollback(Database& db);

    // Undo the last committed transaction
    bool Undo(Database& db);

    // Redo the last undone transaction
    bool Redo(Database& db);

    // Query state
    bool IsActive() const;
    bool CanUndo() const;
    bool CanRedo() const;

    // Record an operation into the active transaction.
    // No-op if no transaction is active (e.g., during undo/redo replay).
    void Record(std::unique_ptr<IOperation> op);

private:
    std::unique_ptr<Transaction> active_;
    std::vector<std::unique_ptr<Transaction>> undo_stack_;
    std::vector<std::unique_ptr<Transaction>> redo_stack_;
};

}  // namespace database
}  // namespace mps
