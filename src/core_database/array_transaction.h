#pragma once

#include "core_database/component_type.h"
#include "core_database/entity.h"
#include "core_database/transaction.h"
#include <vector>

namespace mps {
namespace database {

class Database;

// Operation: set an array on an entity
template<Component T>
class SetArrayOp : public IOperation {
public:
    SetArrayOp(Entity entity, std::vector<T> old_data, std::vector<T> new_data)
        : entity_(entity), old_data_(std::move(old_data)), new_data_(std::move(new_data)) {}

    void Apply(Database& db) override;
    void Revert(Database& db) override;

private:
    Entity entity_;
    std::vector<T> old_data_;
    std::vector<T> new_data_;
};

// Operation: remove an array from an entity
template<Component T>
class RemoveArrayOp : public IOperation {
public:
    RemoveArrayOp(Entity entity, std::vector<T> old_data)
        : entity_(entity), old_data_(std::move(old_data)) {}

    void Apply(Database& db) override;
    void Revert(Database& db) override;

private:
    Entity entity_;
    std::vector<T> old_data_;
};

}  // namespace database
}  // namespace mps
