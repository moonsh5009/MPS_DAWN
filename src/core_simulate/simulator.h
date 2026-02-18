#pragma once

#include "core_util/types.h"
#include <string>

namespace mps {
namespace database { class Database; }
namespace simulate {

class ISimulator {
public:
    virtual ~ISimulator() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;
    virtual void Initialize(database::Database& db) {}
    virtual void Update(database::Database& db, float32 dt) = 0;
    virtual void Shutdown() {}
};

}  // namespace simulate
}  // namespace mps
