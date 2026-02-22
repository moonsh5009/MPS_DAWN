#pragma once

#include "core_util/types.h"
#include <string>

namespace mps {
namespace simulate {

class ISimulator {
public:
    virtual ~ISimulator() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;
    virtual void Initialize() {}
    virtual void Update() = 0;
    virtual void Shutdown() {}
    virtual void OnDatabaseChanged() {}
};

}  // namespace simulate
}  // namespace mps
