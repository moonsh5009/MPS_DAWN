#pragma once

#include <string>

namespace mps {
namespace system {

class System;

class IExtension {
public:
    virtual ~IExtension() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;
    virtual void Register(System& system) = 0;
};

}  // namespace system
}  // namespace mps
