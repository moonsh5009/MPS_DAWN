#pragma once

#include "core_system/extension.h"
#include <string>

namespace mps { namespace system { class System; } }

namespace ext_dynamics {

// Unified dynamics term extension — registers all IDynamicsTerm providers
// (spring, area, inertial, gravity, and future term types).
class DynamicsExtension : public mps::system::IExtension {
public:
    explicit DynamicsExtension(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Register(mps::system::System& system) override;

private:
    mps::system::System& system_;
    static const std::string kName;
};

}  // namespace ext_dynamics
