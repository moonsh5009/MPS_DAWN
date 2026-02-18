#pragma once

#include "core_system/extension.h"
#include <string>

namespace mps { namespace system { class System; } }

namespace ext_cloth {

class ClothExtension : public mps::system::IExtension {
public:
    explicit ClothExtension(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Register(mps::system::System& system) override;

private:
    mps::system::System& system_;
    static const std::string kName;
};

}  // namespace ext_cloth
