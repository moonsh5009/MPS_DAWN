#pragma once

#include "core_system/extension.h"
#include <string>

namespace mps { namespace system { class System; } }

namespace ext_mesh {

class MeshPostProcessor;

class MeshExtension : public mps::system::IExtension {
public:
    explicit MeshExtension(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Register(mps::system::System& system) override;

private:
    mps::system::System& system_;
    MeshPostProcessor* post_processor_ = nullptr;
    static const std::string kName;
};

}  // namespace ext_mesh
