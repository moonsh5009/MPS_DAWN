#include "ext_avbd/avbd_extension.h"
#include "ext_avbd/avbd_system_simulator.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::system;

namespace ext_avbd {

const std::string AVBDExtension::kName = "ext_avbd";

AVBDExtension::AVBDExtension(System& system)
    : system_(system) {}

const std::string& AVBDExtension::GetName() const {
    return kName;
}

void AVBDExtension::Register(System& system) {
    system.AddSimulator(std::make_unique<AVBDSystemSimulator>(system_));
}

}  // namespace ext_avbd
