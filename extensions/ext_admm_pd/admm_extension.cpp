#include "ext_admm_pd/admm_extension.h"
#include "ext_admm_pd/admm_system_simulator.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::system;

namespace ext_admm_pd {

const std::string ADMMExtension::kName = "ext_admm_pd";

ADMMExtension::ADMMExtension(System& system)
    : system_(system) {}

const std::string& ADMMExtension::GetName() const {
    return kName;
}

void ADMMExtension::Register(System& system) {
    system.AddSimulator(std::make_unique<ADMMSystemSimulator>(system_));
}

}  // namespace ext_admm_pd
