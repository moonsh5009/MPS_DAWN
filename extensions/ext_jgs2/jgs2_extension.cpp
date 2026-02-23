#include "ext_jgs2/jgs2_extension.h"
#include "ext_jgs2/jgs2_system_simulator.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::system;

namespace ext_jgs2 {

const std::string JGS2Extension::kName = "ext_jgs2";

JGS2Extension::JGS2Extension(System& system)
    : system_(system) {}

const std::string& JGS2Extension::GetName() const {
    return kName;
}

void JGS2Extension::Register(System& system) {
    system.AddSimulator(std::make_unique<JGS2SystemSimulator>(system_));
}

}  // namespace ext_jgs2
