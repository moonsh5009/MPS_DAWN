#include "ext_newton/newton_extension.h"
#include "ext_newton/newton_system_config.h"
#include "ext_newton/spring_term_provider.h"
#include "ext_newton/area_term_provider.h"
#include "ext_newton/newton_system_simulator.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_constraint.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_database/component_type.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::system;
using namespace mps::database;
using namespace mps::simulate;

namespace ext_newton {

const std::string NewtonExtension::kName = "ext_newton";

NewtonExtension::NewtonExtension(System& system)
    : system_(system) {}

const std::string& NewtonExtension::GetName() const {
    return kName;
}

void NewtonExtension::Register(System& system) {
    // Spring constraint: edge-based forces + Hessian
    system.RegisterTermProvider(
        GetComponentTypeId<ext_dynamics::SpringConstraintData>(),
        std::make_unique<SpringTermProvider>(system_));

    // Area constraint: triangle area preservation
    system.RegisterTermProvider(
        GetComponentTypeId<ext_dynamics::AreaConstraintData>(),
        std::make_unique<AreaTermProvider>(system_));

    // Add the Newton system simulator
    system.AddSimulator(std::make_unique<NewtonSystemSimulator>(system_));
}

}  // namespace ext_newton
