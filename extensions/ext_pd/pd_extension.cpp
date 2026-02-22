#include "ext_pd/pd_extension.h"
#include "ext_pd/pd_system_config.h"
#include "ext_pd/pd_spring_term_provider.h"
#include "ext_pd/pd_area_term_provider.h"
#include "ext_pd/pd_system_simulator.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_constraint.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_database/component_type.h"
#include "core_gpu/gpu_types.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::system;
using namespace mps::database;

namespace ext_pd {

const std::string PDExtension::kName = "ext_pd";

PDExtension::PDExtension(System& system)
    : system_(system) {}

const std::string& PDExtension::GetName() const {
    return kName;
}

void PDExtension::Register(System& system) {
    // Register PD term providers (reuse ext_dynamics constraint data types)
    system.RegisterPDTermProvider(
        GetComponentTypeId<ext_dynamics::SpringConstraintData>(),
        std::make_unique<PDSpringTermProvider>(system_));

    system.RegisterPDTermProvider(
        GetComponentTypeId<ext_dynamics::AreaConstraintData>(),
        std::make_unique<PDAreaTermProvider>(system_));

    // Add the PD system simulator
    system.AddSimulator(std::make_unique<PDSystemSimulator>(system_));
}

}  // namespace ext_pd
