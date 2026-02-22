#include "ext_pd_term/pd_term_extension.h"
#include "ext_pd_term/pd_spring_term_provider.h"
#include "ext_pd_term/pd_area_term_provider.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_constraint.h"
#include "core_system/system.h"
#include "core_database/component_type.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::system;
using namespace mps::database;

namespace ext_pd_term {

const std::string PDTermExtension::kName = "ext_pd_term";

PDTermExtension::PDTermExtension(System& system)
    : system_(system) {}

const std::string& PDTermExtension::GetName() const {
    return kName;
}

void PDTermExtension::Register(System& system) {
    // Register PD term providers (reuse ext_dynamics constraint data types)
    system.RegisterPDTermProvider(
        GetComponentTypeId<ext_dynamics::SpringConstraintData>(),
        std::make_unique<PDSpringTermProvider>(system_));

    system.RegisterPDTermProvider(
        GetComponentTypeId<ext_dynamics::AreaConstraintData>(),
        std::make_unique<PDAreaTermProvider>(system_));
}

}  // namespace ext_pd_term
