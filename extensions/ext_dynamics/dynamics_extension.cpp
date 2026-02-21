#include "ext_dynamics/dynamics_extension.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/spring_types.h"
#include "ext_dynamics/spring_term_provider.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_dynamics/area_types.h"
#include "ext_dynamics/area_term_provider.h"
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
using namespace mps::simulate;

namespace ext_dynamics {

const std::string DynamicsExtension::kName = "ext_dynamics";

DynamicsExtension::DynamicsExtension(System& system)
    : system_(system) {}

const std::string& DynamicsExtension::GetName() const {
    return kName;
}

void DynamicsExtension::Register(System& system) {
    // Register indexed arrays (topology with auto-offset relative to SimPosition)
    system.RegisterIndexedArray<SpringEdge, SimPosition>(
        gpu::BufferUsage::None, "spring_edges",
        [](SpringEdge& e, uint32 off) { e.n0 += off; e.n1 += off; });

    system.RegisterIndexedArray<AreaTriangle, SimPosition>(
        gpu::BufferUsage::None, "area_triangles",
        [](AreaTriangle& t, uint32 off) { t.n0 += off; t.n1 += off; t.n2 += off; });

    // Spring constraint: edge-based forces + Hessian
    system.RegisterTermProvider(
        GetComponentTypeId<SpringConstraintData>(),
        std::make_unique<SpringTermProvider>(system_));

    // Area constraint: triangle area preservation
    system.RegisterTermProvider(
        GetComponentTypeId<AreaConstraintData>(),
        std::make_unique<AreaTermProvider>(system_));
}

}  // namespace ext_dynamics
