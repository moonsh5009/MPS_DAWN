#include "ext_newton/newton_extension.h"
#include "ext_newton/newton_system_config.h"
#include "ext_newton/gravity_constraint.h"
#include "ext_newton/gravity_term_provider.h"
#include "ext_newton/newton_system_simulator.h"
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
    // Register particle simulation arrays (concatenated from per-mesh ArrayStorage)
    system.RegisterArray<SimPosition>(
        gpu::BufferUsage::Vertex, "sim_position");
    system.RegisterArray<SimVelocity>(
        gpu::BufferUsage::None, "sim_velocity");
    system.RegisterArray<SimMass>(
        gpu::BufferUsage::None, "sim_mass");

    // Register built-in gravity term provider
    system.RegisterTermProvider(
        GetComponentTypeId<GravityConstraintData>(),
        std::make_unique<GravityTermProvider>());

    // Add the Newton system simulator
    system.AddSimulator(std::make_unique<NewtonSystemSimulator>(system_));
}

}  // namespace ext_newton
