#include "ext_dynamics/dynamics_extension.h"
#include "ext_dynamics/spring_types.h"
#include "ext_dynamics/area_types.h"
#include "core_simulate/sim_components.h"
#include "ext_dynamics/global_physics_params.h"
#include "core_system/system.h"
#include "core_gpu/gpu_types.h"
#include "core_util/logger.h"

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
    // Register physics params singleton (shared by all solvers)
    system.GetDeviceDB().RegisterSingleton<GlobalPhysicsParams, PhysicsParamsGPU>(
        ToGPU, "physics_params");

    // Register particle simulation arrays (GPU mirroring for positions/velocity/mass)
    // Placed here so both Newton and PD extensions can coexist without duplicate registration.
    system.RegisterArray<SimPosition>(
        gpu::BufferUsage::Vertex, "sim_position");
    system.RegisterArray<SimVelocity>(
        gpu::BufferUsage::None, "sim_velocity");
    system.RegisterArray<SimMass>(
        gpu::BufferUsage::None, "sim_mass");

    // Register indexed arrays (topology with auto-offset relative to SimPosition)
    system.RegisterIndexedArray<SpringEdge, SimPosition>(
        gpu::BufferUsage::None, "spring_edges",
        [](SpringEdge& e, uint32 off) { e.n0 += off; e.n1 += off; });

    system.RegisterIndexedArray<AreaTriangle, SimPosition>(
        gpu::BufferUsage::None, "area_triangles",
        [](AreaTriangle& t, uint32 off) { t.n0 += off; t.n1 += off; t.n2 += off; });
}

}  // namespace ext_dynamics
