#include "core_system/system.h"
#include "ext_newton/newton_extension.h"
#include "ext_newton/newton_system_config.h"
#include "ext_mesh/mesh_extension.h"
#include "ext_mesh/mesh_generator.h"
#include "ext_dynamics/dynamics_extension.h"
#include "ext_dynamics/constraint_builder.h"
#include "ext_dynamics/global_physics_params.h"
#include "core_simulate/sim_components.h"
#include "core_util/types.h"
#include <memory>

using namespace mps;
using namespace mps::system;
using namespace mps::database;
using namespace mps::simulate;

int main() {
    System system;
    if (!system.Initialize()) return 1;

    // Extensions: dynamics (data + GPU arrays), mesh (rendering), newton
    system.AddExtension(std::make_unique<ext_dynamics::DynamicsExtension>(system));
    system.AddExtension(std::make_unique<ext_mesh::MeshExtension>(system));
    system.AddExtension(std::make_unique<ext_newton::NewtonExtension>(system));

    system.Transact([&](Database& db) {
        // ---- Global physics params (singleton) ----
        db.SetSingleton<GlobalPhysicsParams>(
            {1.0f / 120.0f, {0.0f, -9.81f, 0.0f}, 0.999f});

        // ---- Mesh 1: Newton solver ----
        auto mesh1 = ext_mesh::CreateGrid(db, 64, 64, 0.025f, {-1.0f, 0.0f, 0.0f});

        ext_dynamics::BuildAreaConstraints(db, mesh1.mesh_entity, 500000.0f, 500000.0f);
        ext_mesh::PinVertices(db, mesh1.mesh_entity, {0});

        // ---- Newton config -> mesh1 ----
        ext_newton::NewtonSystemConfig newton_cfg{};
        newton_cfg.newton_iterations = 5;
        newton_cfg.cg_max_iterations = 100;
        newton_cfg.mesh_entity = mesh1.mesh_entity;
        newton_cfg.constraint_count = 1;
        newton_cfg.constraint_entities[0] = mesh1.mesh_entity;

        Entity newton_e = db.CreateEntity();
        db.AddComponent<ext_newton::NewtonSystemConfig>(newton_e, newton_cfg);
    });

    system.SetSimulationRunning(true);
    system.Run();
    return 0;
}
