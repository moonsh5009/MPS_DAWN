#include "core_system/system.h"
#include "ext_newton/newton_extension.h"
#include "ext_newton/newton_system_config.h"
#include "ext_newton/gravity_constraint.h"
#include "ext_mesh/mesh_extension.h"
#include "ext_mesh/mesh_generator.h"
#include "ext_dynamics/dynamics_extension.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_dynamics/constraint_builder.h"
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

    // Extensions: Newton system first (runs solver), then mesh (post-processing + rendering)
    auto newton_ext = std::make_unique<ext_newton::NewtonExtension>(system);
    system.AddExtension(std::move(newton_ext));

    auto mesh_ext = std::make_unique<ext_mesh::MeshExtension>(system);
    system.AddExtension(std::move(mesh_ext));

    auto dynamics_ext = std::make_unique<ext_dynamics::DynamicsExtension>(system);
    system.AddExtension(std::move(dynamics_ext));

    // Create scene entities in Database
    system.Transact([&](Database& db) {
        // Create mesh from OBJ file
        auto mesh = ext_mesh::ImportOBJ(db, "LR_cloth.obj", 0.01f);
        ext_dynamics::BuildConstraintsFromFaces(db, mesh.mesh_entity);

        // Pin center of top row
        ext_mesh::PinVertices(db, mesh.mesh_entity, {0});

        // Config-only constraint entities
        Entity gravity_e = db.CreateEntity();
        db.AddComponent<ext_newton::GravityConstraintData>(gravity_e, {0.0f, -9.81f, 0.0f});

        Entity spring_e = db.CreateEntity();
        db.AddComponent<ext_dynamics::SpringConstraintData>(spring_e, {500000.0f});

        Entity area_e = db.CreateEntity();
        db.AddComponent<ext_dynamics::AreaConstraintData>(area_e, {500000.0f});

        // Newton System entity (references constraints)
        ext_newton::NewtonSystemConfig config{};
        config.newton_iterations = 15;
        config.cg_max_iterations = 30;
        config.damping = 0.999f;
        config.cg_tolerance = 1e-6f;
        config.constraint_count = 3;
        config.constraint_entities[0] = gravity_e;
        config.constraint_entities[1] = spring_e;
        config.constraint_entities[2] = area_e;

        Entity newton_e = db.CreateEntity();
        db.AddComponent<ext_newton::NewtonSystemConfig>(newton_e, config);
    });

    // Enter main loop (cleanup in ~System)
    system.Run();
    return 0;
}
