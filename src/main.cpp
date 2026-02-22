#include "core_system/system.h"
#include "ext_newton/newton_extension.h"
#include "ext_newton/newton_system_config.h"
#include "ext_pd/pd_extension.h"
#include "ext_pd/pd_system_config.h"
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

    // Extensions: dynamics (data + GPU arrays), mesh (rendering), newton, pd
    system.AddExtension(std::make_unique<ext_dynamics::DynamicsExtension>(system));
    system.AddExtension(std::make_unique<ext_mesh::MeshExtension>(system));
    system.AddExtension(std::make_unique<ext_newton::NewtonExtension>(system));
    system.AddExtension(std::make_unique<ext_pd::PDExtension>(system));

    system.Transact([&](Database& db) {
        // ---- Global physics params (singleton) ----
        db.SetSingleton<GlobalPhysicsParams>(
            {1.0f / 120.0f, {0.0f, -9.81f, 0.0f}, 0.999f});

        // ---- Mesh 1: Newton solver (left) ----
        auto mesh1 = ext_mesh::CreateGrid(db, 64, 64, 0.01f, {-1.0f, 0.0f, 0.0f});
        // auto mesh1 = ext_mesh::ImportOBJ(db, "HR_cloth.obj", 0.01f, {-5.0f, 0.0f, 0.0f});

        ext_dynamics::BuildSpringConstraints(db, mesh1.mesh_entity, 50000.0f);
        // ext_dynamics::BuildAreaConstraints(db, mesh1.mesh_entity, 50000.0f);
        ext_mesh::PinVertices(db, mesh1.mesh_entity, {0});

        // ---- Newton config → mesh1 ----
        ext_newton::NewtonSystemConfig newton_cfg{};
        newton_cfg.newton_iterations = 15;
        newton_cfg.cg_max_iterations = 30;
        newton_cfg.mesh_entity = mesh1.mesh_entity;
        newton_cfg.constraint_count = 1;
        newton_cfg.constraint_entities[0] = mesh1.mesh_entity;

        Entity newton_e = db.CreateEntity();
        db.AddComponent<ext_newton::NewtonSystemConfig>(newton_e, newton_cfg);

        // ---- Mesh 2: PD solver (right, translated +2 in X) ----
        auto mesh2 = ext_mesh::CreateGrid(db, 64, 64, 0.01f, {1.0f, 0.0f, 0.0f});
        // auto mesh2 = ext_mesh::ImportOBJ(db, "HR_cloth.obj", 0.01f, {5.0f, 0.0f, 0.0f});

        ext_dynamics::BuildSpringConstraints(db, mesh2.mesh_entity, 50000.0f);
        // ext_dynamics::BuildAreaConstraints(db, mesh2.mesh_entity, 50000.0f);
        ext_mesh::PinVertices(db, mesh2.mesh_entity, {0});

        // ---- PD config → mesh2 ----
        ext_pd::PDSystemConfig pd_cfg{};
        pd_cfg.iterations = 450;
        pd_cfg.mesh_entity = mesh2.mesh_entity;
        pd_cfg.constraint_count = 1;
        pd_cfg.constraint_entities[0] = mesh2.mesh_entity;

        Entity pd_e = db.CreateEntity();
        db.AddComponent<ext_pd::PDSystemConfig>(pd_e, pd_cfg);
    });

    system.Run();
    return 0;
}
