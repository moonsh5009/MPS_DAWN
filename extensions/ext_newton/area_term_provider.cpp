#include "ext_newton/area_term_provider.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_newton/area_term.h"
#include "core_simulate/sim_components.h"
#include "core_database/database.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <algorithm>

using namespace mps;
using namespace mps::util;
using namespace mps::database;
using namespace mps::simulate;

using ext_dynamics::AreaConstraintData;
using ext_dynamics::AreaTriangle;

namespace ext_newton {

AreaTermProvider::AreaTermProvider(system::System& system)
    : system_(system) {}

std::string_view AreaTermProvider::GetTermName() const {
    return "AreaTermProvider";
}

bool AreaTermProvider::HasConfig(const Database& db, Entity entity) const {
    return db.HasComponent<AreaConstraintData>(entity);
}

std::unique_ptr<IDynamicsTerm> AreaTermProvider::CreateTerm(
    const Database& db, Entity entity, uint32 /* node_count */) {

    // Config (stiffness) still read from constraint entity
    const auto* config = db.GetComponent<AreaConstraintData>(entity);
    if (!config) {
        LogError("AreaTermProvider: no AreaConstraintData on entity ", entity);
        return nullptr;
    }

    // Gather triangles: if constraint entity itself has AreaTriangle data, scope to it only
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<AreaTriangle>());
    if (!storage) return nullptr;

    std::vector<AreaTriangle> all_triangles;
    bool scoped = storage->GetArrayCount(entity) > 0;

    if (scoped) {
        // Scoped: use only this entity's triangles with local 0-based indices (no offset)
        uint32 count = storage->GetArrayCount(entity);
        const auto* data = static_cast<const AreaTriangle*>(storage->GetArrayData(entity));
        all_triangles.assign(data, data + count);
    } else {
        // Global: merge ALL entities' triangles with position offsets
        auto entities = storage->GetEntities();
        std::sort(entities.begin(), entities.end());

        auto* pos_storage = db.GetArrayStorageById(GetComponentTypeId<SimPosition>());
        std::unordered_map<Entity, uint32> pos_offset_map;
        if (pos_storage) {
            auto pos_entities = pos_storage->GetEntities();
            std::sort(pos_entities.begin(), pos_entities.end());
            uint32 offset = 0;
            for (Entity e : pos_entities) {
                pos_offset_map[e] = offset;
                offset += pos_storage->GetArrayCount(e);
            }
        }

        for (Entity mesh_e : entities) {
            uint32 count = storage->GetArrayCount(mesh_e);
            if (count == 0) continue;
            const auto* data = static_cast<const AreaTriangle*>(storage->GetArrayData(mesh_e));

            uint32 node_offset = 0;
            auto it = pos_offset_map.find(mesh_e);
            if (it != pos_offset_map.end()) {
                node_offset = it->second;
            }

            for (uint32 i = 0; i < count; ++i) {
                AreaTriangle g = data[i];
                g.n0 += node_offset;
                g.n1 += node_offset;
                g.n2 += node_offset;
                all_triangles.push_back(g);
            }
        }
    }

    if (all_triangles.empty()) return nullptr;

    face_count_ = static_cast<uint32>(all_triangles.size());
    return std::make_unique<AreaTerm>(all_triangles, config->stiffness);
}

void AreaTermProvider::DeclareTopology(uint32& out_edge_count, uint32& out_face_count) {
    out_edge_count = 0;
    out_face_count = face_count_;
}

void AreaTermProvider::QueryTopology(const Database& db, Entity entity,
                                      uint32& out_edge_count, uint32& out_face_count) const {
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<AreaTriangle>());
    if (!storage) {
        out_edge_count = 0;
        out_face_count = 0;
        return;
    }
    out_edge_count = 0;
    // Scoped: entity has triangles → return only its count
    if (storage->GetArrayCount(entity) > 0) {
        out_face_count = storage->GetArrayCount(entity);
    } else {
        // Global: sum across ALL entities
        auto entities = storage->GetEntities();
        uint32 total = 0;
        for (Entity e : entities) {
            total += storage->GetArrayCount(e);
        }
        out_face_count = total;
    }
}

}  // namespace ext_newton
