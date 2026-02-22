#include "ext_pd/pd_area_term_provider.h"
#include "ext_pd/pd_area_term.h"
#include "ext_dynamics/area_constraint.h"
#include "ext_dynamics/area_types.h"
#include "core_simulate/sim_components.h"
#include "core_database/database.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <algorithm>

using namespace mps;
using namespace mps::util;
using namespace mps::database;
using namespace mps::simulate;

namespace ext_pd {

PDAreaTermProvider::PDAreaTermProvider(system::System& system)
    : system_(system) {}

std::string_view PDAreaTermProvider::GetTermName() const {
    return "PDAreaTermProvider";
}

bool PDAreaTermProvider::HasConfig(const Database& db, Entity entity) const {
    return db.HasComponent<ext_dynamics::AreaConstraintData>(entity);
}

std::unique_ptr<IProjectiveTerm> PDAreaTermProvider::CreateTerm(
    const Database& db, Entity entity, uint32 /* node_count */) {

    const auto* config = db.GetComponent<ext_dynamics::AreaConstraintData>(entity);
    if (!config) {
        LogError("PDAreaTermProvider: no AreaConstraintData on entity ", entity);
        return nullptr;
    }

    // Gather triangles: if constraint entity itself has AreaTriangle data, scope to it only
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<ext_dynamics::AreaTriangle>());
    if (!storage) return nullptr;

    std::vector<ext_dynamics::AreaTriangle> all_triangles;
    bool scoped = storage->GetArrayCount(entity) > 0;

    if (scoped) {
        // Scoped: use only this entity's triangles with local 0-based indices (no offset)
        uint32 count = storage->GetArrayCount(entity);
        const auto* data = static_cast<const ext_dynamics::AreaTriangle*>(storage->GetArrayData(entity));
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
            const auto* data = static_cast<const ext_dynamics::AreaTriangle*>(storage->GetArrayData(mesh_e));

            uint32 node_offset = 0;
            auto it = pos_offset_map.find(mesh_e);
            if (it != pos_offset_map.end()) {
                node_offset = it->second;
            }

            for (uint32 i = 0; i < count; ++i) {
                ext_dynamics::AreaTriangle g = data[i];
                g.n0 += node_offset;
                g.n1 += node_offset;
                g.n2 += node_offset;
                all_triangles.push_back(g);
            }
        }
    }

    if (all_triangles.empty()) return nullptr;

    face_count_ = static_cast<uint32>(all_triangles.size());
    return std::make_unique<PDAreaTerm>(all_triangles, config->stiffness);
}

void PDAreaTermProvider::DeclareTopology(uint32& out_edge_count, uint32& out_face_count) {
    out_edge_count = 0;
    out_face_count = face_count_;
}

void PDAreaTermProvider::QueryTopology(const Database& db, Entity entity,
                                        uint32& out_edge_count, uint32& out_face_count) const {
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<ext_dynamics::AreaTriangle>());
    if (!storage) {
        out_edge_count = 0;
        out_face_count = 0;
        return;
    }
    out_edge_count = 0;
    if (storage->GetArrayCount(entity) > 0) {
        out_face_count = storage->GetArrayCount(entity);
    } else {
        auto entities = storage->GetEntities();
        uint32 total = 0;
        for (Entity e : entities) {
            total += storage->GetArrayCount(e);
        }
        out_face_count = total;
    }
}

}  // namespace ext_pd
