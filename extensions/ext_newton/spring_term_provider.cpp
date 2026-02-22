#include "ext_newton/spring_term_provider.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_newton/spring_term.h"
#include "core_simulate/sim_components.h"
#include "core_database/database.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <algorithm>

using namespace mps;
using namespace mps::util;
using namespace mps::database;
using namespace mps::simulate;

using ext_dynamics::SpringConstraintData;
using ext_dynamics::SpringEdge;

namespace ext_newton {

SpringTermProvider::SpringTermProvider(system::System& system)
    : system_(system) {}

std::string_view SpringTermProvider::GetTermName() const {
    return "SpringTermProvider";
}

bool SpringTermProvider::HasConfig(const Database& db, Entity entity) const {
    return db.HasComponent<SpringConstraintData>(entity);
}

std::unique_ptr<IDynamicsTerm> SpringTermProvider::CreateTerm(
    const Database& db, Entity entity, uint32 /* node_count */) {

    // Config (stiffness) read from constraint entity
    const auto* config = db.GetComponent<SpringConstraintData>(entity);
    if (!config) {
        LogError("SpringTermProvider: no SpringConstraintData on entity ", entity);
        return nullptr;
    }

    // Gather edges: if constraint entity itself has SpringEdge data, scope to it only
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<SpringEdge>());
    if (!storage) return nullptr;

    std::vector<SpringEdge> all_edges;
    bool scoped = storage->GetArrayCount(entity) > 0;

    if (scoped) {
        // Scoped: use only this entity's edges with local 0-based indices (no offset)
        uint32 count = storage->GetArrayCount(entity);
        const auto* data = static_cast<const SpringEdge*>(storage->GetArrayData(entity));
        all_edges.assign(data, data + count);
    } else {
        // Global: merge ALL entities' edges with position offsets
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
            const auto* data = static_cast<const SpringEdge*>(storage->GetArrayData(mesh_e));

            uint32 node_offset = 0;
            auto it = pos_offset_map.find(mesh_e);
            if (it != pos_offset_map.end()) {
                node_offset = it->second;
            }

            for (uint32 i = 0; i < count; ++i) {
                SpringEdge g = data[i];
                g.n0 += node_offset;
                g.n1 += node_offset;
                all_edges.push_back(g);
            }
        }
    }

    if (all_edges.empty()) return nullptr;

    edge_count_ = static_cast<uint32>(all_edges.size());
    return std::make_unique<SpringTerm>(all_edges, config->stiffness);
}

void SpringTermProvider::DeclareTopology(uint32& out_edge_count, uint32& out_face_count) {
    out_edge_count = edge_count_;
    out_face_count = 0;
}

void SpringTermProvider::QueryTopology(const Database& db, Entity entity,
                                        uint32& out_edge_count, uint32& out_face_count) const {
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<SpringEdge>());
    if (!storage) {
        out_edge_count = 0;
        out_face_count = 0;
        return;
    }
    // Scoped: entity has edges → return only its count
    if (storage->GetArrayCount(entity) > 0) {
        out_edge_count = storage->GetArrayCount(entity);
    } else {
        // Global: sum across ALL entities
        auto entities = storage->GetEntities();
        uint32 total = 0;
        for (Entity e : entities) {
            total += storage->GetArrayCount(e);
        }
        out_edge_count = total;
    }
    out_face_count = 0;
}

}  // namespace ext_newton
