#include "ext_dynamics/spring_term_provider.h"
#include "ext_dynamics/spring_constraint.h"
#include "ext_dynamics/spring_term.h"
#include "core_simulate/sim_components.h"
#include "core_database/database.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <algorithm>

using namespace mps;
using namespace mps::util;
using namespace mps::database;
using namespace mps::simulate;

namespace ext_dynamics {

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

    // Gather edges from ALL entities with ArrayStorage<SpringEdge>
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<SpringEdge>());
    if (!storage) return nullptr;

    auto entities = storage->GetEntities();
    std::sort(entities.begin(), entities.end());

    // Build position offset map (same entity-ID-sorted order as DeviceArrayBuffer)
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

    // Merge all edges with global indices
    std::vector<SpringEdge> all_edges;
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

    if (all_edges.empty()) return nullptr;

    edge_count_ = static_cast<uint32>(all_edges.size());
    return std::make_unique<SpringTerm>(all_edges, config->stiffness);
}

void SpringTermProvider::DeclareTopology(uint32& out_edge_count, uint32& out_face_count) {
    out_edge_count = edge_count_;
    out_face_count = 0;
}

void SpringTermProvider::QueryTopology(const Database& db, Entity /* entity */,
                                        uint32& out_edge_count, uint32& out_face_count) const {
    // Sum SpringEdge arrays across ALL entities
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<SpringEdge>());
    if (!storage) {
        out_edge_count = 0;
        out_face_count = 0;
        return;
    }
    auto entities = storage->GetEntities();
    uint32 total = 0;
    for (Entity e : entities) {
        total += storage->GetArrayCount(e);
    }
    out_edge_count = total;
    out_face_count = 0;
}

}  // namespace ext_dynamics
