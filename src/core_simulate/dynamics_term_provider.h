#pragma once

#include "core_simulate/dynamics_term.h"
#include "core_database/entity.h"
#include <memory>
#include <string_view>

namespace mps {
namespace database { class Database; }
namespace simulate {

// Interface for creating IDynamicsTerm instances from constraint entity data.
// Extensions register providers with System; the Newton system simulator
// discovers and instantiates terms automatically.
class IDynamicsTermProvider {
public:
    virtual ~IDynamicsTermProvider() = default;

    [[nodiscard]] virtual std::string_view GetTermName() const = 0;

    // Check if a constraint entity has this provider's configuration component
    [[nodiscard]] virtual bool HasConfig(const database::Database& db,
                                         database::Entity entity) const = 0;

    // Create a term instance from constraint entity data
    [[nodiscard]] virtual std::unique_ptr<IDynamicsTerm> CreateTerm(
        const database::Database& db, database::Entity entity, uint32 node_count) = 0;

    // Report topology contributions (edges, faces) for sparsity building
    virtual void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) {
        out_edge_count = 0;
        out_face_count = 0;
    }

    // Lightweight topology query (no GPU allocation). Override to check array sizes.
    virtual void QueryTopology(const database::Database& db, database::Entity entity,
                               uint32& out_edge_count, uint32& out_face_count) const {
        out_edge_count = 0;
        out_face_count = 0;
    }
};

}  // namespace simulate
}  // namespace mps
