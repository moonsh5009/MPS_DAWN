#pragma once

#include "core_simulate/dynamics_term_provider.h"
#include "ext_dynamics/spring_types.h"
#include <string_view>

namespace mps { namespace system { class System; } }

namespace ext_dynamics {

// Creates SpringTerm instances from SpringConstraintData entities.
// Edge topology is gathered from ALL entities with ArrayStorage<SpringEdge>,
// with per-entity position offsets applied for multi-mesh support.
class SpringTermProvider : public mps::simulate::IDynamicsTermProvider {
public:
    explicit SpringTermProvider(mps::system::System& system);

    [[nodiscard]] std::string_view GetTermName() const override;

    [[nodiscard]] bool HasConfig(const mps::database::Database& db,
                                 mps::database::Entity entity) const override;

    [[nodiscard]] std::unique_ptr<mps::simulate::IDynamicsTerm> CreateTerm(
        const mps::database::Database& db,
        mps::database::Entity entity,
        mps::uint32 node_count) override;

    void DeclareTopology(mps::uint32& out_edge_count, mps::uint32& out_face_count) override;

    void QueryTopology(const mps::database::Database& db, mps::database::Entity entity,
                       mps::uint32& out_edge_count, mps::uint32& out_face_count) const override;

private:
    mps::system::System& system_;
    mps::uint32 edge_count_ = 0;
};

}  // namespace ext_dynamics
