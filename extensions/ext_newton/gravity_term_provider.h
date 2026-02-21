#pragma once

#include "core_simulate/dynamics_term_provider.h"
#include <string_view>

namespace ext_newton {

// Creates GravityTerm instances from GravityConstraintData entities.
class GravityTermProvider : public mps::simulate::IDynamicsTermProvider {
public:
    [[nodiscard]] std::string_view GetTermName() const override;

    [[nodiscard]] bool HasConfig(const mps::database::Database& db,
                                 mps::database::Entity entity) const override;

    [[nodiscard]] std::unique_ptr<mps::simulate::IDynamicsTerm> CreateTerm(
        const mps::database::Database& db,
        mps::database::Entity entity,
        mps::uint32 node_count) override;
};

}  // namespace ext_newton
