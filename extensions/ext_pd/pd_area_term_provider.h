#pragma once

#include "core_simulate/projective_term_provider.h"
#include <string_view>

namespace mps { namespace system { class System; } }

namespace ext_pd {

class PDAreaTermProvider : public mps::simulate::IProjectiveTermProvider {
public:
    explicit PDAreaTermProvider(mps::system::System& system);

    [[nodiscard]] std::string_view GetTermName() const override;
    [[nodiscard]] bool HasConfig(const mps::database::Database& db,
                                 mps::database::Entity entity) const override;
    [[nodiscard]] std::unique_ptr<mps::simulate::IProjectiveTerm> CreateTerm(
        const mps::database::Database& db,
        mps::database::Entity entity,
        mps::uint32 node_count) override;
    void DeclareTopology(mps::uint32& out_edge_count, mps::uint32& out_face_count) override;
    void QueryTopology(const mps::database::Database& db, mps::database::Entity entity,
                       mps::uint32& out_edge_count, mps::uint32& out_face_count) const override;

private:
    mps::system::System& system_;
    mps::uint32 face_count_ = 0;
};

}  // namespace ext_pd
