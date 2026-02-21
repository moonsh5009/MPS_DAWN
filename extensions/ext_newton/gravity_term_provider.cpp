#include "ext_newton/gravity_term_provider.h"
#include "ext_newton/gravity_constraint.h"
#include "ext_dynamics/gravity_term.h"
#include "core_database/database.h"

using namespace mps;
using namespace mps::database;
using namespace mps::simulate;

namespace ext_newton {

std::string_view GravityTermProvider::GetTermName() const {
    return "GravityTermProvider";
}

bool GravityTermProvider::HasConfig(const Database& db, Entity entity) const {
    return db.HasComponent<GravityConstraintData>(entity);
}

std::unique_ptr<IDynamicsTerm> GravityTermProvider::CreateTerm(
    const Database& /* db */, Entity /* entity */, uint32 /* node_count */) {
    // GravityTerm reads gravity from the params uniform (set by the solver).
    // The GravityConstraintData values are used by NewtonSystemSimulator
    // to configure DynamicsParams.gravity_{x,y,z}.
    return std::make_unique<ext_dynamics::GravityTerm>();
}

}  // namespace ext_newton
