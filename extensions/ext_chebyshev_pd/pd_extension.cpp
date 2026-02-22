#include "ext_chebyshev_pd/pd_extension.h"
#include "ext_chebyshev_pd/pd_system_simulator.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::system;

namespace ext_chebyshev_pd {

const std::string ChebyshevPDExtension::kName = "ext_chebyshev_pd";

ChebyshevPDExtension::ChebyshevPDExtension(System& system)
    : system_(system) {}

const std::string& ChebyshevPDExtension::GetName() const {
    return kName;
}

void ChebyshevPDExtension::Register(System& system) {
    system.AddSimulator(std::make_unique<ChebyshevPDSystemSimulator>(system_));
}

}  // namespace ext_chebyshev_pd
