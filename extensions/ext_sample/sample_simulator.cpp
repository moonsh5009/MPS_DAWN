#include "ext_sample/sample_simulator.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::util;

namespace ext_sample {

const std::string SampleSimulator::kName = "SampleSimulator";

SampleSimulator::SampleSimulator(system::System& system)
    : system_(system) {}

const std::string& SampleSimulator::GetName() const {
    return kName;
}

void SampleSimulator::Initialize() {
    // GPU pipeline setup would go here
    LogInfo("SampleSimulator: initialized");
}

void SampleSimulator::Update(float32 dt) {
    // GPU compute dispatches would go here
}

}  // namespace ext_sample
