#include "ext_sample/sample_extension.h"
#include "ext_sample/sample_components.h"
#include "ext_sample/sample_simulator.h"
#include "ext_sample/sample_renderer.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::system;

namespace ext_sample {

const std::string SampleExtension::kName = "ext_sample";

SampleExtension::SampleExtension(System& system)
    : system_(system) {}

const std::string& SampleExtension::GetName() const {
    return kName;
}

void SampleExtension::Register(System& system) {
    system.RegisterComponent<SampleTransform>(gpu::BufferUsage::Vertex, "sample_transform");
    system.RegisterComponent<SampleVelocity>(gpu::BufferUsage::Storage, "sample_velocity");

    system.AddSimulator(std::make_unique<SampleSimulator>());
    system.AddRenderer(std::make_unique<SampleRenderer>(system_));
}

}  // namespace ext_sample
