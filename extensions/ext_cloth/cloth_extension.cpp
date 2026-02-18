#include "ext_cloth/cloth_extension.h"
#include "ext_cloth/cloth_components.h"
#include "ext_cloth/cloth_simulator.h"
#include "ext_cloth/cloth_renderer.h"
#include "core_system/system.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::system;

namespace ext_cloth {

// Global simulator pointer for renderer access (module-internal linkage)
ClothSimulator* g_cloth_simulator = nullptr;

const std::string ClothExtension::kName = "ext_cloth";

ClothExtension::ClothExtension(System& system)
    : system_(system) {}

const std::string& ClothExtension::GetName() const {
    return kName;
}

void ClothExtension::Register(System& system) {
    // Register ECS components with GPU buffer usage flags
    // Position needs Vertex (for rendering) + Storage (for compute)
    system.RegisterComponent<ClothPosition>(
        gpu::BufferUsage::Storage | gpu::BufferUsage::Vertex, "cloth_position");
    system.RegisterComponent<ClothVelocity>(
        gpu::BufferUsage::Storage, "cloth_velocity");
    system.RegisterComponent<ClothMass>(
        gpu::BufferUsage::Storage, "cloth_mass");

    auto simulator = std::make_unique<ClothSimulator>(system_);
    g_cloth_simulator = simulator.get();
    system.AddSimulator(std::move(simulator));
    system.AddRenderer(std::make_unique<ClothRenderer>(system_));
}

}  // namespace ext_cloth
