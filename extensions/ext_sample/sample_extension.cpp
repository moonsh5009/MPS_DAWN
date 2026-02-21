#include "ext_sample/sample_extension.h"
#include "ext_sample/sample_components.h"
#include "ext_sample/sample_simulator.h"
#include "ext_sample/sample_renderer.h"
#include "core_system/system.h"
#include "core_database/database.h"
#include "core_util/logger.h"
#include <memory>
#include <cmath>

using namespace mps;
using namespace mps::util;
using namespace mps::system;
using namespace mps::database;

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

    // Create sample entities during registration (self-contained reference extension)
    system.Transact([](Database& db) {
        constexpr uint32 kEntityCount = 8;
        for (uint32 i = 0; i < kEntityCount; ++i) {
            Entity entity = db.CreateEntity();

            float32 angle = static_cast<float32>(i) * 6.2831853f / static_cast<float32>(kEntityCount);
            float32 radius = 2.0f;

            SampleTransform transform;
            transform.x = radius * std::cos(angle);
            transform.y = 0.0f;
            transform.z = radius * std::sin(angle);
            db.AddComponent<SampleTransform>(entity, transform);

            SampleVelocity velocity;
            velocity.vx = -std::sin(angle) * 0.5f;
            velocity.vy = 0.0f;
            velocity.vz = std::cos(angle) * 0.5f;
            db.AddComponent<SampleVelocity>(entity, velocity);
        }
        LogInfo("SampleExtension: created ", kEntityCount, " entities");
    });

    system.AddSimulator(std::make_unique<SampleSimulator>(system));
    system.AddRenderer(std::make_unique<SampleRenderer>(system_));
}

}  // namespace ext_sample
