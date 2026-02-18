#include "ext_sample/sample_simulator.h"
#include "ext_sample/sample_components.h"
#include "core_database/database.h"
#include "core_util/logger.h"

using namespace mps;
using namespace mps::database;
using namespace mps::util;

namespace ext_sample {

const std::string SampleSimulator::kName = "SampleSimulator";

const std::string& SampleSimulator::GetName() const {
    return kName;
}

void SampleSimulator::Initialize(Database& db) {
    // Create sample entities with transform + velocity components
    constexpr uint32 kEntityCount = 8;
    for (uint32 i = 0; i < kEntityCount; ++i) {
        Entity entity = db.CreateEntity();

        float32 angle = static_cast<float32>(i) * 6.2831853f / static_cast<float32>(kEntityCount);
        float32 radius = 2.0f;

        SampleTransform transform;
        transform.x = radius * cosf(angle);
        transform.y = 0.0f;
        transform.z = radius * sinf(angle);
        db.AddComponent<SampleTransform>(entity, transform);

        SampleVelocity velocity;
        velocity.vx = -sinf(angle) * 0.5f;
        velocity.vy = 0.0f;
        velocity.vz = cosf(angle) * 0.5f;
        db.AddComponent<SampleVelocity>(entity, velocity);
    }

    LogInfo("SampleSimulator: created ", kEntityCount, " entities");
}

void SampleSimulator::Update(Database& db, float32 dt) {
    // Query all entities with both SampleTransform and SampleVelocity
    auto* transform_storage = db.GetStorageById(GetComponentTypeId<SampleTransform>());
    if (!transform_storage) return;

    uint32 count = transform_storage->GetDenseCount();
    if (count == 0) return;

    // Get entity list from the transform storage
    auto* typed_storage = static_cast<ComponentStorage<SampleTransform>*>(
        db.GetStorageById(GetComponentTypeId<SampleTransform>()));
    const auto& entities = typed_storage->GetEntities();

    for (uint32 i = 0; i < count; ++i) {
        Entity entity = entities[i];
        auto* transform = db.GetComponent<SampleTransform>(entity);
        auto* velocity = db.GetComponent<SampleVelocity>(entity);
        if (!transform || !velocity) continue;

        SampleTransform new_transform = *transform;
        new_transform.x += velocity->vx * dt;
        new_transform.y += velocity->vy * dt;
        new_transform.z += velocity->vz * dt;
        db.SetComponent<SampleTransform>(entity, new_transform);
    }
}

}  // namespace ext_sample
