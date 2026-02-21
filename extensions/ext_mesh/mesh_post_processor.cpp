#include "ext_mesh/mesh_post_processor.h"
#include "ext_mesh/mesh_component.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_database/database.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <span>

using namespace mps;
using namespace mps::util;
using namespace mps::gpu;
using namespace mps::simulate;
using namespace mps::database;

namespace ext_mesh {

const std::string MeshPostProcessor::kName = "MeshPostProcessor";

MeshPostProcessor::MeshPostProcessor(system::System& system)
    : system_(system) {}

MeshPostProcessor::~MeshPostProcessor() = default;

const std::string& MeshPostProcessor::GetName() const {
    return kName;
}

void MeshPostProcessor::Initialize() {
    node_count_ = system_.GetArrayTotalCount<SimPosition>();
    total_face_count_ = system_.GetArrayTotalCount<MeshFace>();

    if (node_count_ == 0 || total_face_count_ == 0) {
        LogInfo("MeshPostProcessor: no mesh data found, skipping initialization");
        return;
    }

    // Normal computation
    normals_ = std::make_unique<NormalComputer>();
    normals_->Initialize(node_count_, total_face_count_);

    // Build face index buffer from DeviceDB's globally-indexed MeshFace buffer.
    // DeviceDB has already applied per-entity offsets via RegisterIndexedArray.
    // We read from the host DB and apply offsets ourselves for the index buffer.
    const auto& db = system_.GetDatabase();
    auto* face_storage = db.GetArrayStorageById(GetComponentTypeId<MeshFace>());
    if (!face_storage) {
        LogInfo("MeshPostProcessor: no face storage, skipping initialization");
        return;
    }

    auto face_entities = face_storage->GetEntities();
    std::sort(face_entities.begin(), face_entities.end());

    // Build position offset map
    auto* pos_storage = db.GetArrayStorageById(GetComponentTypeId<SimPosition>());
    std::unordered_map<Entity, uint32> pos_offset_map;
    if (pos_storage) {
        auto pos_entities = pos_storage->GetEntities();
        std::sort(pos_entities.begin(), pos_entities.end());
        uint32 offset = 0;
        for (Entity e : pos_entities) {
            pos_offset_map[e] = offset;
            offset += pos_storage->GetArrayCount(e);
        }
    }

    // Build globally-indexed face index buffer
    std::vector<uint32> face_idx;
    face_idx.reserve(static_cast<size_t>(total_face_count_) * 3);
    for (Entity e : face_entities) {
        uint32 count = face_storage->GetArrayCount(e);
        if (count == 0) continue;
        const auto* data = static_cast<const MeshFace*>(face_storage->GetArrayData(e));

        uint32 node_offset = 0;
        auto it = pos_offset_map.find(e);
        if (it != pos_offset_map.end()) {
            node_offset = it->second;
        }

        for (uint32 i = 0; i < count; ++i) {
            face_idx.push_back(data[i].n0 + node_offset);
            face_idx.push_back(data[i].n1 + node_offset);
            face_idx.push_back(data[i].n2 + node_offset);
        }
    }

    face_index_buffer_ = std::make_unique<GPUBuffer<uint32>>(
        BufferUsage::Index | BufferUsage::Storage,
        std::span<const uint32>(face_idx), "mesh_face_idx");

    initialized_ = true;
    LogInfo("MeshPostProcessor: initialized (", node_count_, " nodes, ", total_face_count_, " faces)");
}

void MeshPostProcessor::Update(float32 /* dt */) {
    if (!initialized_) return;

    WGPUBuffer pos_h = system_.GetDeviceBuffer<SimPosition>();
    WGPUBuffer face_h = system_.GetDeviceBuffer<MeshFace>();
    if (!pos_h || !face_h) return;

    auto& gpu = GPUCore::GetInstance();

    // Create command encoder
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    enc_desc.label = {"normals_compute", 15};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.GetDevice(), &enc_desc);

    uint64 pos_sz = uint64(node_count_) * sizeof(SimPosition);
    uint64 face_sz = uint64(total_face_count_) * sizeof(MeshFace);

    normals_->Compute(encoder, pos_h, pos_sz, face_h, face_sz);

    // Submit
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(gpu.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}

WGPUBuffer MeshPostProcessor::GetNormalBuffer() const {
    return normals_ ? normals_->GetNormalBuffer() : nullptr;
}

WGPUBuffer MeshPostProcessor::GetIndexBuffer() const {
    return face_index_buffer_ ? face_index_buffer_->GetHandle() : nullptr;
}

uint32 MeshPostProcessor::GetFaceCount() const {
    return total_face_count_;
}

uint32 MeshPostProcessor::ComputeTotalFaceCount() const {
    const auto& db = system_.GetDatabase();
    auto* storage = db.GetArrayStorageById(GetComponentTypeId<MeshFace>());
    if (!storage) return 0;
    auto entities = storage->GetEntities();
    uint32 total = 0;
    for (Entity e : entities) {
        total += storage->GetArrayCount(e);
    }
    return total;
}

void MeshPostProcessor::OnDatabaseChanged() {
    uint32 new_node_count = system_.GetArrayTotalCount<SimPosition>();
    uint32 new_face_count = ComputeTotalFaceCount();

    if (!initialized_) {
        if (new_node_count > 0 && new_face_count > 0) {
            Initialize();
        }
        return;
    }

    if (new_node_count == node_count_ && new_face_count == total_face_count_) {
        return;
    }

    LogInfo("MeshPostProcessor: topology changed, reinitializing...");
    Shutdown();
    Initialize();
}

void MeshPostProcessor::Shutdown() {
    if (normals_) normals_->Shutdown();
    normals_.reset();

    face_index_buffer_.reset();

    initialized_ = false;
    LogInfo("MeshPostProcessor: shutdown");
}

}  // namespace ext_mesh
