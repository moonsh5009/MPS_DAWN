#pragma once

#include "core_database/array_storage.h"
#include "core_database/component_type.h"
#include "core_database/database.h"
#include "core_gpu/gpu_buffer.h"
#include "core_gpu/gpu_types.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace simulate {

// Function type for applying index offsets to topology elements during GPU upload.
template<typename T>
using IndexOffsetFn = std::function<void(T& element, uint32 offset)>;

// Type-erased interface for a GPU buffer that mirrors concatenated host array data.
class IDeviceArrayEntry {
public:
    virtual ~IDeviceArrayEntry() = default;
    virtual void SyncFromHost(database::Database& db) = 0;
    virtual void ForceSyncFromHost(database::Database& db) = 0;
    virtual WGPUBuffer GetBufferHandle() const = 0;
    virtual uint32 GetTotalCount() const = 0;

    // Get offset of entity's data in concatenated buffer. Returns UINT32_MAX if not found.
    virtual uint32 GetEntityOffset(database::Entity entity) const = 0;

    // Signal that reference array layout changed (forces rebuild on next sync).
    virtual void MarkRefLayoutChanged() {}
};

// Region info: describes where a single entity's array lives in the concatenated buffer.
struct ArrayRegion {
    database::Entity entity = database::kInvalidEntity;
    uint32 offset = 0;
    uint32 count = 0;
};

// Concatenates per-entity ArrayStorage<T> data into a single contiguous GPU buffer.
// Entities are sorted by ID for deterministic layout.
template<database::Component T>
class DeviceArrayBuffer : public IDeviceArrayEntry {
public:
    explicit DeviceArrayBuffer(gpu::BufferUsage extra_usage, const std::string& label)
        : usage_(gpu::BufferUsage::Storage | gpu::BufferUsage::CopySrc |
                 gpu::BufferUsage::CopyDst | extra_usage)
        , label_(label) {}

    void SyncFromHost(database::Database& db) override {
        auto type_id = database::GetComponentTypeId<T>();
        const auto* storage = db.GetArrayStorageById(type_id);
        if (!storage) {
            if (buffer_) buffer_->Clear();
            regions_.clear();
            total_count_ = 0;
            return;
        }
        if (!storage->IsDirty() && !ref_layout_changed_ && buffer_) return;
        RebuildFromStorage(storage);
        ref_layout_changed_ = false;
    }

    void ForceSyncFromHost(database::Database& db) override {
        auto type_id = database::GetComponentTypeId<T>();
        const auto* storage = db.GetArrayStorageById(type_id);
        if (!storage) {
            if (buffer_) buffer_->Clear();
            regions_.clear();
            total_count_ = 0;
            return;
        }
        RebuildFromStorage(storage);
        ref_layout_changed_ = false;
    }

    WGPUBuffer GetBufferHandle() const override {
        return buffer_ ? buffer_->GetHandle() : nullptr;
    }

    uint32 GetTotalCount() const override {
        return total_count_;
    }

    uint32 GetEntityOffset(database::Entity entity) const override {
        const auto* region = GetRegion(entity);
        return region ? region->offset : UINT32_MAX;
    }

    void MarkRefLayoutChanged() override { ref_layout_changed_ = true; }

    const std::vector<ArrayRegion>& GetRegions() const { return regions_; }

    const ArrayRegion* GetRegion(database::Entity entity) const {
        for (const auto& r : regions_) {
            if (r.entity == entity) return &r;
        }
        return nullptr;
    }

    // Configure index offset transform: ref array provides entity offsets,
    // fn applies the offset to each element during GPU upload.
    void SetOffsetSource(IDeviceArrayEntry* ref, IndexOffsetFn<T> fn) {
        ref_array_ = ref;
        offset_fn_ = std::move(fn);
    }

private:
    void RebuildFromStorage(const database::IArrayStorage* storage) {
        auto entities = storage->GetEntities();
        std::sort(entities.begin(), entities.end());

        regions_.clear();
        std::vector<T> concat;
        uint32 offset = 0;

        for (database::Entity e : entities) {
            uint32 count = storage->GetArrayCount(e);
            if (count == 0) continue;
            const auto* data = static_cast<const T*>(storage->GetArrayData(e));
            regions_.push_back({e, offset, count});

            // Copy elements, applying index offset if configured
            uint32 node_offset = (ref_array_ && offset_fn_) ? ref_array_->GetEntityOffset(e) : 0;
            for (uint32 i = 0; i < count; ++i) {
                T elem = data[i];
                if (node_offset > 0) {
                    offset_fn_(elem, node_offset);
                }
                concat.push_back(elem);
            }

            offset += count;
        }
        total_count_ = offset;

        if (concat.empty()) {
            if (buffer_) buffer_->Clear();
            return;
        }

        auto data_span = std::span<const T>(concat);
        if (!buffer_) {
            buffer_ = std::make_unique<gpu::GPUBuffer<T>>(usage_, data_span, label_);
        } else {
            if (buffer_->GetCount() != static_cast<uint64>(total_count_)) {
                buffer_->Resize(total_count_);
            }
            buffer_->WriteData(data_span);
        }
    }

    gpu::BufferUsage usage_;
    std::string label_;
    std::unique_ptr<gpu::GPUBuffer<T>> buffer_;
    std::vector<ArrayRegion> regions_;
    uint32 total_count_ = 0;

    // Offset transform support
    IDeviceArrayEntry* ref_array_ = nullptr;
    IndexOffsetFn<T> offset_fn_;
    bool ref_layout_changed_ = false;
};

}  // namespace simulate
}  // namespace mps
