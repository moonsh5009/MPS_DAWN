#pragma once

#include "core_database/component_storage.h"
#include "core_database/component_type.h"
#include "core_gpu/gpu_buffer.h"
#include "core_gpu/gpu_types.h"
#include <memory>
#include <span>
#include <string>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace simulate {

// Type-erased interface for a GPU buffer that mirrors host component data.
class IDeviceBufferEntry {
public:
    virtual ~IDeviceBufferEntry() = default;
    virtual void SyncFromHost(const database::IComponentStorage& storage) = 0;
    virtual WGPUBuffer GetBufferHandle() const = 0;
};

// Typed device buffer entry that owns a GPUBuffer<T> and syncs from host storage.
template<database::Component T>
class DeviceBufferEntry : public IDeviceBufferEntry {
public:
    explicit DeviceBufferEntry(gpu::BufferUsage extra_usage, const std::string& label)
        : usage_(gpu::BufferUsage::Storage | gpu::BufferUsage::CopySrc |
                 gpu::BufferUsage::CopyDst | extra_usage)
        , label_(label) {}

    void SyncFromHost(const database::IComponentStorage& storage) override {
        uint32 count = storage.GetDenseCount();

        if (count == 0) {
            // Nothing to sync; clear buffer if it exists
            if (buffer_) {
                buffer_->Clear();
            }
            return;
        }

        const auto* data = static_cast<const T*>(storage.GetDenseData());
        auto data_span = std::span<const T>(data, count);

        if (!buffer_) {
            // Lazily create the buffer with initial data
            buffer_ = std::make_unique<gpu::GPUBuffer<T>>(usage_, data_span, label_);
            return;
        }

        // Resize if element count changed, then write
        if (buffer_->GetCount() != static_cast<uint64>(count)) {
            buffer_->Resize(count);
        }
        buffer_->WriteData(data_span);
    }

    WGPUBuffer GetBufferHandle() const override {
        return buffer_ ? buffer_->GetHandle() : nullptr;
    }

private:
    gpu::BufferUsage usage_;
    std::string label_;
    std::unique_ptr<gpu::GPUBuffer<T>> buffer_;
};

}  // namespace simulate
}  // namespace mps
