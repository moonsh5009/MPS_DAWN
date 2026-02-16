#pragma once

#include "core_gpu/gpu_types.h"
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <vector>

// Forward-declare WebGPU handle types
struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;
struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;

namespace mps {
namespace gpu {

struct BufferConfig {
    BufferUsage usage = BufferUsage::Vertex;
    uint64 size = 0;
    bool mapped_at_creation = false;
    std::string label;
};

// Non-template core — WGPU calls live in .cpp (no webgpu.h in headers)
class GPUBufferCore {
public:
    explicit GPUBufferCore(const BufferConfig& config);
    ~GPUBufferCore();

    GPUBufferCore(GPUBufferCore&& other) noexcept;
    GPUBufferCore& operator=(GPUBufferCore&& other) noexcept;
    GPUBufferCore(const GPUBufferCore&) = delete;
    GPUBufferCore& operator=(const GPUBufferCore&) = delete;

    // Data operations
    void WriteRawData(const void* data, uint64 size_bytes, uint64 byte_offset = 0);
    void CopyTo(GPUBufferCore& dest, uint64 src_offset = 0,
                uint64 dst_offset = 0, uint64 size_bytes = 0) const;
    void CopyTo(WGPUCommandEncoder encoder, GPUBufferCore& dest,
                uint64 src_offset = 0, uint64 dst_offset = 0, uint64 size_bytes = 0) const;
    std::vector<uint8> ReadRawToHost() const;
    void ReadRawToHostAsync(std::function<void(std::vector<uint8>)> callback) const;

    // Capacity management
    void Reserve(uint64 min_capacity_bytes);
    void Resize(uint64 new_size_bytes);
    void SetSize(uint64 new_size_bytes);    // no-copy — destroys old data
    void Clear();                            // logical clear (size=0, keeps buffer)
    void ShrinkToFit();                      // trim capacity to match size

    // Accessors
    WGPUBuffer GetHandle() const;
    uint64 GetSize() const;
    uint64 GetCapacity() const;
    BufferUsage GetUsage() const;
    bool IsEmpty() const;
    bool IsValid() const;

private:
    void Release();
    void Grow(uint64 min_capacity);
    static uint64 AlignUp(uint64 value, uint64 alignment);

    WGPUBuffer handle_ = nullptr;
    uint64 size_ = 0;
    uint64 capacity_ = 0;
    BufferUsage usage_ = BufferUsage::None;
};

// Typed buffer — element type is baked into the class
template<typename T>
class GPUBuffer {
    template<typename U> friend class GPUBuffer;

public:
    // From typed data (auto-adds CopyDst for initial upload)
    GPUBuffer(BufferUsage usage, std::span<const T> data, const std::string& label = "")
        : core_(BufferConfig{
              .usage = usage | BufferUsage::CopyDst,
              .size = static_cast<uint64>(data.size_bytes()),
              .label = label
          }) {
        if (!data.empty()) {
            core_.WriteRawData(data.data(), data.size_bytes(), 0);
        }
    }

    // Raw config (for empty/pre-sized buffers)
    explicit GPUBuffer(const BufferConfig& config)
        : core_(config) {}

    // Write typed data
    void WriteData(std::span<const T> data, uint64 element_offset = 0) {
        core_.WriteRawData(data.data(), data.size_bytes(), element_offset * sizeof(T));
    }

    // Sync read back as typed vector
    std::vector<T> ReadToHost() const {
        auto raw = core_.ReadRawToHost();
        std::vector<T> result(raw.size() / sizeof(T));
        std::memcpy(result.data(), raw.data(), result.size() * sizeof(T));
        return result;
    }

    // Async read back as typed vector (callback fires during ProcessEvents)
    void ReadToHostAsync(std::function<void(std::vector<T>)> callback) const {
        core_.ReadRawToHostAsync([cb = std::move(callback)](std::vector<uint8> raw) {
            std::vector<T> result(raw.size() / sizeof(T));
            std::memcpy(result.data(), raw.data(), result.size() * sizeof(T));
            cb(std::move(result));
        });
    }

    // GPU-to-GPU copy (cross-type allowed — byte-level operation)
    template<typename U>
    void CopyTo(GPUBuffer<U>& dest, uint64 src_offset = 0,
                uint64 dst_offset = 0, uint64 size_bytes = 0) const {
        core_.CopyTo(dest.core_, src_offset, dst_offset, size_bytes);
    }

    // Encoder-based GPU-to-GPU copy (for batching into external command encoder)
    template<typename U>
    void CopyTo(WGPUCommandEncoder encoder, GPUBuffer<U>& dest,
                uint64 src_offset = 0, uint64 dst_offset = 0, uint64 size_bytes = 0) const {
        core_.CopyTo(encoder, dest.core_, src_offset, dst_offset, size_bytes);
    }

    // Capacity management (element count)
    void Reserve(uint64 element_count) {
        core_.Reserve(element_count * sizeof(T));
    }

    void Resize(uint64 element_count) {
        core_.Resize(element_count * sizeof(T));
    }

    void SetSize(uint64 element_count) {
        core_.SetSize(element_count * sizeof(T));
    }

    void Clear() { core_.Clear(); }
    void ShrinkToFit() { core_.ShrinkToFit(); }

    // Accessors
    WGPUBuffer GetHandle() const { return core_.GetHandle(); }
    uint64 GetSize() const { return core_.GetSize(); }
    uint64 GetByteLength() const { return core_.GetSize(); }
    uint64 GetCount() const { return core_.GetSize() / sizeof(T); }
    uint64 GetCapacity() const { return core_.GetCapacity() / sizeof(T); }
    BufferUsage GetUsage() const { return core_.GetUsage(); }
    bool IsEmpty() const { return core_.IsEmpty(); }
    bool IsValid() const { return core_.IsValid(); }

private:
    GPUBufferCore core_;
};

}  // namespace gpu
}  // namespace mps
