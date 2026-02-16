#include "core_gpu/gpu_buffer.h"
#include "core_gpu/gpu_core.h"
#include "core_util/logger.h"
#include <webgpu/webgpu.h>
#include <algorithm>
#include <cassert>
#include <utility>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Helpers ------------------------------------------------------------------

uint64 GPUBufferCore::AlignUp(uint64 value, uint64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// -- Construction -------------------------------------------------------------

GPUBufferCore::GPUBufferCore(const BufferConfig& config)
    : size_(config.size), capacity_(config.size), usage_(config.usage) {
    auto& core = GPUCore::GetInstance();
    assert(core.IsInitialized());

    if (capacity_ == 0) return;  // deferred allocation — Grow on first Reserve/Resize

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.label = {config.label.data(), config.label.size()};
    // Always add CopySrc internally so Grow can copy old data to new buffer
    desc.usage = static_cast<WGPUBufferUsage>(config.usage) | WGPUBufferUsage_CopySrc;
    desc.size = config.size;
    desc.mappedAtCreation = config.mapped_at_creation ? WGPU_TRUE : WGPU_FALSE;

    handle_ = wgpuDeviceCreateBuffer(core.GetDevice(), &desc);
    if (!handle_) {
        throw GPUException("Failed to create GPU buffer: " + config.label);
    }

    LogInfo("GPUBuffer created: ", config.label, " (", config.size, " bytes)");
}

GPUBufferCore::~GPUBufferCore() {
    Release();
}

// -- Move semantics -----------------------------------------------------------

GPUBufferCore::GPUBufferCore(GPUBufferCore&& other) noexcept
    : handle_(other.handle_), size_(other.size_),
      capacity_(other.capacity_), usage_(other.usage_) {
    other.handle_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.usage_ = BufferUsage::None;
}

GPUBufferCore& GPUBufferCore::operator=(GPUBufferCore&& other) noexcept {
    if (this != &other) {
        Release();
        handle_ = other.handle_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        usage_ = other.usage_;
        other.handle_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.usage_ = BufferUsage::None;
    }
    return *this;
}

// -- Data operations ----------------------------------------------------------

void GPUBufferCore::WriteRawData(const void* data, uint64 size_bytes, uint64 byte_offset) {
    assert(handle_);
    auto& core = GPUCore::GetInstance();
    wgpuQueueWriteBuffer(core.GetQueue(), handle_, byte_offset, data, static_cast<size_t>(size_bytes));
}

void GPUBufferCore::CopyTo(GPUBufferCore& dest, uint64 src_offset,
                            uint64 dst_offset, uint64 size_bytes) const {
    assert(handle_);
    assert(dest.handle_);

    auto& core = GPUCore::GetInstance();
    uint64 copy_size = (size_bytes == 0) ? size_ : size_bytes;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, src_offset,
                                          dest.handle_, dst_offset, copy_size);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}

void GPUBufferCore::CopyTo(WGPUCommandEncoder encoder, GPUBufferCore& dest,
                            uint64 src_offset, uint64 dst_offset, uint64 size_bytes) const {
    assert(handle_);
    assert(dest.handle_);
    assert(encoder);

    uint64 copy_size = (size_bytes == 0) ? size_ : size_bytes;
    wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, src_offset,
                                          dest.handle_, dst_offset, copy_size);
}

std::vector<uint8> GPUBufferCore::ReadRawToHost() const {
    assert(handle_);

    auto& core = GPUCore::GetInstance();

    // Create staging buffer with MapRead | CopyDst
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.size = size_;

    WGPUBuffer staging = wgpuDeviceCreateBuffer(core.GetDevice(), &staging_desc);
    if (!staging) {
        throw GPUException("Failed to create staging buffer for readback");
    }

    // Copy source -> staging
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, 0, staging, 0, size_);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Map the staging buffer (synchronous)
    struct MapContext {
        bool done = false;
        bool success = false;
    };
    MapContext ctx;

    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
#ifdef __EMSCRIPTEN__
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
#else
    map_cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*message*/,
                         void* userdata1, void* /*userdata2*/) {
        auto* c = static_cast<MapContext*>(userdata1);
        c->done = true;
        c->success = (status == WGPUMapAsyncStatus_Success);
    };
    map_cb.userdata1 = &ctx;

    WGPUFuture future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0,
                                            static_cast<size_t>(size_), map_cb);

#ifndef __EMSCRIPTEN__
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = future;
    wgpuInstanceWaitAny(core.GetWGPUInstance(), 1, &wait, UINT64_MAX);
#else
    while (!ctx.done) {
        core.ProcessEvents();
    }
#endif

    if (!ctx.success) {
        wgpuBufferRelease(staging);
        throw GPUException("Failed to map staging buffer for readback");
    }

    // Read data
    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, static_cast<size_t>(size_));
    std::vector<uint8> result(static_cast<size_t>(size_));
    std::memcpy(result.data(), mapped, result.size());

    wgpuBufferUnmap(staging);
    wgpuBufferRelease(staging);

    return result;
}

void GPUBufferCore::ReadRawToHostAsync(std::function<void(std::vector<uint8>)> callback) const {
    assert(handle_);

    auto& core = GPUCore::GetInstance();

    // Create staging buffer with MapRead | CopyDst
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    staging_desc.size = size_;

    WGPUBuffer staging = wgpuDeviceCreateBuffer(core.GetDevice(), &staging_desc);
    if (!staging) {
        throw GPUException("Failed to create staging buffer for async readback");
    }

    // Copy source -> staging (submit immediately)
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, 0, staging, 0, size_);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Heap-allocate context to outlive this scope
    struct AsyncReadContext {
        WGPUBuffer staging;
        uint64 size;
        std::function<void(std::vector<uint8>)> callback;
    };
    auto* ctx = new AsyncReadContext{staging, size_, std::move(callback)};

    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*message*/,
                         void* userdata1, void* /*userdata2*/) {
        auto* c = static_cast<AsyncReadContext*>(userdata1);
        if (status == WGPUMapAsyncStatus_Success) {
            const void* mapped = wgpuBufferGetConstMappedRange(
                c->staging, 0, static_cast<size_t>(c->size));
            std::vector<uint8> result(static_cast<size_t>(c->size));
            std::memcpy(result.data(), mapped, result.size());
            wgpuBufferUnmap(c->staging);
            c->callback(std::move(result));
        } else {
            c->callback({});
        }
        wgpuBufferRelease(c->staging);
        delete c;
    };
    map_cb.userdata1 = ctx;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0,
                        static_cast<size_t>(size_), map_cb);
}

// -- Capacity management ------------------------------------------------------

void GPUBufferCore::Reserve(uint64 min_capacity_bytes) {
    if (min_capacity_bytes > capacity_) {
        Grow(min_capacity_bytes);
    }
}

void GPUBufferCore::Resize(uint64 new_size_bytes) {
    if (new_size_bytes > capacity_) {
        Grow(new_size_bytes);
    }
    size_ = new_size_bytes;
}

void GPUBufferCore::SetSize(uint64 new_size_bytes) {
    if (new_size_bytes <= capacity_) {
        size_ = new_size_bytes;
        return;
    }

    // Need larger buffer — no data preservation
    Release();

    auto& core = GPUCore::GetInstance();
    uint64 new_capacity = AlignUp(new_size_bytes, 16);

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = static_cast<WGPUBufferUsage>(usage_) | WGPUBufferUsage_CopySrc;
    desc.size = new_capacity;

    handle_ = wgpuDeviceCreateBuffer(core.GetDevice(), &desc);
    if (!handle_) {
        throw GPUException("Failed to create GPU buffer in SetSize");
    }

    size_ = new_size_bytes;
    capacity_ = new_capacity;

    LogInfo("GPUBuffer SetSize: ", new_capacity, " bytes (no copy)");
}

void GPUBufferCore::Clear() {
    size_ = 0;
}

void GPUBufferCore::ShrinkToFit() {
    if (size_ == 0) {
        Release();
        capacity_ = 0;
        return;
    }

    uint64 target_capacity = AlignUp(size_, 16);
    if (target_capacity >= capacity_) return;

    auto& core = GPUCore::GetInstance();

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = static_cast<WGPUBufferUsage>(usage_) | WGPUBufferUsage_CopySrc;
    desc.size = target_capacity;

    WGPUBuffer new_handle = wgpuDeviceCreateBuffer(core.GetDevice(), &desc);
    if (!new_handle) {
        throw GPUException("Failed to shrink GPU buffer");
    }

    // GPU-copy old data (copy size must be 4-byte aligned for CopyBufferToBuffer)
    uint64 copy_size = AlignUp(size_, 4);
    copy_size = std::min(copy_size, std::min(capacity_, target_capacity));

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, 0, new_handle, 0, copy_size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    Release();
    handle_ = new_handle;
    capacity_ = target_capacity;

    LogInfo("GPUBuffer shrunk to ", target_capacity, " bytes");
}

void GPUBufferCore::Grow(uint64 min_capacity) {
    auto& core = GPUCore::GetInstance();
    assert(core.IsInitialized());

    // Growth strategy: 1.5x or min_capacity, whichever is larger
    uint64 new_capacity = std::max(min_capacity, capacity_ + (capacity_ >> 1));
    // Align up to 16 bytes
    new_capacity = AlignUp(new_capacity, 16);

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = static_cast<WGPUBufferUsage>(usage_) | WGPUBufferUsage_CopySrc;
    desc.size = new_capacity;

    WGPUBuffer new_handle = wgpuDeviceCreateBuffer(core.GetDevice(), &desc);
    if (!new_handle) {
        throw GPUException("Failed to grow GPU buffer");
    }

    // GPU-copy old data if any
    if (handle_ && size_ > 0) {
        // Copy size must be 4-byte aligned for CopyBufferToBuffer
        uint64 copy_size = AlignUp(size_, 4);
        copy_size = std::min(copy_size, capacity_);  // don't read past old buffer

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(core.GetDevice(), nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, handle_, 0, new_handle, 0, copy_size);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(core.GetQueue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    // Release old buffer
    if (handle_) {
        wgpuBufferRelease(handle_);
    }

    handle_ = new_handle;
    capacity_ = new_capacity;

    LogInfo("GPUBuffer grown to ", new_capacity, " bytes");
}

// -- Accessors ----------------------------------------------------------------

WGPUBuffer GPUBufferCore::GetHandle() const { return handle_; }
uint64 GPUBufferCore::GetSize() const { return size_; }
uint64 GPUBufferCore::GetCapacity() const { return capacity_; }
BufferUsage GPUBufferCore::GetUsage() const { return usage_; }
bool GPUBufferCore::IsEmpty() const { return size_ == 0; }
bool GPUBufferCore::IsValid() const { return handle_ != nullptr; }

// -- Internal -----------------------------------------------------------------

void GPUBufferCore::Release() {
    if (handle_) {
        wgpuBufferRelease(handle_);
        handle_ = nullptr;
    }
}

}  // namespace gpu
}  // namespace mps
