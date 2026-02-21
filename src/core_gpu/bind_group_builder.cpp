#include "core_gpu/bind_group_builder.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>
#include <utility>

namespace mps {
namespace gpu {

BindGroupBuilder::BindGroupBuilder(const std::string& label)
    : label_(label) {}

BindGroupBuilder& BindGroupBuilder::operator=(BindGroupBuilder&& other) noexcept {
    if (this != &other) {
        entries_ = std::move(other.entries_);
        label_ = std::move(other.label_);
    }
    return *this;
}

BindGroupBuilder&& BindGroupBuilder::AddBuffer(
    uint32 binding, WGPUBuffer buffer, uint64 size, uint64 offset) && {
    Entry entry;
    entry.binding = binding;
    entry.buffer = buffer;
    entry.buffer_size = size;
    entry.buffer_offset = offset;
    entries_.push_back(entry);
    return std::move(*this);
}

BindGroupBuilder&& BindGroupBuilder::AddTextureView(
    uint32 binding, WGPUTextureView view) && {
    Entry entry;
    entry.binding = binding;
    entry.texture_view = view;
    entries_.push_back(entry);
    return std::move(*this);
}

BindGroupBuilder&& BindGroupBuilder::AddSampler(
    uint32 binding, WGPUSampler sampler) && {
    Entry entry;
    entry.binding = binding;
    entry.sampler = sampler;
    entries_.push_back(entry);
    return std::move(*this);
}

GPUBindGroup BindGroupBuilder::Build(WGPUBindGroupLayout layout) && {
    auto& gpu = GPUCore::GetInstance();

    std::vector<WGPUBindGroupEntry> wgpu_entries;
    wgpu_entries.reserve(entries_.size());

    for (const auto& e : entries_) {
        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = e.binding;
        if (e.buffer) {
            entry.buffer = e.buffer;
            entry.offset = e.buffer_offset;
            entry.size = e.buffer_size;
        }
        if (e.texture_view) {
            entry.textureView = e.texture_view;
        }
        if (e.sampler) {
            entry.sampler = e.sampler;
        }
        wgpu_entries.push_back(entry);
    }

    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.layout = layout;
    desc.entryCount = wgpu_entries.size();
    desc.entries = wgpu_entries.data();

    return GPUBindGroup(wgpuDeviceCreateBindGroup(gpu.GetDevice(), &desc));
}

}  // namespace gpu
}  // namespace mps
