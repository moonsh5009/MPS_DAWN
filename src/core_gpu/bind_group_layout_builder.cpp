#include "core_gpu/bind_group_layout_builder.h"
#include "core_gpu/gpu_core.h"
#include <webgpu/webgpu.h>
#include <utility>

namespace mps {
namespace gpu {

BindGroupLayoutBuilder::BindGroupLayoutBuilder(const std::string& label)
    : label_(label) {}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddBinding(
    uint32 binding, ShaderStage visibility, BindingType type) && {
    entries_.push_back({binding, visibility, type});
    return std::move(*this);
}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddUniformBinding(
    uint32 binding, ShaderStage visibility) && {
    return std::move(*this).AddBinding(binding, visibility, BindingType::Uniform);
}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddStorageBinding(
    uint32 binding, ShaderStage visibility) && {
    return std::move(*this).AddBinding(binding, visibility, BindingType::Storage);
}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddReadOnlyStorageBinding(
    uint32 binding, ShaderStage visibility) && {
    return std::move(*this).AddBinding(binding, visibility, BindingType::ReadOnlyStorage);
}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddTextureBinding(
    uint32 binding, ShaderStage visibility) && {
    return std::move(*this).AddBinding(binding, visibility, BindingType::Texture2D);
}

BindGroupLayoutBuilder&& BindGroupLayoutBuilder::AddSamplerBinding(
    uint32 binding, ShaderStage visibility) && {
    return std::move(*this).AddBinding(binding, visibility, BindingType::FilteringSampler);
}

WGPUBindGroupLayout BindGroupLayoutBuilder::Build() && {
    auto& gpu = GPUCore::GetInstance();

    std::vector<WGPUBindGroupLayoutEntry> wgpu_entries;
    wgpu_entries.reserve(entries_.size());

    for (const auto& e : entries_) {
        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = e.binding;
        entry.visibility = static_cast<WGPUShaderStage>(e.visibility);

        switch (e.type) {
            case BindingType::Uniform:
                entry.buffer.type = WGPUBufferBindingType_Uniform;
                break;
            case BindingType::Storage:
                entry.buffer.type = WGPUBufferBindingType_Storage;
                break;
            case BindingType::ReadOnlyStorage:
                entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                break;
            case BindingType::Sampler:
            case BindingType::FilteringSampler:
                entry.sampler.type = WGPUSamplerBindingType_Filtering;
                break;
            case BindingType::Texture2D:
                entry.texture.sampleType = WGPUTextureSampleType_Float;
                entry.texture.viewDimension = WGPUTextureViewDimension_2D;
                break;
            case BindingType::StorageTexture2D:
                entry.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
                entry.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                break;
        }
        wgpu_entries.push_back(entry);
    }

    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.label = {label_.data(), label_.size()};
    desc.entryCount = wgpu_entries.size();
    desc.entries = wgpu_entries.data();

    return wgpuDeviceCreateBindGroupLayout(gpu.GetDevice(), &desc);
}

}  // namespace gpu
}  // namespace mps
