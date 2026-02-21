#pragma once

#include "core_gpu/gpu_types.h"
#include "core_gpu/gpu_handle.h"
#include <vector>
#include <string>

namespace mps {
namespace gpu {

class BindGroupLayoutBuilder {
public:
    BindGroupLayoutBuilder() = default;
    explicit BindGroupLayoutBuilder(const std::string& label);

    BindGroupLayoutBuilder(const BindGroupLayoutBuilder&) = delete;
    BindGroupLayoutBuilder& operator=(const BindGroupLayoutBuilder&) = delete;
    BindGroupLayoutBuilder(BindGroupLayoutBuilder&&) noexcept = default;
    BindGroupLayoutBuilder& operator=(BindGroupLayoutBuilder&&) noexcept = default;

    BindGroupLayoutBuilder&& AddBinding(uint32 binding, ShaderStage visibility, BindingType type) &&;
    BindGroupLayoutBuilder&& AddUniformBinding(uint32 binding, ShaderStage visibility) &&;
    BindGroupLayoutBuilder&& AddStorageBinding(uint32 binding, ShaderStage visibility) &&;
    BindGroupLayoutBuilder&& AddReadOnlyStorageBinding(uint32 binding, ShaderStage visibility) &&;
    BindGroupLayoutBuilder&& AddTextureBinding(uint32 binding, ShaderStage visibility) &&;
    BindGroupLayoutBuilder&& AddSamplerBinding(uint32 binding, ShaderStage visibility) &&;

    GPUBindGroupLayout Build() &&;

private:
    struct BindingEntry {
        uint32 binding;
        ShaderStage visibility;
        BindingType type;
    };
    std::vector<BindingEntry> entries_;
    std::string label_;
};

}  // namespace gpu
}  // namespace mps
