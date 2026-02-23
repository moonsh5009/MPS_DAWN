#pragma once

#include "ext_jgs2/jgs2_dynamics.h"
#include "ext_dynamics/spring_types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_jgs2 {

using namespace mps;

// GPU-side spring parameters (16 bytes, uniform-compatible)
struct alignas(16) JGS2SpringParams {
    float32 stiffness = 500.0f;
};

// JGS2 spring term: accumulates gradient + diagonal Hessian per edge via atomics.
class JGS2SpringTerm : public IJGS2Term {
public:
    JGS2SpringTerm(const std::vector<ext_dynamics::SpringEdge>& edges, float32 stiffness);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(const JGS2AssemblyContext& ctx) override;
    void Accumulate(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    std::vector<ext_dynamics::SpringEdge> edges_;
    float32 stiffness_;

    std::unique_ptr<gpu::GPUBuffer<ext_dynamics::SpringEdge>> edge_buffer_;
    std::unique_ptr<gpu::GPUBuffer<JGS2SpringParams>> spring_params_buffer_;
    gpu::GPUComputePipeline pipeline_;
    gpu::GPUBindGroup bg_springs_;
    uint32 wg_count_ = 0;

    static const std::string kName;
};

}  // namespace ext_jgs2
