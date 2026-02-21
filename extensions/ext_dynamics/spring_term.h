#pragma once

#include "core_simulate/dynamics_term.h"
#include "ext_dynamics/spring_types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_dynamics {

// GPU-side parameters for spring stiffness (16 bytes, uniform-compatible)
struct alignas(16) SpringParams {
    mps::float32 stiffness = 500.0f;
};

class SpringTerm : public mps::simulate::IDynamicsTerm {
public:
    SpringTerm(const std::vector<SpringEdge>& edges, mps::float32 stiffness);

    [[nodiscard]] const std::string& GetName() const override;
    void DeclareSparsity(mps::simulate::SparsityBuilder& builder) override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::AssemblyContext& ctx) override;
    void Assemble(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    std::vector<SpringEdge> edges_;
    std::vector<EdgeCSRMapping> edge_csr_mappings_;
    mps::float32 stiffness_;
    mps::uint32 nnz_ = 0;

    std::unique_ptr<mps::gpu::GPUBuffer<SpringEdge>> edge_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<EdgeCSRMapping>> edge_csr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<SpringParams>> spring_params_buffer_;
    mps::gpu::GPUComputePipeline pipeline_;
    mps::gpu::GPUBindGroup bg_springs_;
    mps::uint32 wg_count_ = 0;

    static const std::string kName;
};

}  // namespace ext_dynamics
