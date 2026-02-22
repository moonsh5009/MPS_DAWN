#pragma once

#include "core_simulate/projective_term.h"
#include "ext_dynamics/spring_types.h"
#include "ext_newton/spring_term.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_pd_term {

class PDSpringTerm : public mps::simulate::IProjectiveTerm {
public:
    PDSpringTerm(const std::vector<ext_dynamics::SpringEdge>& edges, mps::float32 stiffness);

    [[nodiscard]] const std::string& GetName() const override;
    void DeclareSparsity(mps::simulate::SparsityBuilder& builder) override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::PDAssemblyContext& ctx) override;
    void AssembleLHS(WGPUCommandEncoder encoder) override;
    void ProjectRHS(WGPUCommandEncoder encoder) override;

    // ADMM methods
    void InitializeADMM(const mps::simulate::PDAssemblyContext& ctx) override;
    void ProjectLocal(WGPUCommandEncoder encoder) override;
    void AssembleADMMRHS(WGPUCommandEncoder encoder) override;
    void UpdateDual(WGPUCommandEncoder encoder) override;
    void ResetDual(WGPUCommandEncoder encoder) override;

    void Shutdown() override;

private:
    std::vector<ext_dynamics::SpringEdge> edges_;
    std::vector<ext_dynamics::EdgeCSRMapping> edge_csr_mappings_;
    mps::float32 stiffness_;
    mps::uint32 nnz_ = 0;

    // GPU buffers
    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::SpringEdge>> edge_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::EdgeCSRMapping>> edge_csr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_newton::SpringParams>> spring_params_buffer_;

    // Chebyshev pipelines
    mps::gpu::GPUComputePipeline lhs_pipeline_;
    mps::gpu::GPUComputePipeline project_rhs_pipeline_;

    // Chebyshev bind groups
    mps::gpu::GPUBindGroup bg_lhs_;
    mps::gpu::GPUBindGroup bg_project_rhs_;

    // ADMM buffers (z, u per edge)
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> z_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> u_buffer_;

    // ADMM pipelines
    mps::gpu::GPUComputePipeline admm_project_pipeline_;
    mps::gpu::GPUComputePipeline admm_rhs_pipeline_;
    mps::gpu::GPUComputePipeline admm_dual_pipeline_;
    mps::gpu::GPUComputePipeline admm_reset_pipeline_;

    // ADMM bind groups
    mps::gpu::GPUBindGroup bg_admm_project_;
    mps::gpu::GPUBindGroup bg_admm_rhs_;
    mps::gpu::GPUBindGroup bg_admm_dual_;
    mps::gpu::GPUBindGroup bg_admm_reset_;

    mps::uint32 wg_count_ = 0;

    static const std::string kName;
};

}  // namespace ext_pd_term
