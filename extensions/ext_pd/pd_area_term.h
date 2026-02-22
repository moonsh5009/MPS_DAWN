#pragma once

#include "core_simulate/projective_term.h"
#include "ext_dynamics/area_types.h"
#include "ext_newton/area_term.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_pd {

class PDAreaTerm : public mps::simulate::IProjectiveTerm {
public:
    PDAreaTerm(const std::vector<ext_dynamics::AreaTriangle>& triangles, mps::float32 stiffness);

    [[nodiscard]] const std::string& GetName() const override;
    void DeclareSparsity(mps::simulate::SparsityBuilder& builder) override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::PDAssemblyContext& ctx) override;
    void AssembleLHS(WGPUCommandEncoder encoder) override;
    void ProjectRHS(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    std::vector<ext_dynamics::AreaTriangle> triangles_;
    std::vector<ext_dynamics::FaceCSRMapping> face_csr_mappings_;
    mps::float32 stiffness_;
    mps::uint32 nnz_ = 0;

    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::AreaTriangle>> triangle_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::FaceCSRMapping>> face_csr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_newton::AreaParams>> area_params_buffer_;

    mps::gpu::GPUComputePipeline lhs_pipeline_;
    mps::gpu::GPUComputePipeline project_rhs_pipeline_;

    mps::gpu::GPUBindGroup bg_lhs_;
    mps::gpu::GPUBindGroup bg_project_rhs_;
    mps::uint32 wg_count_ = 0;

    static const std::string kName;
};

}  // namespace ext_pd
