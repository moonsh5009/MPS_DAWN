#pragma once

#include "core_simulate/dynamics_term.h"
#include "ext_dynamics/area_types.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace ext_newton {

// GPU-side parameters for area constraint stiffness (16 bytes, uniform-compatible)
struct alignas(16) AreaParams {
    mps::float32 stiffness = 1.0f;       // area preservation (bulk modulus k)
    mps::float32 shear_stiffness = 0.0f;  // trace energy / shear modulus (μ)
};

// Area preservation constraint term.
// Penalizes deviation from rest triangle area using Gauss-Newton approximation.
// Force + full Hessian (diagonal + off-diagonal CSR blocks).
class AreaTerm : public mps::simulate::IDynamicsTerm {
public:
    AreaTerm(const std::vector<ext_dynamics::AreaTriangle>& triangles, mps::float32 stiffness);

    [[nodiscard]] const std::string& GetName() const override;
    void DeclareSparsity(mps::simulate::SparsityBuilder& builder) override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::AssemblyContext& ctx) override;
    void Assemble(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    std::vector<ext_dynamics::AreaTriangle> triangles_;
    std::vector<ext_dynamics::FaceCSRMapping> face_csr_mappings_;
    mps::float32 stiffness_;
    mps::uint32 nnz_ = 0;

    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::AreaTriangle>> triangle_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_dynamics::FaceCSRMapping>> face_csr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<AreaParams>> area_params_buffer_;
    mps::gpu::GPUComputePipeline pipeline_;
    mps::gpu::GPUBindGroup bg_area_;
    mps::uint32 wg_count_ = 0;

    static const std::string kName;
};

}  // namespace ext_newton
