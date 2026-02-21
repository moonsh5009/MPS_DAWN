#pragma once

#include "core_simulate/dynamics_term.h"
#include "core_gpu/gpu_handle.h"
#include <string>

namespace ext_dynamics {

// Adds gravitational force to the RHS force buffer: force[i] += mass_i * gravity.
// No sparsity declaration (force-only term, no Hessian contribution).
class GravityTerm : public mps::simulate::IDynamicsTerm {
public:
    GravityTerm() = default;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::AssemblyContext& ctx) override;
    void Assemble(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    mps::gpu::GPUComputePipeline pipeline_;
    mps::gpu::GPUBindGroup bg_gravity_;
    mps::uint32 wg_count_ = 0;
    static const std::string kName;
};

}  // namespace ext_dynamics
