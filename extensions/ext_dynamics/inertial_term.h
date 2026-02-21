#pragma once

#include "core_simulate/dynamics_term.h"
#include "core_gpu/gpu_handle.h"
#include <string>

namespace ext_dynamics {

// Adds mass (inertia) to the system matrix diagonal: A_ii += M_i * I3x3.
// No sparsity declaration (diagonal only, no off-diagonal entries).
class InertialTerm : public mps::simulate::IDynamicsTerm {
public:
    InertialTerm() = default;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(const mps::simulate::SparsityBuilder& sparsity,
                    const mps::simulate::AssemblyContext& ctx) override;
    void Assemble(WGPUCommandEncoder encoder) override;
    void Shutdown() override;

private:
    mps::gpu::GPUComputePipeline pipeline_;
    mps::gpu::GPUBindGroup bg_inertia_;
    mps::uint32 wg_count_ = 0;
    static const std::string kName;
};

}  // namespace ext_dynamics
