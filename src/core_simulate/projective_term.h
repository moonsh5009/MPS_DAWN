#pragma once

#include "core_simulate/dynamics_term.h"
#include "core_util/types.h"
#include <string>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace mps {
namespace simulate {

// Context passed to PD terms during Initialize for bind group caching
struct PDAssemblyContext {
    WGPUBuffer physics_buffer;     // global physics params uniform (binding 0)
    WGPUBuffer q_buffer;           // current iterate q (read)
    WGPUBuffer s_buffer;           // predicted positions s (read)
    WGPUBuffer mass_buffer;        // mass data (read)
    WGPUBuffer rhs_buffer;         // RHS accumulation (atomic u32, read_write)
    WGPUBuffer diag_buffer;        // LHS diagonal 3x3 blocks (atomic u32, read_write)
    WGPUBuffer csr_values_buffer;  // LHS off-diagonal CSR 3x3 blocks (read_write)
    WGPUBuffer params_buffer;      // solver params uniform (binding 1)
    uint32 node_count;
    uint32 edge_count;
    uint32 workgroup_size;
    uint64 physics_size;       // size of physics buffer in bytes
    uint64 params_size;        // size of solver params buffer in bytes
};

// Interface for Projective Dynamics constraint terms.
// Each term contributes to the LHS (constant S^T*S) and per-iteration
// local projection + RHS assembly.
class IProjectiveTerm {
public:
    virtual ~IProjectiveTerm() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;

    // Phase 1: Declare which (i,j) entries of A this term will write to
    virtual void DeclareSparsity(SparsityBuilder& builder) {}

    // Phase 2: Initialize GPU resources and cache bind groups
    virtual void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) = 0;

    // Phase 3a: Assemble constant LHS contribution (w * S^T * S)
    // Called once when dt changes or at init
    virtual void AssembleLHS(WGPUCommandEncoder encoder) = 0;

    // Phase 3b: Fused local projection + RHS assembly in a single dispatch.
    // Computes p from current q and immediately scatters w * S^T * p to RHS.
    virtual void ProjectRHS(WGPUCommandEncoder encoder) = 0;

    virtual void Shutdown() = 0;
};

}  // namespace simulate
}  // namespace mps
