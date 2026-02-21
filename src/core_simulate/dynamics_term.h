#pragma once

#include "core_util/types.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>

struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;
struct WGPUBufferImpl;          typedef WGPUBufferImpl*          WGPUBuffer;

namespace mps {
namespace simulate {

// Context passed to terms during Initialize for bind group caching
struct AssemblyContext {
    WGPUBuffer position_buffer;     // predicted positions (read)
    WGPUBuffer velocity_buffer;     // current velocities (read)
    WGPUBuffer mass_buffer;         // mass data (read)
    WGPUBuffer force_buffer;        // RHS force vector (atomic u32, read_write)
    WGPUBuffer diag_buffer;         // A diagonal 3x3 blocks (atomic u32 for springs, read_write)
    WGPUBuffer csr_values_buffer;   // A off-diagonal 3x3 blocks (read_write)
    WGPUBuffer params_buffer;       // system params uniform (read)
    WGPUBuffer dv_total_buffer;     // accumulated velocity delta (read)
    uint32 node_count;
    uint32 edge_count;
    uint32 workgroup_size;
    uint64 params_size;         // size of params buffer in bytes (48 for DynamicsParams)
};

// Builds CSR sparsity pattern from declared edges
class SparsityBuilder {
public:
    explicit SparsityBuilder(uint32 node_count);

    // Declare an edge (i,j) that the term will write to
    void AddEdge(uint32 node_a, uint32 node_b);

    // Finalize the CSR structure. Must be called after all edges are declared.
    void Build();

    // Accessors (valid after Build)
    [[nodiscard]] const std::vector<uint32>& GetRowPtr() const { return row_ptr_; }
    [[nodiscard]] const std::vector<uint32>& GetColIdx() const { return col_idx_; }
    [[nodiscard]] uint32 GetNNZ() const { return static_cast<uint32>(col_idx_.size()); }
    [[nodiscard]] uint32 GetNodeCount() const { return node_count_; }

    // Get CSR index for entry (row, col). Returns UINT32_MAX if not found.
    [[nodiscard]] uint32 GetCSRIndex(uint32 row, uint32 col) const;

private:
    uint32 node_count_;
    std::vector<std::set<uint32>> adjacency_;
    std::vector<uint32> row_ptr_;
    std::vector<uint32> col_idx_;
    std::map<std::pair<uint32, uint32>, uint32> csr_lookup_;
    bool built_ = false;
};

// Interface for dynamics contributions (force terms that modify A and b)
class IDynamicsTerm {
public:
    virtual ~IDynamicsTerm() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;

    // Phase 1: Declare which (i,j) entries of A this term will write to
    virtual void DeclareSparsity(SparsityBuilder& builder) {}

    // Phase 2: Initialize GPU resources (pipelines, buffers) and cache bind groups
    virtual void Initialize(const SparsityBuilder& sparsity, const AssemblyContext& ctx) = 0;

    // Phase 3: Dispatch cached bind groups to assemble contributions to A and b
    virtual void Assemble(WGPUCommandEncoder encoder) = 0;

    virtual void Shutdown() = 0;
};

}  // namespace simulate
}  // namespace mps
