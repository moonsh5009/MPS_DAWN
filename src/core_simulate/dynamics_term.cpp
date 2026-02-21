#include "core_simulate/dynamics_term.h"
#include <algorithm>

namespace mps {
namespace simulate {

SparsityBuilder::SparsityBuilder(uint32 node_count)
    : node_count_(node_count), adjacency_(node_count) {}

void SparsityBuilder::AddEdge(uint32 node_a, uint32 node_b) {
    adjacency_[node_a].insert(node_b);
    adjacency_[node_b].insert(node_a);
}

void SparsityBuilder::Build() {
    row_ptr_.resize(node_count_ + 1, 0);
    col_idx_.clear();
    csr_lookup_.clear();

    for (uint32 i = 0; i < node_count_; ++i) {
        row_ptr_[i] = static_cast<uint32>(col_idx_.size());
        for (uint32 j : adjacency_[i]) {
            csr_lookup_[{i, j}] = static_cast<uint32>(col_idx_.size());
            col_idx_.push_back(j);
        }
    }
    row_ptr_[node_count_] = static_cast<uint32>(col_idx_.size());
    built_ = true;
}

uint32 SparsityBuilder::GetCSRIndex(uint32 row, uint32 col) const {
    auto it = csr_lookup_.find({row, col});
    if (it != csr_lookup_.end()) return it->second;
    return UINT32_MAX;
}

}  // namespace simulate
}  // namespace mps
