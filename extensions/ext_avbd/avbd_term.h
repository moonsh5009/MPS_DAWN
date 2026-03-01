#pragma once

#include "core_util/types.h"
#include <string>
#include <vector>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;
struct WGPUCommandEncoderImpl;  typedef WGPUCommandEncoderImpl* WGPUCommandEncoder;

namespace ext_avbd {

// Per-color group data for IAVBDTerm initialization.
struct AVBDColorGroup {
    mps::uint32 color_offset;
    mps::uint32 color_vertex_count;
    WGPUBuffer params_buf;   // VBDColorParams uniform buffer
    mps::uint64 params_sz;   // sizeof(VBDColorParams) = 16
};

// Context passed to IAVBDTerm::Initialize.
// Provides shared buffers and color group info for bind group creation.
struct AVBDTermContext {
    mps::uint32 node_count;
    mps::uint32 edge_count;
    mps::uint32 face_count;
    WGPUBuffer q_buf;              mps::uint64 q_sz;
    WGPUBuffer gradient_buf;       mps::uint64 gradient_sz;
    WGPUBuffer hessian_buf;        mps::uint64 hessian_sz;
    WGPUBuffer vertex_order_buf;   mps::uint64 vertex_order_sz;
    const std::vector<AVBDColorGroup>& color_groups;
};

// Extension-local term interface for AVBD solver.
// Each term accumulates energy gradient and diagonal Hessian block
// into shared buffers. Per-color dispatch ensures no write conflicts.
// Augmented Lagrangian: DualUpdate/WarmstartDecay manage per-constraint dual variables.
class IAVBDTerm {
public:
    virtual ~IAVBDTerm() = default;

    [[nodiscard]] virtual const std::string& GetName() const = 0;

    // Create pipeline and per-color bind groups using shared context.
    virtual void Initialize(const AVBDTermContext& ctx) = 0;

    // Dispatch compute pass for one color group (vertex-centric).
    virtual void AccumulateColor(WGPUCommandEncoder encoder, mps::uint32 color_index) = 0;

    // Augmented Lagrangian: update dual variables after each VBD iteration.
    // Default no-op for terms without constraint dual variables.
    virtual void DualUpdate(WGPUCommandEncoder encoder) {}

    // Augmented Lagrangian: decay dual variables at frame start for warmstarting.
    // Default no-op for terms without constraint dual variables.
    virtual void WarmstartDecay(WGPUCommandEncoder encoder) {}

    virtual void Shutdown() = 0;
};

}  // namespace ext_avbd
