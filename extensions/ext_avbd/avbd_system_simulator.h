#pragma once

#include "core_simulate/simulator.h"
#include "core_gpu/gpu_handle.h"
#include "core_gpu/gpu_buffer.h"
#include "ext_avbd/graph_coloring.h"
#include "ext_avbd/vbd_dynamics.h"
#include "ext_mesh/mesh_types.h"
#include <memory>
#include <string>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps { namespace system { class System; } }

namespace ext_avbd {

// AVBD (Vertex Block Descent) system simulator.
// Discovers mesh topology, runs GPU graph coloring, builds color groups,
// creates VBDDynamics solver with term discovery, then per-frame solves
// with velocity/position integration.
class AVBDSystemSimulator : public mps::simulate::ISimulator {
public:
    explicit AVBDSystemSimulator(mps::system::System& system);
    ~AVBDSystemSimulator() override;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize() override;
    void Update() override;
    void Shutdown() override;
    void OnDatabaseChanged() override;

private:
    mps::system::System& system_;

    // Graph coloring
    GraphColoring coloring_;
    std::unique_ptr<mps::gpu::GPUBuffer<ext_mesh::MeshEdge>> local_edge_buf_;

    // VBD solver
    std::unique_ptr<VBDDynamics> dynamics_;

    // Scoped mode buffers (isolated per-mesh simulation)
    bool scoped_ = false;
    mps::uint32 node_offset_ = 0;
    WGPUBuffer local_pos_ = nullptr;
    WGPUBuffer local_vel_ = nullptr;
    WGPUBuffer local_mass_ = nullptr;
    WGPUBuffer global_pos_ = nullptr;
    WGPUBuffer global_vel_ = nullptr;

    // Integration pipelines (velocity + position update, outside VBD loop)
    mps::gpu::GPUComputePipeline update_velocity_pipeline_;
    mps::gpu::GPUComputePipeline update_position_pipeline_;
    mps::gpu::GPUBindGroup bg_update_velocity_;
    mps::gpu::GPUBindGroup bg_update_position_;

    mps::uint32 node_count_ = 0;
    mps::uint32 edge_count_ = 0;
    mps::uint32 face_count_ = 0;
    bool initialized_ = false;

    struct TopologySignature {
        mps::uint32 node_count = 0;
        mps::uint32 edge_count = 0;
        mps::uint32 face_count = 0;
    };
    TopologySignature topology_sig_;
    TopologySignature ComputeTopologySignature() const;

    static const std::string kName;
    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_avbd
