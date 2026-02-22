#pragma once

#include "core_simulate/simulator.h"
#include "core_database/entity.h"
#include "core_gpu/gpu_handle.h"
#include <memory>
#include <string>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace system { class System; }
namespace simulate { class NewtonDynamics; }
}

namespace ext_newton {

// Generic Newton-Raphson dynamics simulator.
// Discovers constraint terms from entity references in NewtonSystemConfig,
// then runs the Newton solver and integrates velocity/position.
class NewtonSystemSimulator : public mps::simulate::ISimulator {
public:
    explicit NewtonSystemSimulator(mps::system::System& system);
    ~NewtonSystemSimulator() override;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize() override;
    void Update() override;
    void Shutdown() override;
    void OnDatabaseChanged() override;

private:
    mps::system::System& system_;

    // Dynamics solver (Newton-Raphson + CG + pluggable terms)
    std::unique_ptr<mps::simulate::NewtonDynamics> dynamics_;

    // Velocity/position update pipelines
    mps::gpu::GPUComputePipeline update_velocity_pipeline_;
    mps::gpu::GPUComputePipeline update_position_pipeline_;

    // Cached bind groups for velocity/position update
    mps::gpu::GPUBindGroup bg_vel_;
    mps::gpu::GPUBindGroup bg_pos_;

    // Cached counts
    mps::uint32 node_count_ = 0;

    bool initialized_ = false;
    mps::uint32 debug_frame_ = 0;

    // Scoped mode (mesh_entity != kInvalidEntity): local buffer copy
    WGPUBuffer local_pos_ = nullptr;
    WGPUBuffer local_vel_ = nullptr;
    WGPUBuffer local_mass_ = nullptr;
    WGPUBuffer global_pos_ = nullptr;
    WGPUBuffer global_vel_ = nullptr;
    mps::uint32 mesh_entity_ = mps::database::kInvalidEntity;
    mps::uint32 node_offset_ = 0;
    bool scoped_ = false;

    struct TopologySignature {
        mps::uint32 node_count = 0;
        mps::uint32 total_edges = 0;
        mps::uint32 total_faces = 0;
        mps::uint32 constraint_count = 0;
    };
    TopologySignature topology_sig_;
    TopologySignature ComputeTopologySignature() const;

    static const std::string kName;
    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_newton
