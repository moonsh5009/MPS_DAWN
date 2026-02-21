#pragma once

#include "core_simulate/simulator.h"
#include "core_gpu/gpu_handle.h"
#include <memory>
#include <string>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace system { class System; }
namespace simulate { class NewtonDynamics; }
}
namespace ext_dynamics { class InertialTerm; }

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
    void Update(mps::float32 dt) override;
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

    // Newton config (from entity)
    mps::uint32 newton_iterations_ = 1;
    mps::uint32 cg_max_iterations_ = 30;

    bool initialized_ = false;

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
