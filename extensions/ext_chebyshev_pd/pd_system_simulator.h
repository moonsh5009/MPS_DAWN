#pragma once

#include "core_simulate/simulator.h"
#include "core_gpu/gpu_handle.h"
#include <memory>
#include <string>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps { namespace system { class System; } }

namespace ext_chebyshev_pd {

class PDDynamics;

class ChebyshevPDSystemSimulator : public mps::simulate::ISimulator {
public:
    explicit ChebyshevPDSystemSimulator(mps::system::System& system);
    ~ChebyshevPDSystemSimulator() override;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize() override;
    void Update() override;
    void Shutdown() override;
    void OnDatabaseChanged() override;

private:
    mps::system::System& system_;
    std::unique_ptr<PDDynamics> dynamics_;

    mps::gpu::GPUComputePipeline update_velocity_pipeline_;
    mps::gpu::GPUComputePipeline update_position_pipeline_;

    mps::gpu::GPUBindGroup bg_vel_;
    mps::gpu::GPUBindGroup bg_pos_;

    mps::uint32 node_count_ = 0;
    bool initialized_ = false;
    bool rho_calibrated_ = false;
    mps::uint32 debug_frame_ = 0;

    WGPUBuffer local_pos_ = nullptr;
    WGPUBuffer local_vel_ = nullptr;
    WGPUBuffer local_mass_ = nullptr;
    WGPUBuffer global_pos_ = nullptr;
    WGPUBuffer global_vel_ = nullptr;
    mps::uint32 mesh_entity_ = 0;
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

}  // namespace ext_chebyshev_pd
