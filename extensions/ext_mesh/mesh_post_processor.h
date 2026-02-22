#pragma once

#include "ext_mesh/mesh_types.h"
#include "ext_mesh/normal_computer.h"
#include "core_simulate/simulator.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>

struct WGPUBufferImpl;  typedef WGPUBufferImpl* WGPUBuffer;

namespace mps {
namespace system { class System; }
}

namespace ext_mesh {

// Post-processing simulator that computes vertex normals after Newton solve.
// Supports multiple mesh entities: uses DeviceDB's globally-indexed face buffer
// and iterates all mesh entities for face index buffer construction.
class MeshPostProcessor : public mps::simulate::ISimulator {
public:
    explicit MeshPostProcessor(mps::system::System& system);
    ~MeshPostProcessor() override;

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize() override;
    void Update() override;
    void Shutdown() override;
    void OnDatabaseChanged() override;

    [[nodiscard]] WGPUBuffer GetNormalBuffer() const;
    [[nodiscard]] WGPUBuffer GetIndexBuffer() const;
    [[nodiscard]] mps::uint32 GetFaceCount() const;

private:
    mps::system::System& system_;

    mps::uint32 node_count_ = 0;
    mps::uint32 total_face_count_ = 0;

    // Normal computation
    std::unique_ptr<NormalComputer> normals_;

    // GPU face index buffer (rendering: 3 uint32 per face, globally indexed)
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> face_index_buffer_;

    bool initialized_ = false;

    mps::uint32 ComputeTotalFaceCount() const;

    static const std::string kName;
};

}  // namespace ext_mesh
