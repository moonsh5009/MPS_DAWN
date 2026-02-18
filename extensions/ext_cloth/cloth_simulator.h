#pragma once

#include "ext_cloth/cloth_types.h"
#include "ext_cloth/cloth_mesh.h"
#include "core_simulate/simulator.h"
#include "core_gpu/gpu_buffer.h"
#include <memory>
#include <string>
#include <vector>

struct WGPUComputePipelineImpl;  typedef WGPUComputePipelineImpl* WGPUComputePipeline;
struct WGPUBindGroupImpl;        typedef WGPUBindGroupImpl*       WGPUBindGroup;
struct WGPUBindGroupLayoutImpl;  typedef WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
struct WGPUPipelineLayoutImpl;   typedef WGPUPipelineLayoutImpl*  WGPUPipelineLayout;
struct WGPUBufferImpl;           typedef WGPUBufferImpl*          WGPUBuffer;

namespace mps { namespace system { class System; } }

namespace ext_cloth {

class ClothSimulator : public mps::simulate::ISimulator {
public:
    explicit ClothSimulator(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(mps::database::Database& db) override;
    void Update(mps::database::Database& db, mps::float32 dt) override;
    void Shutdown() override;

    WGPUBuffer GetNormalBuffer() const;
    WGPUBuffer GetIndexBuffer() const;
    mps::uint32 GetFaceCount() const;
    mps::uint32 GetNodeCount() const;

private:
    void CreateMesh(mps::database::Database& db);
    void BuildCSRSparsity();
    void CreateGPUBuffers();
    void CreateComputePipelines();
    void ReadbackPositionsVelocities(mps::database::Database& db);

    mps::system::System& system_;

    // Mesh data (CPU)
    ClothMeshData mesh_data_;
    std::vector<mps::uint32> csr_row_ptr_;
    std::vector<mps::uint32> csr_col_idx_;
    std::vector<EdgeCSRMapping> edge_csr_mappings_;
    mps::uint32 nnz_ = 0;

    // GPU topology buffers
    std::unique_ptr<mps::gpu::GPUBuffer<ClothEdge>> edge_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<ClothFace>> face_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<EdgeCSRMapping>> edge_csr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> face_index_buffer_;

    // GPU solver buffers
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> force_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> normal_buffer_;         // f32, renderer output
    std::unique_ptr<mps::gpu::GPUBuffer<mps::int32>> normal_atomic_buffer_;    // i32, atomic scatter

    // CSR Hessian
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> csr_row_ptr_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::uint32>> csr_col_idx_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> csr_values_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> diag_values_buffer_;

    // Newton solver
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> x_old_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> dv_total_buffer_;

    // CG solver
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_x_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_r_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_p_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_ap_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_partial_buffer_;
    std::unique_ptr<mps::gpu::GPUBuffer<mps::float32>> cg_scalar_buffer_;

    // Params uniform
    std::unique_ptr<mps::gpu::GPUBuffer<ClothSimParams>> params_buffer_;

    // CG constant uniforms (created once, reused every frame)
    struct DotConfig { mps::uint32 target; mps::uint32 count; mps::uint32 pad0; mps::uint32 pad1; };
    struct ScalarMode { mps::uint32 mode; mps::uint32 pad0; mps::uint32 pad1; mps::uint32 pad2; };
    std::unique_ptr<mps::gpu::GPUBuffer<DotConfig>> dc_rr_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<DotConfig>> dc_pap_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<DotConfig>> dc_rr_new_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<ScalarMode>> mode_alpha_buf_;
    std::unique_ptr<mps::gpu::GPUBuffer<ScalarMode>> mode_beta_buf_;

    // Compute pipelines
    WGPUComputePipeline newton_init_pipeline_ = nullptr;
    WGPUComputePipeline newton_predict_pos_pipeline_ = nullptr;
    WGPUComputePipeline newton_accumulate_dv_pipeline_ = nullptr;
    WGPUComputePipeline clear_forces_pipeline_ = nullptr;
    WGPUComputePipeline accumulate_gravity_pipeline_ = nullptr;
    WGPUComputePipeline accumulate_springs_pipeline_ = nullptr;
    WGPUComputePipeline assemble_rhs_pipeline_ = nullptr;
    WGPUComputePipeline cg_init_pipeline_ = nullptr;
    WGPUComputePipeline cg_spmv_pipeline_ = nullptr;
    WGPUComputePipeline cg_dot_pipeline_ = nullptr;
    WGPUComputePipeline cg_dot_final_pipeline_ = nullptr;
    WGPUComputePipeline cg_compute_scalars_pipeline_ = nullptr;
    WGPUComputePipeline cg_update_xr_pipeline_ = nullptr;
    WGPUComputePipeline cg_update_p_pipeline_ = nullptr;
    WGPUComputePipeline update_velocity_pipeline_ = nullptr;
    WGPUComputePipeline update_position_pipeline_ = nullptr;
    WGPUComputePipeline clear_normals_pipeline_ = nullptr;
    WGPUComputePipeline scatter_normals_pipeline_ = nullptr;
    WGPUComputePipeline normalize_normals_pipeline_ = nullptr;

    mps::uint32 workgroup_count_ = 0;
    mps::uint32 dot_partial_count_ = 0;
    bool initialized_ = false;

    static const std::string kName;
    static constexpr mps::uint32 kWorkgroupSize = 64;
};

}  // namespace ext_cloth
