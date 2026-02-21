#include "ext_mesh/mesh_extension.h"
#include "ext_mesh/mesh_types.h"
#include "ext_mesh/mesh_post_processor.h"
#include "ext_mesh/mesh_renderer.h"
#include "core_simulate/sim_components.h"
#include "core_system/system.h"
#include "core_gpu/gpu_types.h"
#include "core_util/logger.h"
#include <memory>

using namespace mps;
using namespace mps::util;
using namespace mps::system;
using namespace mps::simulate;

namespace ext_mesh {

const std::string MeshExtension::kName = "ext_mesh";

MeshExtension::MeshExtension(System& system)
    : system_(system) {}

const std::string& MeshExtension::GetName() const {
    return kName;
}

void MeshExtension::Register(System& system) {
    // Register indexed arrays (topology with auto-offset relative to SimPosition)
    system.RegisterIndexedArray<MeshFace, SimPosition>(
        gpu::BufferUsage::None, "mesh_faces",
        [](MeshFace& f, uint32 off) { f.n0 += off; f.n1 += off; f.n2 += off; });

    system.RegisterIndexedArray<FixedVertex, SimPosition>(
        gpu::BufferUsage::None, "fixed_vertices",
        [](FixedVertex& fv, uint32 off) { fv.vertex_index += off; });

    // Create post-processor (normal computation after Newton solve)
    auto post_proc = std::make_unique<MeshPostProcessor>(system_);
    post_processor_ = post_proc.get();
    system.AddSimulator(std::move(post_proc));

    // Create renderer (references post-processor for normal/index buffers)
    system.AddRenderer(std::make_unique<MeshRenderer>(system_, *post_processor_));
}

}  // namespace ext_mesh
