#pragma once

#include <string>

namespace mps {
namespace gpu {

// Resolve a path relative to the assets/ directory.
// Searches CWD, ../CWD, exe directory (Windows) for assets/.
// Example: ResolveAssetPath("objs/cube.obj") → "<base>/assets/objs/cube.obj"
std::string ResolveAssetPath(const std::string& relative_path);

}  // namespace gpu
}  // namespace mps
