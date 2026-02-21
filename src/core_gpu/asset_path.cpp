#include "core_gpu/asset_path.h"
#include "core_util/logger.h"
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace mps::util;

namespace mps {
namespace gpu {

static std::string ResolveAssetsBasePath() {
    // Try relative to CWD
    if (std::filesystem::exists("assets/"))
        return "assets/";
    if (std::filesystem::exists("../assets/"))
        return "../assets/";

    // Try relative to executable directory
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(NULL, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        auto exe_dir = std::filesystem::path(buf).parent_path();
        auto assets_dir = exe_dir / "assets";
        if (std::filesystem::exists(assets_dir))
            return (assets_dir / "").string();  // trailing separator
    }
#endif

    LogWarning("Assets base path not found, defaulting to assets/");
    return "assets/";
}

std::string ResolveAssetPath(const std::string& relative_path) {
    static std::string base = ResolveAssetsBasePath();
    return base + relative_path;
}

}  // namespace gpu
}  // namespace mps
