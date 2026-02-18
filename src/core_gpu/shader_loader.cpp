#include "core_gpu/shader_loader.h"
#include "core_util/logger.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <unordered_set>
#include <filesystem>
#include <functional>

using namespace mps::util;

namespace mps {
namespace gpu {

// -- Source loading with #import support ---------------------------------------

std::string ShaderLoader::LoadSource(const std::string& path) {
    std::string source;
    const std::regex dir_pattern{"\\w+\\.wgsl"};
    const std::regex import_pattern{R"_(^[ \t]*#import[ \t]+"(.*)"\s*$)_", std::regex::optimize};
    std::unordered_set<std::string> processed;

    std::function<void(const std::string&)> read_source;
    read_source = [&](const std::string& file_path) {
        auto normalized = std::filesystem::path(file_path).lexically_normal().string();
        if (!processed.emplace(normalized).second) return;
        auto path_dir = std::regex_replace(normalized, dir_pattern, "");

        std::ifstream file(normalized);
        if (!file.is_open()) {
            LogError("Failed to open shader: ", normalized);
            return;
        }
        std::stringstream ss;
        ss << file.rdbuf();
        file.close();

        std::string line;
        std::smatch matches;
        while (std::getline(ss, line)) {
            if (std::regex_search(line, matches, import_pattern))
                read_source(path_dir + matches[1].str());
            else
                source.append(line + '\n');
        }
    };

    try {
        read_source("shaders/" + path);
    } catch (...) {
        try {
            read_source("../shaders/" + path);
        } catch (...) {
            LogError("Unable to open shader source: ", path);
        }
    }

    return source;
}

// -- Module creation ----------------------------------------------------------

GPUShader ShaderLoader::CreateModule(const std::string& path, const std::string& label) {
    auto code = LoadSource(path);
    if (code.empty()) {
        LogError("Shader source is empty: ", path);
    }

    std::string shader_label = label.empty() ? path : label;
    ShaderConfig config{code, shader_label};
    return GPUShader(config);
}

// -- Base path resolution -----------------------------------------------------

std::string ShaderLoader::ResolveBasePath() {
    if (std::filesystem::exists("shaders/")) {
        return "shaders/";
    }
    if (std::filesystem::exists("../shaders/")) {
        return "../shaders/";
    }
    LogWarning("Shader base path not found, defaulting to shaders/");
    return "shaders/";
}

}  // namespace gpu
}  // namespace mps
