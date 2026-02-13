#include "core_platform/window_wasm.h"
#include "core_util/logger.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>
#endif

using namespace mps::util;

namespace mps {
namespace platform {

WindowWasm::WindowWasm() = default;

WindowWasm::~WindowWasm() {
    Shutdown();
}

bool WindowWasm::Initialize(const WindowConfig& config) {
#ifdef __EMSCRIPTEN__
    config_ = config;

    // Get canvas size
    double width, height;
    emscripten_get_element_css_size("#canvas", &width, &height);

    current_width_ = static_cast<uint32>(width);
    current_height_ = static_cast<uint32>(height);

    // Set canvas size
    emscripten_set_canvas_element_size("#canvas", current_width_, current_height_);

    // Set document title
    emscripten_run_script(("document.title = '" + config.title + "';").c_str());

    LogInfo("WASM Window initialized: ", current_width_, "x", current_height_);
    return true;
#else
    LogError("WindowWasm can only be used in Emscripten builds");
    return false;
#endif
}

void WindowWasm::Shutdown() {
    // Nothing to clean up for WASM
    LogInfo("WASM Window shutdown");
}

void WindowWasm::PollEvents() {
#ifdef __EMSCRIPTEN__
    // Update canvas size in case it changed
    double width, height;
    emscripten_get_element_css_size("#canvas", &width, &height);
    current_width_ = static_cast<uint32>(width);
    current_height_ = static_cast<uint32>(height);
#endif
}

bool WindowWasm::ShouldClose() const {
    return should_close_;
}

bool WindowWasm::IsMinimized() const {
    // WASM windows can't be minimized in the traditional sense
    return false;
}

bool WindowWasm::IsFocused() const {
    return is_focused_;
}

uint32 WindowWasm::GetWidth() const {
    return current_width_;
}

uint32 WindowWasm::GetHeight() const {
    return current_height_;
}

float32 WindowWasm::GetAspectRatio() const {
    return current_height_ > 0 ?
        static_cast<float32>(current_width_) / static_cast<float32>(current_height_) : 0.0f;
}

const std::string& WindowWasm::GetTitle() const {
    return config_.title;
}

void WindowWasm::SetTitle(const std::string& title) {
#ifdef __EMSCRIPTEN__
    config_.title = title;
    emscripten_run_script(("document.title = '" + title + "';").c_str());
#endif
}

void WindowWasm::SetSize(uint32 width, uint32 height) {
#ifdef __EMSCRIPTEN__
    config_.width = width;
    config_.height = height;
    current_width_ = width;
    current_height_ = height;
    emscripten_set_canvas_element_size("#canvas", width, height);
#endif
}

void WindowWasm::SetFullscreen(bool fullscreen) {
#ifdef __EMSCRIPTEN__
    config_.fullscreen = fullscreen;
    if (fullscreen) {
        EmscriptenFullscreenStrategy strategy = {};
        strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
        strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_HIDEF;
        strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
        emscripten_request_fullscreen_strategy("#canvas", 1, &strategy);
    } else {
        emscripten_exit_fullscreen();
    }
#endif
}

WGPUSurface WindowWasm::CreateSurface(WGPUInstance instance) {
#ifdef __EMSCRIPTEN__
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc = {};
    canvasDesc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    canvasDesc.selector = {"#canvas", 7};

    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&canvasDesc);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfaceDesc);
    if (!surface) {
        LogError("Failed to create WebGPU surface for WASM");
        return nullptr;
    }

    LogInfo("WebGPU surface created successfully (WASM)");
    return surface;
#else
    LogError("WindowWasm::CreateSurface can only be used in Emscripten builds");
    return nullptr;
#endif
}

}  // namespace platform
}  // namespace mps
