#include "core_platform/window_native.h"
#include "core_platform/input.h"
#include "core_util/logger.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <webgpu/webgpu.h>

using namespace mps::util;

namespace mps {
namespace platform {

WindowNative::WindowNative() = default;

WindowNative::~WindowNative() {
    Shutdown();
}

bool WindowNative::Initialize(const WindowConfig& config) {
    config_ = config;

    // Initialize GLFW
    if (!glfwInit()) {
        LogError("Failed to initialize GLFW");
        return false;
    }

    // Configure GLFW for no client API (we're using WebGPU)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);

    // Create window
    window_ = glfwCreateWindow(
        static_cast<int>(config.width),
        static_cast<int>(config.height),
        config.title.c_str(),
        config.fullscreen ? glfwGetPrimaryMonitor() : nullptr,
        nullptr
    );

    if (!window_) {
        LogError("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    // Set user pointer for callbacks
    glfwSetWindowUserPointer(window_, this);

    // Set callbacks
    glfwSetFramebufferSizeCallback(window_, FramebufferSizeCallback);
    glfwSetWindowFocusCallback(window_, WindowFocusCallback);
    glfwSetKeyCallback(window_, KeyCallback);
    glfwSetMouseButtonCallback(window_, MouseButtonCallback);
    glfwSetCursorPosCallback(window_, CursorPosCallback);
    glfwSetScrollCallback(window_, ScrollCallback);

    LogInfo("Window created: ", config.width, "x", config.height);
    return true;
}

void WindowNative::Shutdown() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void WindowNative::PollEvents() {
    glfwPollEvents();
}

bool WindowNative::ShouldClose() const {
    return window_ ? glfwWindowShouldClose(window_) : true;
}

bool WindowNative::IsMinimized() const {
    if (!window_) {
        return false;
    }
    return glfwGetWindowAttrib(window_, GLFW_ICONIFIED) != 0;
}

bool WindowNative::IsFocused() const {
    return is_focused_;
}

uint32 WindowNative::GetWidth() const {
    if (!window_) {
        return 0;
    }
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    return static_cast<uint32>(width);
}

uint32 WindowNative::GetHeight() const {
    if (!window_) {
        return 0;
    }
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    return static_cast<uint32>(height);
}

float32 WindowNative::GetAspectRatio() const {
    uint32 width = GetWidth();
    uint32 height = GetHeight();
    return height > 0 ? static_cast<float32>(width) / static_cast<float32>(height) : 0.0f;
}

const std::string& WindowNative::GetTitle() const {
    return config_.title;
}

void WindowNative::SetTitle(const std::string& title) {
    config_.title = title;
    if (window_) {
        glfwSetWindowTitle(window_, title.c_str());
    }
}

void WindowNative::SetSize(uint32 width, uint32 height) {
    config_.width = width;
    config_.height = height;
    if (window_) {
        glfwSetWindowSize(window_, static_cast<int>(width), static_cast<int>(height));
    }
}

void WindowNative::SetFullscreen(bool fullscreen) {
    if (!window_ || config_.fullscreen == fullscreen) {
        return;
    }

    config_.fullscreen = fullscreen;

    if (fullscreen) {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    } else {
        glfwSetWindowMonitor(window_, nullptr, 100, 100, config_.width, config_.height, 0);
    }
}

WGPUSurface WindowNative::CreateSurface(WGPUInstance instance) {
    if (!window_) {
        LogError("Cannot create surface: window not initialized");
        return nullptr;
    }

#ifdef _WIN32
    HWND hwnd = glfwGetWin32Window(window_);
    WGPUSurfaceSourceWindowsHWND surfaceSource = {};
    surfaceSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
    surfaceSource.hinstance = GetModuleHandle(nullptr);
    surfaceSource.hwnd = hwnd;

    WGPUSurfaceDescriptor surfaceDesc = {};
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&surfaceSource);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfaceDesc);
    if (!surface) {
        LogError("Failed to create WebGPU surface");
        return nullptr;
    }

    LogInfo("WebGPU surface created successfully");
    return surface;
#else
    // TODO: Implement for other platforms (Linux, macOS)
    LogError("Surface creation not implemented for this platform");
    return nullptr;
#endif
}

// Callbacks
void WindowNative::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* win = static_cast<WindowNative*>(glfwGetWindowUserPointer(window));
    if (win) {
        LogDebug("Framebuffer resized: ", width, "x", height);
    }
}

void WindowNative::WindowFocusCallback(GLFWwindow* window, int focused) {
    auto* win = static_cast<WindowNative*>(glfwGetWindowUserPointer(window));
    if (win) {
        win->is_focused_ = (focused != 0);
        LogDebug("Window focus changed: ", focused ? "gained" : "lost");
    }
}

void WindowNative::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Key mapped_key = static_cast<Key>(key);

    // Update input manager
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    InputManager::GetInstance().SetKeyState(mapped_key, pressed);
}

void WindowNative::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    MouseButton mapped_button = static_cast<MouseButton>(button);

    // Update input manager
    bool pressed = (action == GLFW_PRESS);
    InputManager::GetInstance().SetMouseButtonState(mapped_button, pressed);
}

void WindowNative::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    InputManager::GetInstance().SetMousePosition(
        static_cast<float32>(xpos),
        static_cast<float32>(ypos)
    );
}

void WindowNative::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    InputManager::GetInstance().SetMouseScroll(
        static_cast<float32>(xoffset),
        static_cast<float32>(yoffset)
    );
}

}  // namespace platform
}  // namespace mps
