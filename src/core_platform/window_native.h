#pragma once

#include "core_platform/window.h"

// Forward declaration for GLFW
struct GLFWwindow;

namespace mps {
namespace platform {

// Native window implementation using GLFW
class WindowNative : public IWindow {
public:
    WindowNative();
    ~WindowNative() override;

    // IWindow implementation
    bool Initialize(const WindowConfig& config) override;
    void Shutdown() override;
    void PollEvents() override;

    bool ShouldClose() const override;
    bool IsMinimized() const override;
    bool IsFocused() const override;

    uint32 GetWidth() const override;
    uint32 GetHeight() const override;
    float32 GetAspectRatio() const override;
    const std::string& GetTitle() const override;

    void SetTitle(const std::string& title) override;
    void SetSize(uint32 width, uint32 height) override;
    void SetFullscreen(bool fullscreen) override;

    WGPUSurface CreateSurface(WGPUInstance instance) override;

    // GLFW window handle
    GLFWwindow* GetGLFWWindow() const { return window_; }

private:
    // Callbacks
    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void WindowFocusCallback(GLFWwindow* window, int focused);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    GLFWwindow* window_ = nullptr;
    WindowConfig config_;
    bool is_focused_ = true;
};

}  // namespace platform
}  // namespace mps
