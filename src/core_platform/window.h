#pragma once

#include "core_util/types.h"
#include <string>
#include <memory>

namespace mps {
namespace platform {

// Window configuration
struct WindowConfig {
    std::string title = "MPS_DAWN";
    uint32 width = 1280;
    uint32 height = 720;
    bool resizable = true;
    bool fullscreen = false;
};

// Abstract Window interface
class IWindow {
public:
    virtual ~IWindow() = default;

    // Window lifecycle
    virtual bool Initialize(const WindowConfig& config) = 0;
    virtual void Shutdown() = 0;
    virtual void PollEvents() = 0;

    // Window state
    virtual bool ShouldClose() const = 0;
    virtual bool IsMinimized() const = 0;
    virtual bool IsFocused() const = 0;

    // Window properties
    virtual uint32 GetWidth() const = 0;
    virtual uint32 GetHeight() const = 0;
    virtual float32 GetAspectRatio() const = 0;
    virtual const std::string& GetTitle() const = 0;

    // Window operations
    virtual void SetTitle(const std::string& title) = 0;
    virtual void SetSize(uint32 width, uint32 height) = 0;
    virtual void SetFullscreen(bool fullscreen) = 0;

    // Native handle access (for surface creation in core_gpu)
    virtual void* GetNativeWindowHandle() const = 0;
    virtual void* GetNativeDisplayHandle() const = 0;

    // Factory method
    static std::unique_ptr<IWindow> Create();
};

}  // namespace platform
}  // namespace mps
