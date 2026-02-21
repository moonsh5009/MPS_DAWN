#pragma once

#include "core_platform/input.h"
#include "core_platform/window.h"

namespace mps {
namespace platform {

// WASM window implementation using Emscripten
class WindowWasm : public IWindow {
public:
    WindowWasm();
    ~WindowWasm() override;

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

    void* GetNativeWindowHandle() const override;
    void* GetNativeDisplayHandle() const override;

private:
    void RegisterInputCallbacks();

    // Static input event handlers
    static Key MapDOMKeyCode(unsigned long code, unsigned long location);
    static MouseButton MapDOMMouseButton(unsigned short button);

    WindowConfig config_;
    bool should_close_ = false;
    bool is_focused_ = true;
    uint32 current_width_ = 0;
    uint32 current_height_ = 0;
};

}  // namespace platform
}  // namespace mps
