#include "core_platform/window.h"

#ifdef __EMSCRIPTEN__
#include "core_platform/window_wasm.h"
#else
#include "core_platform/window_native.h"
#endif

namespace mps {
namespace platform {

std::unique_ptr<IWindow> IWindow::Create() {
#ifdef __EMSCRIPTEN__
    return std::make_unique<WindowWasm>();
#else
    return std::make_unique<WindowNative>();
#endif
}

}  // namespace platform
}  // namespace mps
