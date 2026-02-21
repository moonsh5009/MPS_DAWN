# core_platform

> Window management and input handling with cross-platform support.

## Module Structure

```
src/core_platform/
├── CMakeLists.txt        # STATIC library → mps::core_platform (depends: core_util, glfw [native only])
├── window.h / .cpp       # IWindow interface + factory (Create())
├── window_native.h/.cpp  # GLFW-based native window
├── window_wasm.h/.cpp    # Emscripten canvas window
└── input.h / .cpp        # InputManager singleton + convenience functions
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `WindowConfig` | `window.h` | `title`, `width`, `height`, `resizable`, `fullscreen` |
| `IWindow` | `window.h` | Abstract: `Initialize`, `PollEvents`, `ShouldClose`, `GetNativeWindowHandle`, `Create()` factory |
| `WindowNative` | `window_native.h` | GLFW-based native window implementation |
| `WindowWasm` | `window_wasm.h` | Emscripten canvas window implementation |
| `Key` | `input.h` | `enum class : uint16` — alphanumeric, function, arrow, control keys (GLFW-compatible values) |
| `MouseButton` | `input.h` | `enum class : uint8` — Left, Right, Middle, Button4, Button5 |
| `InputState` | `input.h` | `enum class : uint8` — Released, Pressed, Held |
| `InputManager` | `input.h` | Singleton: keyboard + mouse state tracking |

## API

### IWindow

```cpp
static std::unique_ptr<IWindow> Create();  // Factory: WindowNative or WindowWasm
virtual bool Initialize(const WindowConfig& config) = 0;
virtual void Shutdown() = 0;
virtual void PollEvents() = 0;

virtual bool ShouldClose() const = 0;
virtual bool IsMinimized() const = 0;
virtual bool IsFocused() const = 0;

virtual uint32 GetWidth() const = 0;
virtual uint32 GetHeight() const = 0;
virtual float32 GetAspectRatio() const = 0;
virtual const std::string& GetTitle() const = 0;

virtual void SetTitle(const std::string& title) = 0;
virtual void SetSize(uint32 width, uint32 height) = 0;
virtual void SetFullscreen(bool fullscreen) = 0;

virtual void* GetNativeWindowHandle() const = 0;
virtual void* GetNativeDisplayHandle() const = 0;
```

### InputManager

```cpp
static InputManager& GetInstance();
void Update();

// Keyboard
void SetKeyState(Key key, bool pressed);
bool IsKeyPressed(Key key) const;
bool IsKeyHeld(Key key) const;
bool IsKeyReleased(Key key) const;

// Mouse buttons
void SetMouseButtonState(MouseButton button, bool pressed);
bool IsMouseButtonPressed(MouseButton button) const;
bool IsMouseButtonHeld(MouseButton button) const;
bool IsMouseButtonReleased(MouseButton button) const;

// Mouse position
void SetMousePosition(float32 x, float32 y);
util::vec2 GetMousePosition() const;
util::vec2 GetMouseDelta() const;

// Mouse scroll (double-buffered: accumulated between frames, consumed on Update)
void SetMouseScroll(float32 x, float32 y);         // Direct set (native GLFW)
void AccumulateMouseScroll(float32 x, float32 y);  // Additive (WASM async events)
util::vec2 GetMouseScroll() const;
```

### Free Functions (`mps::platform`)

```cpp
// Keyboard
IsKeyPressed(Key), IsKeyHeld(Key), IsKeyReleased(Key)

// Mouse
IsMouseButtonPressed(MouseButton), IsMouseButtonHeld(MouseButton), IsMouseButtonReleased(MouseButton)
GetMousePosition(), GetMouseDelta(), GetMouseScroll()
```
