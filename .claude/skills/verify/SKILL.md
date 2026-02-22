---
name: verify
description: Build and run verification for both native (Windows) and WASM platforms
user_invocable: true
---

# Build & Run Verification

Full verification workflow: build both platforms, run the native executable, and check for errors.

## Step 1: Native Debug Build

```bash
cmake -B build && cmake --build build --config Debug
```

**Check:**
- [ ] CMake configures without errors
- [ ] Build completes without errors (C4819 codepage warnings OK)
- [ ] Output: `build\bin\x64\Debug\mps_dawn.exe`

## Step 2: WASM Debug Build

```bash
# Windows (requires emsdk + standalone ninja, NOT depot_tools ninja):
cmd.exe //c "C:\emsdk\emsdk_env.bat >nul 2>&1 && set PATH=C:\Users\user\AppData\Local\Programs\Python\Python313\Scripts;%PATH% && emcmake cmake -B build-wasm/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-wasm/debug"
```

**Check:**
- [ ] CMake configures with Emscripten toolchain
- [ ] Build completes without errors
- [ ] Output: `build-wasm/bin/Debug/mps_dawn.html`

## Step 3: Native Runtime Verification

Claude Code sandbox blocks GPU `LoadLibrary`, so use **Task Scheduler** to run the exe outside the sandbox.

**IMPORTANT — Enable simulation before runtime test:**
Task Scheduler cannot send keyboard input, so the simulation will stay paused (default). Temporarily set the initial state to running:

1. In `src/core_system/system.h`, change `simulation_running_` initial value:
   ```cpp
   bool simulation_running_ = true;  // was false
   ```
2. Rebuild: `cmake --build build --config Debug`
3. Run the runtime test (steps below)
4. **After verification, revert immediately:**
   ```cpp
   bool simulation_running_ = false;  // restore default
   ```

```bash
# 1. Create a bat file for clean execution
cat > /c/repositories/cpp/MPS_DAWN/do_verify.bat << 'EOF'
@echo off
cd /d C:\repositories\cpp\MPS_DAWN
schtasks /Create /TN "mps_verify" /TR "cmd /c cd /d C:\repositories\cpp\MPS_DAWN & build\bin\x64\Debug\mps_dawn.exe > verify_output.txt 2>&1" /SC ONCE /ST 00:00 /F
schtasks /Run /TN "mps_verify"
EOF

# 2. Run via cmd.exe to avoid git-bash path mangling
cmd.exe //c "C:\\repositories\\cpp\\MPS_DAWN\\do_verify.bat"

# 3. Wait for startup + capture output
sleep 8
cat /c/repositories/cpp/MPS_DAWN/verify_output.txt

# 4. Cleanup
cmd.exe //c "schtasks /End /TN mps_verify" > /dev/null 2>&1
cmd.exe //c "schtasks /Delete /TN mps_verify /F" > /dev/null 2>&1
```

**Check log output for:**
- [ ] `GPU initialized:` — D3D12 backend active
- [ ] All extensions registered (ext_dynamics, ext_mesh, ext_newton, ext_pd)
- [ ] All term providers registered (SpringTermProvider, AreaTermProvider, PDSpringTermProvider, PDAreaTermProvider)
- [ ] All simulators initialized (NewtonSystemSimulator, PDSystemSimulator, MeshPostProcessor)
- [ ] All renderers initialized (MeshRenderer)
- [ ] `Entering main loop...` — no crash before frame loop
- [ ] No `[ERROR]` lines in output

## Step 4: Verify Expected Counts

From a 64x64 cloth grid, expected values:

| Metric | Value |
|--------|-------|
| Nodes (SimPosition) | 4096 |
| Edges (springs) | ~16000 (structural + bending) |
| Faces | 7938 |
| NNZ (CSR off-diagonal) | ~32000 |
| Newton terms | 2 (Spring + Area) — Inertial + Gravity are built into NewtonDynamics |
| PD terms | 2 (PDSpringTerm + PDAreaTerm) |
| Simulators | 3 (NewtonSystemSimulator + PDSystemSimulator + MeshPostProcessor) |
| Renderers | 1 (MeshRenderer) |

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `schtasks` path mangled | Git bash converts `/Create` to path | Use bat file wrapper with `cmd.exe //c` |
| `verify_output.txt` empty | Task didn't start or crashed immediately | Check `schtasks /Query /TN mps_verify` status |
| Incomplete type error (WASM) | `unique_ptr<T>` with forward-declared T | Add out-of-line destructor (`~Class(); ... = default;` in .cpp) |
| `d3dcompiler_47.dll` not found | Dawn DLL path mismatch | Set `DAWN_FORCE_SYSTEM_COMPONENT_LOAD ON` |
| Shader not found | Missing `assets/` copy | Check `POST_BUILD copy_directory` in CMakeLists |
| WASM ninja not found | depot_tools ninja is broken | Use pip-installed ninja: `pip install ninja` |

## Cleanup

Remove temp files after verification:
```bash
rm -f verify_output.txt do_verify.bat
```
