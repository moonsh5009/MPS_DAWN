# core_system

> System controller — orchestrates database, simulate, render, and extensions.

## Module Structure

```
src/core_system/
├── CMakeLists.txt         # STATIC library → mps::core_system (depends: core_util, core_platform, core_gpu, core_database, core_simulate, core_render)
├── system.h / system.cpp  # System facade (lifecycle, transactions, extension orchestration)
└── extension.h            # IExtension interface
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `System` | `system.h` | Top-level controller: component registration, transactions with auto GPU sync, undo/redo, simulation controls, extension lifecycle, term provider registry |
| `IExtension` | `extension.h` | Extension entry point: `GetName()`, `Register(System&)` |

## API

### System

```cpp
// Lifecycle
bool Initialize();   // Creates window, GPU, RenderEngine
void Run();          // Initializes extensions, enters main loop

// Simulation control
bool IsSimulationRunning() const;
void SetSimulationRunning(bool running);
void ResetSimulation();  // ForceSync + pause

// Component registration (sparse-set ECS)
template<database::Component T>
void RegisterComponent(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                       const std::string& label = "");

// Array registration (concatenated GPU buffers from ArrayStorage)
template<database::Component T>
void RegisterArray(gpu::BufferUsage extra_usage = gpu::BufferUsage::None,
                   const std::string& label = "");

// Array queries
template<database::Component T>
uint32 GetArrayTotalCount() const;

// Transactions (auto GPU sync + simulator topology notification)
void Transact(std::function<void(database::Database&)> fn);
void Undo();
void Redo();
bool CanUndo() const;
bool CanRedo() const;

// GPU buffer access
template<database::Component T>
WGPUBuffer GetDeviceBuffer() const;

// Read-only queries
template<database::Component T>
uint32 GetComponentCount() const;

// GPU → Database readback
template<database::Component T>
void Snapshot();

// Raw GPU buffer readback (synchronous map)
std::vector<uint8> ReadbackBuffer(WGPUBuffer src, uint64 size);

// Host database access
const database::Database& GetDatabase() const;
database::Database& GetDatabase();  // non-const overload (for simulators)
simulate::DeviceDB& GetDeviceDB();

// Extension registration (called by extensions during Register)
void AddExtension(std::unique_ptr<IExtension> extension);
void AddSimulator(std::unique_ptr<simulate::ISimulator> simulator);
void AddRenderer(std::unique_ptr<render::IObjectRenderer> renderer);

// Term provider registry (for Newton system)
void RegisterTermProvider(database::ComponentTypeId config_type,
                          std::unique_ptr<simulate::IDynamicsTermProvider> provider);
simulate::IDynamicsTermProvider* FindTermProvider(database::Entity constraint_entity) const;
```

### IExtension

```cpp
virtual const std::string& GetName() const = 0;
virtual void Register(System& system) = 0;
```

## Extension Flow

```
main.cpp
  ├── system.Initialize()              ← window + GPU (native: sync wait; WASM: async, deferred)
  ├── system.AddExtension(ext)         ← calls ext->Register(system) immediately
  │     └── ext->Register(system)      ← registers components, simulators, renderers
  ├── system.Transact(...)             ← create initial scene entities
  ├── system.Run()
  │     ├─ [Native] InitializeExtensions() → while loop
  │     ├─ [WASM]   emscripten_set_main_loop_arg → EmscriptenMainLoop callback
  │     │     ├── Wait for GPU ready → FinishGPUInit() → InitializeExtensions()
  │     │     └── RunFrame(dt) on subsequent frames
  │     └── RunFrame(dt):
  │           ├── PollEvents + keyboard/mouse checks
  │           ├── UpdateSimulators(dt)  ← sim->Update(dt) (no wrapping transaction)
  │           ├── engine->UpdateUniforms(dt)
  │           ├── RenderFrame()        ← renderers sorted by GetOrder()
  │           └── input.Update()       ← transitions Pressed→Held (at END for WASM timing)
  └── ~System()                        ← shutdown extensions, engine, GPU, window
```

## Term Provider Registry

Extensions register `IDynamicsTermProvider` implementations via `RegisterTermProvider()`. Newton system simulators discover terms at runtime by iterating constraint entity references and calling `FindTermProvider()`.

```
Extension::Register()              System (registry)              NewtonSystemSimulator
  │                                   │                               │
  ├─ RegisterTermProvider ──────►  type → Provider              Initialize():
  │   (GravityConstraintData        │                               │
  │    → GravityTermProvider)       │                            1. Read config.constraint_refs
  │                                 │                            2. For each ref entity:
  ├─ RegisterTermProvider ──────►  SpringConstraintData          3.   FindTermProvider(entity)
  │   (→ SpringTermProvider)       │                            4.   provider->CreateTerm()
  └─                               ▼                            5.   dynamics->AddTerm(term)
```

## Data Flow

```
User code ──Transact()──► Database (host)
                              │
                         SyncToDevice()
                              │
                              ▼
                         DeviceDB (GPU)
                              │
                    NotifyDatabaseChanged()
                              │
                              ▼
                    sim->OnDatabaseChanged()  ← topology change detection + reinit
                              │
                    GetDeviceBuffer<T>()
                              │
                              ▼
                      core_render (reads handles)
```
