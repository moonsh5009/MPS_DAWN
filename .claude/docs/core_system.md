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

// Indexed array registration (topology arrays with per-entity offset transform)
template<database::Component T, database::Component RefT>
void RegisterIndexedArray(gpu::BufferUsage extra_usage, const std::string& label,
                          simulate::IndexOffsetFn<T> offset_fn);

// Array queries
template<database::Component T>
uint32 GetArrayTotalCount() const;

// Get a type-erased array entry by component type id
simulate::IDeviceArrayEntry* GetArrayEntryById(database::ComponentTypeId id) const;

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

// Host database access
const database::Database& GetDatabase() const;
database::Database& GetDatabase();  // non-const overload (for simulators)
simulate::DeviceDB& GetDeviceDB();

// Extension registration (called by extensions during Register)
void AddExtension(std::unique_ptr<IExtension> extension);
void AddSimulator(std::unique_ptr<simulate::ISimulator> simulator);
void AddRenderer(std::unique_ptr<render::IObjectRenderer> renderer);

// Newton term provider registry
void RegisterTermProvider(database::ComponentTypeId config_type,
                          std::unique_ptr<simulate::IDynamicsTermProvider> provider);
simulate::IDynamicsTermProvider* FindTermProvider(database::Entity constraint_entity) const;

// Find ALL providers whose config component exists on the given entity.
std::vector<simulate::IDynamicsTermProvider*> FindAllTermProviders(
    database::Entity constraint_entity) const;

// PD term provider registry
void RegisterPDTermProvider(database::ComponentTypeId config_type,
                            std::unique_ptr<simulate::IProjectiveTermProvider> provider);
simulate::IProjectiveTermProvider* FindPDTermProvider(database::Entity constraint_entity) const;
std::vector<simulate::IProjectiveTermProvider*> FindAllPDTermProviders(
    database::Entity constraint_entity) const;
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
  │     │     └── RunFrame() on subsequent frames
  │     └── RunFrame():  // fixed timestep (1/60), no frame timer
  │           ├── PollEvents + keyboard/mouse checks
  │           ├── UpdateSimulators()   ← sim->Update() (no wrapping transaction)
  │           ├── engine->UpdateUniforms(dt)  ← constexpr dt for camera/render only
  │           ├── RenderFrame()        ← renderers sorted by GetOrder()
  │           └── input.Update()       ← transitions Pressed→Held (at END for WASM timing)
  └── ~System()                        ← shutdown extensions, engine, GPU, window
```

## Term Provider Registries

Two separate registries — one for Newton terms (`IDynamicsTermProvider`), one for PD terms (`IProjectiveTermProvider`).

### Newton Term Registry

Extensions register `IDynamicsTermProvider` via `RegisterTermProvider()`. `NewtonSystemSimulator` discovers terms via `FindTermProvider()`.

```
ext_newton::Register()              System (registry)              NewtonSystemSimulator
  │                                   │                               │
  ├─ RegisterTermProvider ──────►  SpringConstraintData          Initialize():
  │   (→ SpringTermProvider)       │                            1. Read config.constraint_refs
  ├─ RegisterTermProvider ──────►  AreaConstraintData            2. For each ref entity:
  │   (→ AreaTermProvider)         │                            3.   FindAllTermProviders(entity)
  └─                               ▼                            4.   provider->CreateTerm()
                                                                 5.   dynamics->AddTerm(term)
```

### PD Term Registry

Extensions register `IProjectiveTermProvider` via `RegisterPDTermProvider()`. `PDSystemSimulator` discovers terms via `FindPDTermProvider()`.

```
ext_pd::Register()                  System (registry)              PDSystemSimulator
  │                                   │                               │
  ├─ RegisterPDTermProvider ────►  SpringConstraintData          Initialize():
  │   (→ PDSpringTermProvider)     │                            1. Read config.constraint_refs
  ├─ RegisterPDTermProvider ────►  AreaConstraintData            2. For each ref entity:
  │   (→ PDAreaTermProvider)       │                            3.   FindAllPDTermProviders(entity)
  └─                               ▼                            4.   provider->CreateTerm()
                                                                 5.   dynamics->AddTerm(term)
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
