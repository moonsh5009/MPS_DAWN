# core_database

> Host-side ECS (Entity-Component-System) with undo/redo transactions.

## Module Structure

```
src/core_database/
├── CMakeLists.txt          # STATIC library → mps::core_database (depends: core_util)
├── component_type.h        # ComponentTypeId, GetComponentTypeId<T>(), Component concept
├── entity.h / entity.cpp   # Entity (uint32 alias), EntityManager (create/destroy/recycle)
├── component_storage.h     # IComponentStorage, ComponentStorage<T> (sparse set)
├── array_storage.h         # IArrayStorage, ArrayStorage<T> (per-entity variable-length arrays)
├── transaction.h / .cpp    # IOperation, Add/Remove/SetComponentOp<T>, Transaction, TransactionManager
├── array_transaction.h     # SetArrayOp<T>, RemoveArrayOp<T> (undo/redo for arrays)
└── database.h / database.cpp  # Database facade (Transact, Undo/Redo, CRUD, array ops)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `ComponentTypeId` | `component_type.h` | `uint32` alias, unique per component type |
| `Component` concept | `component_type.h` | `trivially_copyable && standard_layout` |
| `Entity` | `entity.h` | `uint32` alias, `kInvalidEntity = UINT32_MAX` |
| `EntityManager` | `entity.h` | Creates/destroys/recycles entity IDs |
| `IComponentStorage` | `component_storage.h` | Type-erased base for sparse set storage |
| `ComponentStorage<T>` | `component_storage.h` | Sparse set with swap-and-pop O(1) removal, dirty flag |
| `IOperation` | `transaction.h` | Base for undoable operations |
| `AddComponentOp<T>` | `transaction.h` | Records component addition (Apply/Revert) |
| `RemoveComponentOp<T>` | `transaction.h` | Records component removal |
| `SetComponentOp<T>` | `transaction.h` | Records component value change |
| `Transaction` | `transaction.h` | Group of operations committed atomically |
| `TransactionManager` | `transaction.h` | Manages transaction stack and undo/redo history |
| `IArrayStorage` | `array_storage.h` | Type-erased base for per-entity array storage |
| `ArrayStorage<T>` | `array_storage.h` | Stores variable-length `vector<T>` per entity, dirty flag |
| `SetArrayOp<T>` | `array_transaction.h` | Records array set (Apply/Revert) |
| `RemoveArrayOp<T>` | `array_transaction.h` | Records array removal |
| `Database` | `database.h` | ECS facade with transactions, undo/redo, and array ops |

## API

### Database

```cpp
// Entity management
Entity CreateEntity();
void DestroyEntity(Entity entity);

// Component CRUD (records into active transaction)
template<Component T> void AddComponent(Entity, const T&);
template<Component T> void RemoveComponent(Entity);
template<Component T> void SetComponent(Entity, const T&);
template<Component T> T* GetComponent(Entity);
template<Component T> const T* GetComponent(Entity) const;
template<Component T> bool HasComponent(Entity) const;

// Transaction — lambda-based RAII
void Transact(std::function<void()> fn);

// Undo/Redo
bool Undo();
bool Redo();
bool CanUndo() const;
bool CanRedo() const;

// Storage access (for DeviceDB sync)
IComponentStorage* GetStorageById(ComponentTypeId);
const IComponentStorage* GetStorageById(ComponentTypeId) const;
std::vector<ComponentTypeId> GetDirtyTypeIds() const;
void ClearAllDirty();

// Array operations (per-entity variable-length arrays)
template<Component T> void SetArray(Entity, std::vector<T> data);
template<Component T> const std::vector<T>* GetArray(Entity) const;
template<Component T> void RemoveArray(Entity);
template<Component T> bool HasArray(Entity) const;

// Array storage access (for DeviceArrayBuffer sync)
IArrayStorage* GetArrayStorageById(ComponentTypeId);
const IArrayStorage* GetArrayStorageById(ComponentTypeId) const;
std::vector<ComponentTypeId> GetDirtyArrayTypeIds() const;

// Direct operations (no transaction recording — used by undo/redo replay)
template<Component T> void DirectAddComponent(Entity, const T&);
template<Component T> void DirectRemoveComponent(Entity);
template<Component T> void DirectSetComponent(Entity, const T&);
template<Component T> void DirectSetArray(Entity, std::vector<T> data);
template<Component T> void DirectRemoveArray(Entity);
```

### IComponentStorage

```cpp
virtual uint32 GetDenseCount() const = 0;
virtual const void* GetDenseData() const = 0;
virtual bool IsDirty() const = 0;
virtual void ClearDirty() = 0;
```

### ComponentStorage<T>

```cpp
void Add(Entity, const T&);
void Remove(Entity);
void Set(Entity, const T&);
T* Get(Entity);
const T* Get(Entity) const;
bool Contains(Entity) const;
const std::vector<Entity>& GetEntities() const;
```

### ArrayStorage<T>

```cpp
void SetArray(Entity, std::vector<T> data);
const std::vector<T>* GetArray(Entity) const;
uint32 GetCount(Entity) const;
bool Has(Entity) const override;
void Remove(Entity) override;
bool IsDirty() const override;
void ClearDirty() override;
std::vector<Entity> GetEntities() const override;
const void* GetArrayData(Entity) const override;
uint32 GetArrayCount(Entity) const override;
uint32 GetElementSize() const override;
```

## Design Patterns

- **Lambda Transact**: `BeginTransaction()` → `fn()` → `Commit()`, catch → `Rollback()` → rethrow
- **DirectXxxComponent** methods bypass transaction recording — used by operation Apply/Revert during undo/redo
- **TransactionManager::Record()** is a no-op when no transaction is active (prevents double-recording during undo/redo)
- **Sparse set**: `sparse_[]` maps entity → dense index, `dense_[]` is contiguous (GPU-upload friendly)
- **Dirty flags** are set even during undo/redo (triggers DeviceDB sync)
- **Operation template impls** live in `database.h` after Database class definition because they need `DirectXxxComponent` methods
