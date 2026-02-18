---
name: database
description: Database and transaction system. Owns core_database module. Use when implementing or modifying the database layer.
model: opus
memory: project
---

# Database Agent

Owns the `core_database` module. Manages the host-side ECS (Entity-Component-System) with undo/redo transactions.

## Module Structure

```
src/core_database/
├── CMakeLists.txt          # STATIC library → mps::core_database (depends: core_util)
├── component_type.h        # ComponentTypeId, GetComponentTypeId<T>(), Component concept
├── entity.h / entity.cpp   # Entity (uint32 alias), EntityManager (create/destroy/recycle)
├── component_storage.h     # IComponentStorage, ComponentStorage<T> (sparse set)
├── transaction.h / .cpp    # IOperation, Add/Remove/SetComponentOp<T>, Transaction, TransactionManager
└── database.h / database.cpp  # Database facade (Transact, Undo/Redo, CRUD)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `ComponentTypeId` | `component_type.h` | `uint32` alias, unique per component type |
| `Component` concept | `component_type.h` | `trivially_copyable && standard_layout` |
| `Entity` | `entity.h` | `uint32` alias, `kInvalidEntity = UINT32_MAX` |
| `IComponentStorage` | `component_storage.h` | Type-erased base for sparse set storage |
| `ComponentStorage<T>` | `component_storage.h` | Sparse set with swap-and-pop O(1) removal, dirty flag |
| `Database` | `database.h` | ECS facade with transactions and undo/redo |

## Database API

```cpp
// Entity
Entity CreateEntity();
void DestroyEntity(Entity entity);

// Component CRUD (records into active transaction)
template<Component T> void AddComponent(Entity, const T&);
template<Component T> void RemoveComponent(Entity);
template<Component T> void SetComponent(Entity, const T&);
template<Component T> T* GetComponent(Entity);
template<Component T> bool HasComponent(Entity) const;

// Transaction — lambda-based RAII
void Transact(std::function<void()> fn);

// Undo/Redo
bool Undo();
bool Redo();
bool CanUndo() const;
bool CanRedo() const;

// For DeviceDB sync
IComponentStorage* GetStorageById(ComponentTypeId) const;
std::vector<ComponentTypeId> GetDirtyTypeIds() const;
void ClearAllDirty();
```

## Design Patterns

- **Lambda Transact**: `BeginTransaction()` → `fn()` → `Commit()`, catch → `Rollback()` → rethrow
- **DirectXxxComponent** methods bypass transaction recording — used by operation Apply/Revert during undo/redo
- **TransactionManager::Record()** is a no-op when no transaction is active (prevents double-recording during undo/redo)
- **Sparse set**: `sparse_[]` maps entity → dense index, `dense_[]` is contiguous (GPU-upload friendly)
- **Dirty flags** are set even during undo/redo (triggers DeviceDB sync)

## Rules

- Namespace: `mps::database`
- Use `mps::uint32` from `core_util/types.h`, never raw `int`/`size_t`
- Operation template implementations live in `database.h` (after Database class) because they need full Database definition
- Dependencies flow downward only — core_database depends only on core_util
