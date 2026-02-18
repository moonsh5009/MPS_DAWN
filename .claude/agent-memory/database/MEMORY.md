# Database Agent Memory

## Module Structure

The `core_database` module lives at `src/core_database/` and provides an ECS (Entity-Component-System) database with undo/redo transactions.

### Files
- `component_type.h` — ComponentTypeId alias, GetComponentTypeId<T>() static counter, Component concept
- `entity.h` / `entity.cpp` — Entity (uint32 alias), kInvalidEntity, EntityManager (create/destroy/recycle with free-list)
- `component_storage.h` — IComponentStorage interface, ComponentStorage<T> sparse set (swap-and-pop removal)
- `transaction.h` / `transaction.cpp` — IOperation, AddComponentOp<T>, RemoveComponentOp<T>, SetComponentOp<T>, Transaction, TransactionManager
- `database.h` / `database.cpp` — Database facade (Transact/Undo/Redo, entity/component CRUD)

### Key Design Decisions
- **Operation templates** are defined in `transaction.h` but their implementations are in `database.h` (after Database class) because they need full Database definition to call DirectAdd/Remove/SetComponent
- **DirectXxxComponent** methods bypass transaction recording — used by operation Apply/Revert during undo/redo replay
- **TransactionManager::Record()** is a no-op when no transaction is active, which is how undo/redo avoids double-recording
- Namespace: `mps::database`

## Build Notes
- The stub CMakeLists.txt existed but `src/CMakeLists.txt` did NOT include `add_subdirectory(core_database)` — had to add it
- Library output: `build/lib/x64/Debug/core_database.lib`
