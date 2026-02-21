---
name: database
description: Database and transaction system. Owns core_database module. Use when implementing or modifying the database layer.
model: opus
---

# Database Agent

Owns the `core_database` module. Manages the host-side ECS (Entity-Component-System) with undo/redo transactions.

> **CRITICAL**: ALWAYS read `.claude/docs/core_database.md` FIRST before any task. This doc contains the complete file tree, types, APIs, and shader references. DO NOT read source files (.h/.cpp) to understand the module — only read source files when you need to edit them.

## When to Use This Agent

- Implementing or modifying entity/component CRUD operations
- Adding new component types or storage behavior
- Modifying transaction, undo, or redo logic
- Changing dirty flag or DeviceDB sync interaction

## Task Guidelines

- Namespace: `mps::database`
- Use `mps::uint32` from `core_util/types.h`, never raw `int`/`size_t`
- Operation template implementations (`AddComponentOp<T>::Apply`, etc.) must live in `database.h` **after** the Database class definition, because they need `DirectXxxComponent` methods
- Dependencies flow downward only — core_database depends only on core_util
- `DirectXxxComponent` methods bypass transaction recording — used only by operation Apply/Revert during undo/redo
- `TransactionManager::Record()` is a no-op when no transaction is active (prevents double-recording during undo/redo)
- Dirty flags are set even during undo/redo (ensures DeviceDB sync triggers)

## Common Tasks

### Adding a new component type

No changes to core_database needed — any `trivially_copyable && standard_layout` struct satisfying the `Component` concept works automatically. Registration happens in the extension's `Register()` via `System::RegisterComponent<T>()`.

### Adding a new operation type

1. Define in `transaction.h` inheriting from `IOperation`
2. Implement `Apply(Database&)` and `Revert(Database&)` using `DirectXxxComponent` methods
3. Place template implementation in `database.h` after Database class definition
4. Record via `transaction_manager_.Record()` in the corresponding Database method
