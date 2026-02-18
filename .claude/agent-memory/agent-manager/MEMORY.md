# Agent Manager Memory

## Agent System Architecture

- Claude Code native subagents: `.claude/agents/*.md` (YAML frontmatter)
- Memory: `memory: project` → `.claude/agent-memory/<name>/MEMORY.md`

## Evolution

1. **2026-02-12**: Initial CLAUDE.md + role-based agents (Architect, Build, Review, etc.)
2. **2026-02-12**: Split into CLAUDE.md (workflow) + `.claude/guide/*.md` (technical)
3. **2026-02-14**: Domain-based agents with folder structure (`guide.md` + `history.md`)
4. **2026-02-14**: Migrated to Claude Code native subagent format (current)

## Key Design Decision

Domain-based over role-based: each agent owns specific modules (e.g., "GPU Agent" owns `core_gpu`) rather than generic roles (e.g., "Build Agent").

## Completed Work

- **2026-02-18**: Updated database and simulate agent definitions after ECS + DeviceDB implementation
- **2026-02-18**: Updated render agent with clarified role (GPU buffer consumer, no ECS knowledge)
- **2026-02-18**: Created system agent for core_system module
- **2026-02-18**: Module audit — moved builders/surface/shader from core_render to core_gpu, updated all agent docs

## Pending Work

- Refine agent boundaries if overlap is discovered
