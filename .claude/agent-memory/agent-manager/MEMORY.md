# Agent Manager Memory

## Agent System Architecture

- Claude Code native subagents: `.claude/agents/*.md` (YAML frontmatter)
- Memory: `memory: project` → `.claude/agent-memory/<name>/MEMORY.md`

## Key Design Decision

Domain-based over role-based: each agent owns specific modules (e.g., "GPU Agent" owns `core_gpu`) rather than generic roles (e.g., "Build Agent").

## Evolution

1. **2026-02-12**: Initial CLAUDE.md + role-based agents
2. **2026-02-14**: Migrated to domain-based Claude Code native subagent format (current)
3. **2026-02-18**: ECS + DeviceDB + extension system → added system agent, updated simulate/database/render agents
4. **2026-02-19**: Added build-debug agent; ext_cloth extension; core_gpu gained compute pipeline/encoder

## Pending Work

- Refine agent boundaries if overlap is discovered
