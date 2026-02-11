# Git Changes Summary

Summarize git changes for review or documentation.

## What to Analyze
1. Commits since last push/tag
2. Files changed (added, modified, deleted)
3. Lines changed (additions/deletions)
4. Key functionality changes

## Commands to Use
```bash
# Recent commits
git log --oneline -10

# Commits since specific point
git log main..HEAD --oneline

# Detailed diff stats
git diff --stat main...HEAD

# Changed files only
git diff --name-status main...HEAD

# Full diff for review
git diff main...HEAD
```

## Summary Format
Provide a structured summary:

### Commits
- List recent commits with messages
- Highlight any breaking changes

### Files Changed
Categorize by type:
- **Added**: New files
- **Modified**: Changed files
- **Deleted**: Removed files

### Key Changes
- Functional changes (new features, bug fixes)
- Technical changes (refactoring, performance)
- Documentation/config changes

### Statistics
- Total commits
- Files changed
- Lines added/removed

## Example Output
```markdown
# Changes Summary (last 5 commits)

## Commits
1. Add WebGPU compute shader pipeline
2. Implement particle system update kernel
3. Fix memory leak in buffer allocation
4. Update CMakeLists for compute shader
5. Add compute shader documentation

## Files Changed
**Added** (3):
- src/compute/particle_update.wgsl
- src/compute/compute_pipeline.cpp
- src/compute/compute_pipeline.h

**Modified** (5):
- CMakeLists.txt
- src/main.cpp
- src/renderer.cpp
- Guide/ARCHITECTURE.md
- README.md

**Deleted** (1):
- src/legacy/old_particle.cpp

## Key Changes
- ✨ New: GPU-accelerated particle physics using compute shaders
- 🐛 Fixed: Memory leak in buffer pool management
- 📚 Docs: Architecture guide updated with compute pipeline

## Statistics
- 5 commits
- 9 files changed
- +523 insertions, -87 deletions
```
