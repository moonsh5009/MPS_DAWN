# MPS_DAWN Project Instructions

This file contains project-wide instructions that apply to all Claude Code sessions working on this project.

## Build and Testing Policy

### Build Tests
- **DO NOT** perform build tests unless explicitly requested by the user
- Only execute build commands when the user specifically asks to:
  - "빌드해줘" / "build it"
  - "테스트해줘" / "test it"
  - "실행해줘" / "run it"
- When making changes to CMakeLists.txt or build configuration:
  - Explain the changes
  - DO NOT automatically verify with a build
  - Wait for user confirmation before building

### Example
❌ **Incorrect**:
```
I've updated CMakeLists.txt. Let me build to verify...
[runs cmake --build]
```

✅ **Correct**:
```
I've updated CMakeLists.txt with the new output directory settings.
The changes are complete. Let me know if you'd like me to test the build.
```

---

## Project Understanding

### Reference Documentation
- **ALWAYS** refer to the markdown files in `.claude/guide/` to understand the project's current state
- Key guide files:
  - `DEVELOPMENT_SETUP.md` - Complete development environment setup and build configuration
  - (More guides will be added as the project progresses)
- These guides contain:
  - Current project structure
  - Build system configuration
  - Completed setup steps
  - Known issues and solutions
- Read these files before making suggestions or changes to understand what's already been done

---

## Additional Instructions
(Instructions will be added here as the project progresses)
