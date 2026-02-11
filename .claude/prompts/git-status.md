# Git Status

Check and summarize the current git repository status.

## Information to Gather
1. Current branch name
2. Staged changes (ready to commit)
3. Unstaged changes (modified but not staged)
4. Untracked files (new files not in git)
5. Branch status (ahead/behind remote)

## Commands to Run
```bash
# Basic status
git status

# Concise status
git status -s

# Show branch tracking info
git status -sb

# See what would be committed
git diff --cached --stat

# See working directory changes
git diff --stat
```

## Provide Summary
After checking status, provide:
- Clear summary of what's changed
- Recommendation for next action (stage, commit, push, etc.)
- Warning if there are merge conflicts or other issues

## Example Output Format
```
Current branch: feature/webgpu-init
Status: 2 commits ahead of origin/main

Staged changes (ready to commit):
  M CMakeLists.txt
  A src/webgpu_init.cpp

Unstaged changes:
  M src/main.cpp

Untracked files:
  src/webgpu_init.h

Recommendation: Review unstaged changes, stage relevant files, then commit.
```
