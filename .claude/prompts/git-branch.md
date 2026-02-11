# Git Branch Management

Manage git branches - create, switch, delete, and organize branches.

## Common Branch Operations

### List Branches
```bash
# Local branches
git branch

# All branches (local + remote)
git branch -a

# With last commit info
git branch -v
```

### Create New Branch
```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Create from specific commit/branch
git checkout -b feature/branch-name origin/main
```

### Switch Branches
```bash
# Switch to existing branch
git checkout branch-name

# Or using newer syntax
git switch branch-name
```

### Delete Branches
```bash
# Delete local branch (safe - won't delete if unmerged)
git branch -d branch-name

# Force delete local branch
git branch -D branch-name

# Delete remote branch
git push origin --delete branch-name
```

## Branch Naming Convention
Follow these conventions for this project:
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates
- `test/description` - Test additions/updates

Examples:
- `feature/webgpu-compute-shader`
- `bugfix/memory-leak-in-particle-system`
- `refactor/simplify-renderer`

## Best Practices
1. Keep branch names short but descriptive
2. Use lowercase with hyphens (kebab-case)
3. Delete branches after merging
4. Regularly sync with main: `git fetch origin && git rebase origin/main`
5. One feature/fix per branch

## Branch Status Check
Before creating a new branch, check:
- Are you on the correct base branch? (usually main)
- Is your base branch up to date? `git pull origin main`
- Any uncommitted changes? `git status`
