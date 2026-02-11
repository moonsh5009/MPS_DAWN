# Git Push

Push commits to the remote repository safely.

## Safety Checks
Before pushing, verify:
1. You're on the correct branch (`git branch --show-current`)
2. Local branch is up to date with remote (`git fetch && git status`)
3. No unwanted files are being pushed (check recent commits with `git log -3 --stat`)

## Push Process
1. For first push of a new branch: `git push -u origin <branch-name>`
2. For subsequent pushes: `git push`
3. If rejected, check for remote changes: `git pull --rebase` then `git push`

## Force Push Warning
⚠️ **NEVER** use `git push --force` on shared branches (main/master/develop)
- Only use on personal feature branches if absolutely necessary
- Prefer `git push --force-with-lease` for safer force pushing
- Always communicate with team before force pushing shared branches

## Common Scenarios

### Push new branch
```bash
git push -u origin feature/new-feature
```

### Push with upstream tracking
```bash
git push -u origin HEAD
```

### Push tags
```bash
git push --tags
```

## After Push
Verify push was successful:
- Check GitHub/remote repository
- Verify CI/CD pipeline starts (if configured)
