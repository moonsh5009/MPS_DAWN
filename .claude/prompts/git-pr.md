# Create Pull Request

Create a pull request for the current branch.

## Before Creating PR
1. Ensure all changes are committed
2. Push branch to remote: `git push -u origin <branch-name>`
3. Review diff against base branch: `git diff main...HEAD`
4. Check all tests pass locally

## PR Information Needed
Gather information for PR description:
- **Title**: Clear, concise summary (under 70 chars)
- **Summary**: What changes were made and why
- **Test Plan**: How to test/verify the changes
- **Related Issues**: Link to issues (if any)

## PR Creation

### Using GitHub CLI (gh)
```bash
gh pr create --title "Title" --body "$(cat <<'EOF'
## Summary
- Bullet points of changes

## Test Plan
- Steps to test

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Manual (provide GitHub URL)
If `gh` is not available, provide:
1. GitHub PR creation URL
2. Formatted PR description to copy/paste

## PR Description Template
```markdown
## Summary
[Brief description of what this PR does]

### Changes
- Change 1
- Change 2
- Change 3

## Test Plan
- [ ] Built and tested locally
- [ ] All tests passing
- [ ] Manual testing completed

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Related Issues
Fixes #[issue number]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## After PR Creation
- Return PR URL
- Mention if there are any CI/CD checks to monitor
- Suggest requesting reviewers if appropriate
