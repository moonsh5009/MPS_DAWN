# Git Commit

Create a git commit with the staged changes. Follow these guidelines:

## Commit Message Format
- First line: Brief summary (50 chars or less) in imperative mood
- Blank line
- Detailed description (if needed):
  - What changes were made
  - Why the changes were necessary
  - Any important technical details
- Always end with: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`

## Process
1. Check `git status` to see staged changes
2. Review `git diff --cached` to understand what's being committed
3. Write a meaningful commit message
4. Create the commit
5. Confirm with `git log -1` to verify

## Example Format
```
Add WebGPU initialization and window creation

- Initialize WebGPU instance with Dawn
- Create GLFW window for rendering
- Set up device and adapter selection
- Add error handling for GPU unavailability

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Guidelines
- Use present tense ("Add feature" not "Added feature")
- Be specific about what changed
- Explain why if it's not obvious
- Reference issue numbers if applicable (e.g., "Fixes #123")
