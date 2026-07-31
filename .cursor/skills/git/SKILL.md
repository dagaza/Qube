---
name: git
description: Git command reference for multi-repo workspace operations; use when checking status, branches, history, worktrees, or performing git operations.
---

# Git Commands

```bash
# Status
git -C <repo> status --short
git -C <repo> status --porcelain # Empty = clean

# Branches
git -C <repo> branch -v
git -C <repo> branch --show-current

# History
git -C <repo> log --oneline -20
git -C <repo> diff <A>..<B>
git -C <repo> diff <A>...<B> --name-only

# Worktrees & submodules
git -C <repo> worktree list
git -C <repo> submodule update --init --recursive

# Write ops (confirm first, may fail -> provide manual command)
# Never push to main/master; push feature branches only
git -C <repo> checkout -b <branch>
git -C <repo> push -u origin <feature-branch>
```
