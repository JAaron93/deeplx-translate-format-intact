# Pre-commit Workflow Guide

## Overview

This project uses a CI/CD-friendly pre-commit configuration that balances code quality with development productivity. The configuration automatically handles formatting. Note: if a hook modifies files (e.g., Black, trailing whitespace fixes), your commit will be aborted; re-stage the changes and commit again. Only critical issues are intended to fail commits otherwise.
## What Gets Auto-Fixed

Auto-fixed issues (the commit that triggers a change will abort; re-stage and commit again):

- **Trailing whitespace** — `trailing-whitespace` hook
- **Line endings (LF)** — `mixed-line-ending` (configured to `lf`)
- **Python code formatting** — `black`
- **End-of-file newlines** — `end-of-file-fixer`
- **Many style violations** — `ruff --fix` (for fixable rules)

Only critical issues will prevent commits:

- **Syntax errors** (E9)
- **High-severity security vulnerabilities**
- **Invalid Python AST**
- **Debug statements in production code**

## Development Workflow

### Normal Development
```bash
# Install pre-commit hooks
pre-commit install

# Keep hooks up to date (optional, recommended)
pre-commit autoupdate

# Normal commits: if hooks modify files (formatting, whitespace), re-stage and commit again.
git commit -m "Your message"
```

### Full Code Quality Checks
```bash
# Run comprehensive linting with manual-stage hooks (only runs hooks that declare this stage)
pre-commit run --all-files --hook-stage manual

# Config-agnostic alternative: run all hooks across all stages
pre-commit run --all-files

# Run specific tools
pre-commit run ruff --all-files --hook-stage manual  # If ruff has manual stage
pre-commit run bandit --all-files                    # Standard stage
pre-commit run black --all-files                     # Formatting

# Black check-only mode (as used in CI - doesn't modify files)
black --check --diff .                               # Direct Black invocation
```

**When to use each approach:**
- `--hook-stage manual`: Use when you want comprehensive linting from hooks specifically configured for manual review
- `--all-files` (no stage): Use as fallback that works with any pre-commit config, runs hooks at their default stages
- Direct tool invocation: Use for CI checks or when you need specific tool options not available in pre-commit config

### Troubleshooting

If a commit is blocked:
1. Check which hook failed
If a commit is blocked:
1. Check which hook failed
2. Run `pre-commit run --all-files` to see detailed output
3. If files were modified by hooks, `git add -A`
4. Fix any remaining reported issues
5. Commit again

## CI/CD Behavior

In CI/CD environments:
- ✅ Commits proceed with auto-fixed formatting
- ✅ Only critical issues cause build failures
- ✅ Security scans run but only high-severity issues fail builds
- ✅ Full linting available via manual stage for comprehensive checks

## Configuration Details

- **Ruff**: Only blocks on critical syntax errors (E9) during normal commits
- **Bandit**: Only blocks on high-severity security vulnerabilities
- **Black**: Auto-formats code and never blocks commits
- **Trailing whitespace**: Auto-removed and never blocks commits
- **Most linting violations**: Auto-fixed or ignored in CI environments

## Key Benefits for CI/CD

- ✅ **Faster commits** - Style issues don't block development
- ✅ **Auto-formatting** - Code is automatically formatted on commit
- ✅ **Security focus** - Only critical security issues block commits
- ✅ **Syntax safety** - Critical syntax errors still prevent broken code
- ✅ **Development productivity** - Developers can focus on features, not formatting
- ℹ️ Note: when a hook auto-fixes files, the initial commit aborts by design; re-stage and commit again.
