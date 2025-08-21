# Pre-commit Workflow Guide

## Overview

This project uses a CI/CD-friendly pre-commit configuration that balances code quality with development productivity. The configuration automatically handles formatting and only blocks commits on critical issues.

## What Gets Auto-Fixed

The following issues are automatically resolved and never block your commits:

- **Trailing whitespace** - Automatically removed from all files
- **Line endings** - Fixed to use consistent LF endings
- **Python code formatting** - Black formats your code automatically
- **End-of-file newlines** - Added automatically where needed
- **Most style violations** - Auto-fixed by Ruff and Black

## What Blocks Commits

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

# Normal commits will only be blocked by critical issues
git commit -m "Your message"
```

### Full Code Quality Checks
```bash
# Run all linting rules (for thorough code review)
pre-commit run --all-files --hook-stage manual

# Run specific linters
pre-commit run ruff --all-files --hook-stage manual
pre-commit run bandit --all-files
```

### Troubleshooting

If a commit is blocked:
1. Check which hook failed
2. Run `pre-commit run --all-files` to see detailed output
3. Fix the reported issues
4. Commit again

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
