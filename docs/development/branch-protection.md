# Branch Protection Guide

This guide explains the branch protection rules configured for the AI Assistant project and how they affect the development workflow.

## Overview

Branch protection rules are configured to maintain code quality, prevent accidental changes to critical branches, and ensure that all changes are properly reviewed and tested.

## Protected Branches

### Main Branch (`main`)
- **Purpose**: Production-ready code
- **Protection Level**: Highest
- **Rules**: 
  - Require pull requests for changes
  - Require code reviews
  - Require status checks to pass
  - Require linear history
  - Restrict force pushes

### Development Branch (`develop`)
- **Purpose**: Integration and testing branch
- **Protection Level**: High
- **Rules**:
  - Require pull requests for changes
  - Require code reviews
  - Require status checks to pass
  - Allow squash merging

## Branch Protection Rules

### Required Status Checks

The following status checks must pass before merging:

#### Code Quality Checks
- **`black`**: Code formatting check
- **`ruff`**: Linting and style enforcement
- **`mypy`**: Type checking
- **`bandit`**: Security scanning
- **`pip-audit`**: Dependency vulnerability checking

#### Testing Checks
- **`pytest`**: Unit and integration tests
- **`coverage`**: Test coverage threshold (minimum 80%)
- **`integration-tests`**: End-to-end integration tests

#### Build Checks
- **`build`**: Package build verification
- **`docs-build`**: Documentation build verification

### Required Reviews

#### Review Requirements
- At least **1 approved review** from code owners
- No changes requested reviews blocking merge
- Reviews from code owners for specific file changes

#### Code Owners
The `.github/CODEOWNERS` file defines code ownership:
```
# Core application code
/app/ @jsieber7 @core-maintainers

# Documentation
/docs/ @jsieber7 @documentation-maintainers

# GitHub Actions workflows
/.github/workflows/ @jsieber7 @devops-maintainers
```

### Merge Restrictions

#### Merge Methods
- **Squash and Merge**: Preferred for feature branches
- **Rebase and Merge**: Allowed for maintainers
- **Merge Commit**: Restricted to specific circumstances

#### Commit Requirements
- **Signed commits**: Recommended but not required
- **Conventional commits**: Encouraged for better changelog generation
- **Descriptive messages**: Required for all commits

## Development Workflow with Branch Protection

### Standard Feature Development

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Test**:
   ```bash
   # Run tests locally
   python run_tests.py --coverage
   
   # Format code
   uv run black .
   uv run ruff check .
   ```

3. **Push and Create PR**:
   ```bash
   git push origin feature/your-feature-name
   # Then create PR on GitHub
   ```

4. **Address Review Feedback**:
   - Make requested changes
   - Push updates to the same branch
   - Re-request review when ready

5. **Wait for Checks**:
   - GitHub Actions will run automatically
   - All checks must pass (green ✓)
   - Fix any failing checks

6. **Merge**:
   - Once approved and checks pass
   - Use "Squash and Merge" for feature branches
   - Delete the feature branch after merge

### Hotfix Workflow

For critical bug fixes:

1. **Create Hotfix Branch**:
   ```bash
   git checkout -b hotfix/issue-description main
   ```

2. **Follow Standard Process**:
   - Same testing and review requirements
   - Expedited review process for critical fixes

3. **Merge to Main and Develop**:
   - Merge hotfix to `main`
   - Then merge `main` to `develop` to sync

### Release Process

1. **Create Release Branch**:
   ```bash
   git checkout -b release/v1.2.0 develop
   ```

2. **Final Testing**:
   - Run extended test suite
   - Update version in `pyproject.toml`
   - Update changelog

3. **Merge to Main**:
   - PR from `release/*` to `main`
   - Tag release after merge

4. **Sync to Develop**:
   - Merge `main` back to `develop`

## Bypassing Branch Protection

### When Bypass is Allowed

**Never for external contributors**
**Rarely for maintainers** in specific circumstances:

- **Emergency security fixes**
- **CI/CD pipeline failures**
- **Infrastructure emergencies**

### Bypass Procedure

1. **Get approval** from project lead
2. **Document the reason** in the commit message
3. **Notify the team** about the bypass
4. **Follow up** with proper PR and review post-emergency

## Common Issues and Solutions

### Failing Status Checks

#### Code Formatting Issues
```bash
# Fix formatting
uv run black .

# Fix linting
uv run ruff check --fix
```

#### Test Failures
- Run tests locally to reproduce: `python run_tests.py`
- Check test logs for specific failure details
- Update tests if functionality changed

#### Type Checking Errors
- Fix type annotations
- Use `# type: ignore` sparingly with explanation
- Update type stubs if needed

### Review Stuck

#### No Reviewers Available
- Ping the team in PR comments
- Use `@` mentions for specific code owners
- Consider adding more reviewers to CODEOWNERS

#### Review Requests Changes
- Address all review comments
- Request re-review when changes are made
- Discuss alternative approaches if needed

### Merge Conflicts

#### Resolving Conflicts
```bash
# Update your branch
git fetch origin
git rebase origin/main  # or origin/develop

# Resolve conflicts
# Then continue rebase
git add .
git rebase --continue
```

#### Preventing Conflicts
- Keep branches short-lived
- Regularly sync with base branch
- Communicate with team about overlapping changes

## Best Practices

### Branch Naming
- Use descriptive names: `feature/user-auth`, `fix/api-timeout`
- Follow convention: `type/description`
- Avoid generic names: `update`, `fix`, `patch`

### Commit Messages
- Use conventional commit format
- Reference issue numbers: `Closes #123`
- Be descriptive but concise

### PR Management
- Keep PRs focused and small
- Use draft PRs for work in progress
- Request reviews when ready
- Respond promptly to review feedback

## Monitoring and Enforcement

### Compliance Monitoring
- Regular audits of merge history
- Review of bypass instances
- Team training on branch protection

### Violation Handling
- Education for first-time violations
- Escalation for repeated violations
- Temporary access restrictions if needed

## Tools and Automation

### GitHub Actions Integration
The project uses GitHub Actions to automate:
- Code quality checks
- Testing and coverage
- Security scanning
- Documentation building

### Pre-commit Hooks
Local pre-commit hooks can prevent common issues:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/charliermarsh/ruff
    rev: v0.0.260
    hooks:
      - id: ruff
```

### IDE Integration
Configure your IDE to help with compliance:
- Auto-format on save
- Linting in real-time
- Type checking integration

## Training and Resources

### Learning Resources
- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Code Review Best Practices](https://google.github.io/eng-practices/review/)

### Team Training
- Regular workshops on Git workflows
- Code review practice sessions
- Branch protection rule discussions

## Troubleshooting

### Common Error Messages

#### "Required status check expected"
- Wait for all checks to complete
- Check if any checks are stuck
- Re-run failed checks if appropriate

#### "Review required"
- Ensure at least one approved review
- Check if reviewer has required permissions
- Request review from code owners

#### "Merge conflict"
- Resolve conflicts locally
- Push resolved changes
- Request re-review if needed

### Getting Help

#### Internal Support
- Contact project maintainers
- Use team communication channels
- Reference this documentation

#### External Resources
- GitHub documentation
- Community forums
- Professional training resources

## Continuous Improvement

### Feedback Collection
- Regular team feedback on branch protection
- Survey developers about pain points
- Monitor metrics on PR cycle time

### Rule Updates
- Review and update rules quarterly
- Adapt to team size and project maturity
- Balance protection with developer productivity

### Process Refinement
- Streamline review processes
- Automate repetitive tasks
- Improve documentation and training

## Related Documentation

- [Contributing Guide](contributing.md)
- [Development Setup](setup.md)
- [GitHub Actions Workflows](https://github.com/jSieber7/ai-assistant/tree/main/.github/workflows/)
- [Code Owners File](https://github.com/jSieber7/ai-assistant/blob/main/.github/CODEOWNERS)

This branch protection guide helps maintain code quality while enabling efficient collaboration. Follow these guidelines to ensure smooth development workflows.