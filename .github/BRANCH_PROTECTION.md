# Branch Protection Configuration

*This document outlines the recommended branch protection rules for this repository and forks of this repository. These settings help maintain code quality and prevent accidental pushes to protected branches.*

## Recommended Settings

### Main Branch Protection
- **Require pull request reviews before merging**: At least 1 approval
- **Dismiss stale pull request approvals when new commits are pushed**: Enabled
- **Require status checks to pass before merging**: 
  - `Test Python 3.11 on ubuntu-latest`
  - `Test Python 3.12 on ubuntu-latest`
  - `Security Scan`
  - `Lint and Code Quality`
- **Require branches to be up to date before merging**: Enabled
- **Require linear history**: Enabled
- **Include administrators**: Enabled (optional)

### Development Branch Protection
- **Require pull request reviews before merging**: At least 1 approval
- **Require status checks to pass before merging**: Same as main branch
- **Allow force pushes**: Disabled

## GitHub Actions Workflow Protection

The following workflows provide push protection:

1. **python-tests.yml**: Runs tests on push and pull requests
2. **branch-protection.yml**: Monitors direct pushes to main branch

## Manual Setup Instructions

### Via GitHub Web Interface
1. Go to repository **Settings**
2. Click on **Branches** in the left sidebar
3. Click **Add branch protection rule**
4. Set the branch name pattern to `main`
5. Configure the following:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date before merging
   - ✅ Require linear history
   - ✅ Include administrators

### Via GitHub CLI (if available)
```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --input - << EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Test Python 3.11 on ubuntu-latest",
      "Test Python 3.12 on ubuntu-latest",
      "Security Scan",
      "Lint and Code Quality"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null
}
EOF
```

## Workflow Dependencies

The branch protection relies on the following workflows passing:
- **Python CI with UV** (`python-tests.yml`)
- **Branch Protection Enforcement** (`branch-protection.yml`)

## Troubleshooting

### If workflows are not showing up in branch protection:
1. Ensure workflows are in the `.github/workflows/` directory
2. Push a commit to trigger the workflows at least once
3. Wait for the workflows to complete successfully

### If direct pushes are still allowed:
1. Check that branch protection rules are properly configured
2. Verify that the required status checks are correctly specified
3. Ensure that the "Include administrators" setting is configured as desired

## Security Considerations

- **Never disable branch protection for convenience**
- **Use feature branches for all changes**
- **Require code review for all changes**
- **Monitor direct pushes to protected branches**

## Emergency Override

In case of emergency, repository administrators can temporarily disable branch protection via repository settings, but this should be re-enabled immediately after the emergency is resolved.