## Testing

The project includes a comprehensive test suite to ensure code quality and reliability.

### Running Tests

**Quick Test (Recommended):**
```bash
python run_tests.py
```

**Run with Coverage:**
```bash
python run_tests.py --coverage
```

**Run Only Unit Tests:**
```bash
python run_tests.py --unit
```

**Run Only Integration Tests:**
```bash
python run_tests.py --integration
```

**Run Tests in Parallel:**
```bash
python run_tests.py --parallel
```

### Manual Testing

For manual testing, you can use the provided test script:
```bash
./test_branch.sh
```

### Test Structure
- **Unit Tests**: Mock external dependencies for fast execution
- **Integration Tests**: Test the full application flow
- **Error Handling**: Comprehensive error scenario testing
- **Performance**: Response time and resource usage testing

## GitHub Actions & CI/CD

This project includes comprehensive GitHub Actions workflows for continuous integration and deployment with push protection.

### Available Workflows

1. **Python CI with UV** (`.github/workflows/python-tests.yml`)
   - Runs tests on Python 3.12
   - Includes security scanning and linting
   - Generates coverage reports
   - Uses UV for fast dependency management

2. **Branch Protection Enforcement** (`.github/workflows/branch-protection.yml`)
   - Monitors direct pushes to protected branches
   - Validates pull request requirements
   - Provides push protection

### Setting Up GitHub Actions

The workflows are automatically triggered on:
- **Push** to `main`, `development`, and `feature/*` branches
- **Pull requests** targeting `main` or `development`
- **Weekly schedule** (Sunday at midnight UTC)

### Branch Protection Setup

To enable full push protection, configure branch protection rules in your repository settings:

1. Go to **Settings** → **Branches**
2. Add branch protection rule for `main` branch
3. Enable the following:
   - ✅ Require pull request reviews before merging
   - ✅ Require approvals (1)
   - ✅ Require status checks to pass:
     - `Test Python 3.12 on ubuntu-latest`
     - `Security Scan`
     - `Lint and Code Quality`
   - ✅ Require branches to be up to date before merging
   - ✅ Require linear history
   - ✅ Include administrators

See [.github/branch-protection.md](.github/branch-protection.md) for detailed configuration instructions.
