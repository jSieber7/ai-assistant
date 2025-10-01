### Security and API Key Handling

**Important**: Your GitHub Actions workflow is configured to never require real API keys for testing. All tests use mocked dependencies and test environment variables to ensure your OpenRouter API key is never exposed.

- **Mock-Based Testing**: All external API calls are mocked during testing
- **No Real Secrets Required**: The workflow uses test environment variables
- **Secure by Design**: API keys are never hardcoded or committed to version control

For detailed security information, see [SECURITY.md](https://github.com/jSieber7/ai-assistant/blob/main/.github/SECURITY.md).

### Workflow Features

- **UV Integration**: Uses Astral UV for fast dependency resolution
- **Caching**: Automatic caching of UV dependencies for faster builds
- **Multi-version Testing**: Tests against Python 3.11 and 3.12
- **Security Scanning**: Includes bandit security scanner and pip-audit
- **Code Quality**: Runs ruff, black, and mypy checks
- **Coverage Reporting**: Generates coverage reports and uploads to Codecov
- **Artifact Uploads**: Saves test results and security scan reports
