# An AI assistant built with a tool-calling ensemble of LLMs

This repository hosts an ai assistant powered by series of LLM Agents. It is designed to provide a human-like assistant with LLM tool calling that can be used through an OpenAI API type interface. This interface will allow this API to be used with many different LLM front ends. Built with LangChain and FastAPI.

## Key Features
* **OpenAI-Compatible API**: Full compatibility with OpenAI API specification
* **OpenRouter Integration**: Support for multiple LLM providers via OpenRouter
* **Streaming Support**: Real-time streaming responses for chat completions
* **Extensible Architecture**: Easy to add new tools and capabilities
* **Comprehensive Testing**: Full test suite with unit and integration tests

## Key Features to be Implemented / Roadmap
* **Extendable Codebase**: Allows the implementation of more tool-calling capabilities.
  * The most important tool will be web search through a SearX instance. In this manner, any chosen AI will have access to the latest information at any given time. 
* **Customizability**: Freedom in LLM model selection. This will allow users to use cloud based models as well as local models.
* **Rapid Experimentation**: One example, small LLM writers alongside larger LLM proof readers to generate messages for the user. Techniques such as these have often been limited to private, hidden code. 
* **Docker Ready**: Eventually the project will be Dockerized to allow a consistent and efficient way to install and host the API on remote computing platforms.


## Quick Start
```bash
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.template .env

uv venv .venv
.venv\Scripts\activate # On Windows
# source .venv/bin/activate On Linux
uv sync --no-dev
uvicorn app.main:app --reload # For the current build in development
```

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
   - Runs tests on Python 3.11 and 3.12
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
     - `Test Python 3.11 on ubuntu-latest`
     - `Test Python 3.12 on ubuntu-latest`
     - `Security Scan`
     - `Lint and Code Quality`
   - ✅ Require branches to be up to date before merging
   - ✅ Require linear history
   - ✅ Include administrators

See [.github/branch-protection.md](.github/branch-protection.md) for detailed configuration instructions.

### Security and API Key Handling

**Important**: Your GitHub Actions workflow is configured to never require real API keys for testing. All tests use mocked dependencies and test environment variables to ensure your OpenRouter API key is never exposed.

- **Mock-Based Testing**: All external API calls are mocked during testing
- **No Real Secrets Required**: The workflow uses test environment variables
- **Secure by Design**: API keys are never hardcoded or committed to version control

For detailed security information, see [.github/SECURITY.md](.github/SECURITY.md).

### Workflow Features

- **UV Integration**: Uses Astral UV for fast dependency resolution
- **Caching**: Automatic caching of UV dependencies for faster builds
- **Multi-version Testing**: Tests against Python 3.11 and 3.12
- **Security Scanning**: Includes bandit security scanner and pip-audit
- **Code Quality**: Runs ruff, black, and mypy checks
- **Coverage Reporting**: Generates coverage reports and uploads to Codecov
- **Artifact Uploads**: Saves test results and security scan reports

See [TESTING.md](TESTING.md) for detailed testing documentation.
