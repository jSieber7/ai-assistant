# Development Setup Guide

This guide will help you set up the AI Assistant project for development and contribute to the project.

## Prerequisites

### Required Software
- **Python 3.12** (required - see note below)
- **UV** (Python package manager)
- **Git** (version control)

### Python Version Note
This project requires **Python 3.12** specifically. The `pyproject.toml` enforces this version constraint. Do not use older or newer versions.

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
```

### 2. Set Up Environment
```bash
# Copy environment template
cp .env.template .env

# Create virtual environment with UV
uv venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv sync --dev
```

### 3. Configure Environment Variables
Edit the `.env` file with your settings:

```bash
# OpenRouter API Key (required for LLM functionality)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Server settings
HOST=127.0.0.1
PORT=8000
ENVIRONMENT=development
DEBUG=true
RELOAD=true

# Model settings
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

### 4. Verify Installation
```bash
# Run tests to verify setup
python run_tests.py --unit

# Start the development server
uvicorn app.main:app --reload
```

## Development Workflow

### Code Standards
This project uses several code quality tools:

```bash
# Format code with black
uv run black .

# Check code style with ruff
uv run ruff check .

# Type checking with mypy
uv run mypy app/

# Run all code quality checks
uv run black --check . && uv run ruff check . && uv run mypy app/
```

### Testing
The project includes comprehensive testing:

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration

# Run tests in parallel
python run_tests.py --parallel
```

### Git Workflow
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**:
   ```bash
   python run_tests.py --coverage
   ```

3. **Commit with descriptive messages**:
   ```bash
   git add .
   git commit -m "feat: add new tool integration"
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
ai_assistant/
├── app/                   # Application code
│   ├── api/               # FastAPI routes and endpoints
│   ├── core/              # Core functionality and configuration
│   └── main.py            # FastAPI application entry point
├── tests/                 # Test suite
│   ├── test_main.py       # Unit tests
│   └── test_integration.py # Integration tests
├── docs/                  # Documentation
│   ├── api/               # API documentation
│   ├── architecture/      # System architecture
│   ├── development/       # Development guides
│   └── tools/             # Tool integration docs
├── .github/               # GitHub Actions workflows
├── pyproject.toml         # Project dependencies and configuration
└── README.md              # Project overview
```

## Development Tools

### IDE Configuration

**VS Code Recommended Extensions:**
- Python
- Pylance
- Black Formatter
- Ruff
- GitLens

**.vscode/settings.json:**
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true
}
```

### Debugging

**VS Code Launch Configuration:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Development",
            "type": "python",
            "request": "launch",
            "program": "uvicorn",
            "args": ["app.main:app", "--reload"],
            "console": "integratedTerminal",
            "env": {"PYTHONPATH": "${workspaceFolder}"}
        }
    ]
}
```

## Common Development Tasks

### Adding a New API Endpoint

1. **Create the route** in `app/api/routes.py`:
```python
@router.post("/v1/new-endpoint")
async def new_endpoint(request: NewRequest):
    """New endpoint description"""
    # Implementation here
    return {"message": "Success"}
```

2. **Add Pydantic models** for request/response:
```python
class NewRequest(BaseModel):
    param1: str
    param2: Optional[int] = None

class NewResponse(BaseModel):
    result: str
    status: str
```

3. **Write tests** in `tests/test_main.py`:
```python
def test_new_endpoint(client: TestClient):
    response = client.post("/v1/new-endpoint", json={"param1": "test"})
    assert response.status_code == 200
```

### Adding a New Tool

1. **Create tool documentation** in `docs/tools/`
2. **Implement tool functionality** in a new module
3. **Integrate with LangChain** tool system
4. **Add comprehensive tests**

## Environment-Specific Setup

### Development Environment
- Uses local OpenRouter API calls
- Debug mode enabled
- Auto-reload on code changes
- Detailed logging

### Testing Environment
- Mocked external APIs
- Test-specific environment variables
- Coverage reporting enabled

### Production Environment
- Optimized for performance
- Minimal logging
- Health checks and monitoring
- Security hardening

## Troubleshooting

### Common Issues

**Module Import Errors**
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync --dev
```

**Test Failures**
```bash
# Clear test cache
uv run pytest --cache-clear

# Run with verbose output
python run_tests.py --verbose
```

**API Key Issues**
- Verify `.env` file exists and is properly formatted
- Check that `OPENROUTER_API_KEY` is set
- Ensure the API key has sufficient permissions

### Getting Help

1. **Check existing documentation** in the `docs/` folder
2. **Review GitHub Issues** for similar problems
3. **Create a new issue** with detailed error information
4. **Include logs and environment details**

## Next Steps

After setting up the development environment:

1. **Explore the codebase** and understand the architecture
2. **Run the test suite** to ensure everything works
3. **Try the API endpoints** using the interactive docs at `http://localhost:8000/docs`
4. **Read the architecture documentation** to understand the system design
5. **Join the community** and start contributing!

## Contributing Guidelines

- Follow the existing code style and conventions
- Write tests for new functionality
- Update documentation for changes
- Use descriptive commit messages
- Create focused pull requests

See [Contributing Guide](contributing.md) for detailed contribution guidelines.