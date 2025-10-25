# Test Suite Documentation

This document provides comprehensive information about the test structure, how to run tests, and best practices for testing the AI Assistant application.

## Table of Contents

1. [Test Structure Overview](#test-structure-overview)
2. [Test Types](#test-types)
3. [Running Tests](#running-tests)
4. [Test Configuration](#test-configuration)
5. [Test Utilities](#test-utilities)
6. [Writing New Tests](#writing-new-tests)
7. [Debugging Tests](#debugging-tests)
8. [Continuous Integration](#continuous-integration)
9. [Performance Testing](#performance-testing)
10. [Troubleshooting](#troubleshooting)

## Test Structure Overview

The test suite is organized into the following structure:

```
tests/
├── README.md                           # This documentation
├── conftest.py                         # Global pytest configuration and fixtures
├── pytest.ini                          # Pytest configuration file
├── test_utils.py                        # Utility functions for testing
├── fixtures/                            # Test data and mock objects
│   ├── __init__.py
│   ├── sample_data.py                   # Sample data for tests
│   ├── mock_responses.py                # Mock response classes
│   └── test_configs.py                 # Test configuration objects
├── unit/                               # Unit tests
│   ├── app/
│   │   ├── api/                        # API route tests
│   │   │   ├── test_config.py
│   │   │   ├── test_context_sharing.py
│   │   │   ├── test_dynamic_executor.py
│   │   │   ├── test_langchain_routes.py
│   │   │   ├── test_llm_manager.py
│   │   │   ├── test_main.py
│   │   │   ├── test_routes.py
│   │   │   ├── test_secure_settings.py
│   │   │   └── test_tool_routes.py
│   │   ├── core/                       # Core component tests
│   │   │   ├── agents/                  # Agent system tests
│   │   │   ├── caching/                 # Caching system tests
│   │   │   ├── langchain/               # LangChain integration tests
│   │   │   ├── services/               # Service layer tests
│   │   │   ├── storage/                # Storage layer tests
│   │   │   ├── tools/                  # Tool system tests
│   │   │   └── ui/                     # UI component tests
│   │   └── test_config.py              # Configuration tests
├── integration/                         # Integration tests
│   ├── app/
│   │   ├── api/                        # API integration tests
│   │   ├── core/                       # Core component integration tests
│   │   │   ├── caching/                 # Caching integration tests
│   │   │   └── langchain/               # LangChain integration tests
│   │   └── agents/                     # Agent integration tests
│   └── system/                         # System-level integration tests
└── performance/                         # Performance and load tests
    ├── test_load_testing.py              # Load testing suite
    └── benchmarks/                      # Performance benchmarks
```

## Test Types

### Unit Tests
Unit tests test individual components in isolation. They are fast, focused, and don't depend on external services.

**Location**: `tests/unit/`

**Characteristics**:
- Test single functions, methods, or classes
- Mock external dependencies
- Fast execution (milliseconds)
- High coverage of edge cases

### Integration Tests
Integration tests test how multiple components work together. They verify that components integrate correctly.

**Location**: `tests/integration/`

**Characteristics**:
- Test component interactions
- Use real dependencies where possible
- Slower execution (seconds)
- Focus on integration points

### System Tests
System tests test the entire application stack from end to end. They simulate real user scenarios.

**Location**: `tests/system/`

**Characteristics**:
- Test complete workflows
- Use full application setup
- Slower execution (seconds to minutes)
- Focus on user scenarios

### Performance Tests
Performance tests test application performance under various load conditions.

**Location**: `tests/performance/`

**Characteristics**:
- Test response times, throughput, resource usage
- Generate load and measure metrics
- Variable execution time
- Focus on performance characteristics

## Running Tests

### Prerequisites

Before running tests, ensure you have the following installed:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install development dependencies
pip install -e .
```

### Basic Test Commands

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run tests with coverage and show missing lines
pytest --cov=app --cov-report=term-missing
```

### Running Specific Test Types

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only system tests
pytest tests/system/

# Run only performance tests
pytest tests/performance/
```

### Running Specific Test Files

```bash
# Run specific test file
pytest tests/unit/app/api/test_routes.py

# Run specific test class
pytest tests/unit/app/api/test_routes.py::TestRoutes

# Run specific test method
pytest tests/unit/app/api/test_routes.py::TestRoutes::test_health_check
```

### Running Tests by Marker

Tests are marked with pytest markers for better organization:

```bash
# Run only fast tests
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance
```

### Running Tests in Parallel

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run tests in parallel with specific number of workers
pytest -n 4
```

### Running Tests with Docker

```bash
# Build test environment
docker build -f docker/app/Dockerfile.test -t ai-assistant-test .

# Run tests in Docker
docker run --rm ai-assistant-test pytest

# Run tests with Docker Compose
docker-compose -f docker/docker-compose.test.yml up --abort-on-container-exit --build
```

## Test Configuration

### Pytest Configuration

The test suite is configured via `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=app
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    system: System tests
    performance: Performance tests
    slow: Slow running tests
    external: Tests that require external services
```

### Environment Variables

Tests can be configured with environment variables:

```bash
# Set test environment
export TEST_ENVIRONMENT=test

# Set test database URL
export TEST_DATABASE_URL=postgresql://test:test@localhost:5432/test_db

# Set test Redis URL
export TEST_REDIS_URL=redis://localhost:6379/1

# Set test API keys
export TEST_OPENAI_API_KEY=test-key
export TEST_FIRECRAWL_API_KEY=test-key
```

### Test Configuration Files

Test configurations are located in `tests/fixtures/test_configs.py`:

```python
# Sample test configuration
test_settings = Settings(
    environment="test",
    debug=True,
    api_host="localhost",
    api_port=8000,
    redis_url="redis://localhost:6379/0",
    cache_enabled=True,
    cache_ttl=300
)
```

## Test Utilities

### Common Fixtures

Global fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def client():
    """Create a test client for FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_settings():
    """Create mock settings for testing"""
    return Settings(environment="test", debug=True)

@pytest.fixture
def mock_cache():
    """Create a mock cache for testing"""
    cache = MemoryCache()
    cache.configure(ttl=300, max_size=1000)
    return cache
```

### Test Data

Sample test data is provided in `tests/fixtures/sample_data.py`:

```python
# Sample user data
SAMPLE_USER = {
    "id": "test-user-123",
    "name": "Test User",
    "email": "test@example.com"
}

# Sample chat message
SAMPLE_MESSAGE = {
    "role": "user",
    "content": "Hello, AI assistant!"
}
```

### Mock Responses

Mock response classes are defined in `tests/fixtures/mock_responses.py`:

```python
class MockLLMResponse:
    """Mock LLM response for testing"""
    
    def __init__(self, content="Test response"):
        self.content = content
        self.usage = {"total_tokens": 50}
```

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestNewFeature:
    """Test new feature functionality"""
    
    @pytest.fixture
    def new_feature_instance(self):
        """Create an instance of the new feature"""
        return NewFeature()
    
    def test_basic_functionality(self, new_feature_instance):
        """Test basic functionality"""
        # Arrange
        input_data = {"key": "value"}
        
        # Act
        result = new_feature_instance.process(input_data)
        
        # Assert
        assert result.success is True
        assert result.data["processed_key"] == "value"
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, new_feature_instance):
        """Test async functionality"""
        # Arrange
        input_data = {"key": "value"}
        
        # Act
        result = await new_feature_instance.process_async(input_data)
        
        # Assert
        assert result.success is True
    
    def test_error_handling(self, new_feature_instance):
        """Test error handling"""
        # Arrange
        invalid_input = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Input cannot be None"):
            new_feature_instance.process(invalid_input)
    
    @patch('app.core.external_service.call_external_api')
    def test_with_external_dependency(self, mock_external, new_feature_instance):
        """Test with external dependency mocked"""
        # Arrange
        mock_external.return_value = {"result": "mocked"}
        
        # Act
        result = new_feature_instance.use_external_service()
        
        # Assert
        assert result.success is True
        mock_external.assert_called_once()
```

### Best Practices

1. **Arrange, Act, Assert Pattern**: Structure tests clearly with setup, execution, and verification
2. **Descriptive Test Names**: Use descriptive names that explain what is being tested
3. **Test One Thing**: Each test should verify one specific behavior
4. **Use Fixtures**: Use fixtures for common setup code
5. **Mock External Dependencies**: Mock external services to ensure test isolation
6. **Test Edge Cases**: Test boundary conditions and error scenarios
7. **Use Assertions**: Use specific assertions with clear error messages
8. **Clean Up**: Ensure tests clean up any resources they create

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with Python debugger
pytest --pdb

# Run with debugger on failure
pytest --pdb -x

# Run with verbose output
pytest -v -s
```

### Debugging Failed Tests

```bash
# Run only failed tests
pytest --lf

# Run tests with output capture disabled
pytest -s

# Run tests with maximum verbosity
pytest -vv
```

### Print Debug Information

```python
def test_with_debug_info():
    """Test with debug information"""
    # Enable debug prints
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Your test code here
    result = some_function()
    
    # Print debug information
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
```

## Continuous Integration

### GitHub Actions

Tests are configured to run on GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

Configure pre-commit hooks to run tests before commits:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Performance Testing

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/

# Run specific performance test
pytest tests/performance/test_load_testing.py::TestLoadTesting::test_concurrent_requests_performance

# Run performance tests with markers
pytest -m performance
```

### Performance Metrics

Performance tests measure:

- **Response Time**: Time to process requests
- **Throughput**: Requests per second
- **Memory Usage**: Memory consumption under load
- **Error Rate**: Percentage of failed requests
- **Resource Utilization**: CPU, disk, network usage

### Load Testing Scenarios

1. **Single Request**: Baseline performance
2. **Concurrent Requests**: Performance under load
3. **Sustained Load**: Performance over time
4. **Stress Testing**: Behavior beyond limits
5. **Performance Regression**: Detect performance degradation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Connection**: Check test database configuration
3. **External Services**: Mock external services in tests
4. **Async Tests**: Use `@pytest.mark.asyncio` for async tests
5. **Fixture Errors**: Check fixture definitions and scope

### Test Isolation

Ensure tests are isolated:

```python
@pytest.fixture
def isolated_cache():
    """Create isolated cache for each test"""
    cache = MemoryCache()
    cache.configure(ttl=300, max_size=1000)
    yield cache
    cache.clear()  # Cleanup after test
```

### Time-sensitive Tests

For tests that depend on timing:

```python
@pytest.mark.asyncio
async def test_time_sensitive():
    """Test time-sensitive functionality"""
    # Use frozen time for deterministic tests
    with freeze_time("2023-01-01"):
        result = await some_async_function()
        assert result.timestamp == "2023-01-01"
```

### Resource Cleanup

Always clean up resources:

```python
@pytest.fixture
async def temporary_resource():
    """Create temporary resource with cleanup"""
    resource = await create_resource()
    try:
        yield resource
    finally:
        await resource.cleanup()
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Async Testing](https://pytest-asyncio.readthedocs.io/)
- [Test Coverage](https://coverage.readthedocs.io/)

## Contributing to Tests

When contributing new tests:

1. Follow the existing naming conventions
2. Add appropriate markers
3. Include both positive and negative test cases
4. Mock external dependencies
5. Ensure tests are fast and reliable
6. Add documentation for complex test scenarios

For questions about testing, please refer to the development team or create an issue in the project repository.