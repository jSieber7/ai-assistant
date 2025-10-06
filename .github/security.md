# Security Guide: API Key Management for GitHub Actions

This guide explains how to securely handle API keys and secrets in your GitHub Actions workflows.

## Current Security Setup

Your GitHub Actions workflow is configured to **never require real API keys** for testing. All tests use mocked dependencies and test environment variables.

## How Testing Works Without Real API Keys

### Mock-Based Testing Strategy
Your test suite uses comprehensive mocking to avoid external API calls:

1. **LLM Mocking**: The [`tests/conftest.py`](tests/conftest.py) file provides `mock_llm` fixtures that simulate OpenRouter API responses
2. **Environment Mocking**: Test environment variables are set within the workflow, not from secrets
3. **Dependency Injection**: All external dependencies are mocked during testing

### GitHub Actions Configuration
The workflow sets test environment variables:

```yaml
- name: Set test environment variables
  run: |
    echo "OPENROUTER_API_KEY=test-key-ci-123" >> $GITHUB_ENV
    echo "DEFAULT_MODEL=test-model" >> $GITHUB_ENV
    echo "ENVIRONMENT=testing" >> $GITHUB_ENV
```

## When You Need Real API Keys

### For Local Development
Create a `.env` file (never commit this to version control):

```bash
# Copy the template
cp .env.template .env

# Edit with your real API keys
# .env is in .gitignore, so it won't be committed
```

### For Production Deployments
If you need to deploy with real API keys:

1. **Use GitHub Secrets** for sensitive values
2. **Never hardcode API keys** in your code or configuration files
3. **Use environment-specific configuration**

## GitHub Secrets Setup (Optional)

If you ever need real API keys in CI/CD (for deployment, not testing):

### 1. Add Secrets to GitHub
- Go to your repository on GitHub
- Navigate to **Settings** → **Secrets and variables** → **Actions**
- Click **New repository secret**
- Add secrets with names like `OPENROUTER_API_KEY`

### 2. Use Secrets Securely in Workflows
```yaml
- name: Deploy to production
  if: github.ref == 'refs/heads/main'
  run: |
    # Secrets are automatically available as environment variables
    echo "Deploying with API key: ${{ secrets.OPENROUTER_API_KEY }}"
```

## Security Best Practices

### ✅ DO:
- Use mocked dependencies for CI/CD testing
- Store real secrets in GitHub Secrets
- Use `.env` files for local development (add to `.gitignore`)
- Regularly rotate API keys
- Monitor API key usage

### ❌ DON'T:
- Commit API keys to version control
- Hardcode secrets in your application
- Use real API keys for automated testing
- Share API keys in logs or error messages

## Testing Philosophy

Your current test strategy follows security best practices:

1. **Isolation**: Tests don't depend on external services
2. **Speed**: Mocked tests run faster than network calls
3. **Reliability**: Tests aren't affected by API rate limits or downtime
4. **Security**: No risk of API key exposure

## Emergency Procedures

### If an API Key is Exposed:
1. Immediately revoke the exposed key
2. Generate a new key from your API provider
3. Update all environments where the key was used
4. Investigate how the exposure occurred

### If You Suspect a Security Breach:
1. Rotate all API keys and secrets
2. Review access logs
3. Check for unauthorized usage
4. Update your security procedures

## Additional Security Measures

### Code Scanning
The workflow includes security scanning with:
- **Bandit**: Python security linter
- **pip-audit**: Dependency vulnerability scanner

### Branch Protection
- Prevents direct pushes to main branch
- Requires code review for all changes
- Ensures tests pass before merging

## Questions?

If you have security concerns or need to modify the security setup:
1. Review this document
2. Check the GitHub Actions workflows
3. Consult the test mocking strategy
4. Consider the security implications of any changes