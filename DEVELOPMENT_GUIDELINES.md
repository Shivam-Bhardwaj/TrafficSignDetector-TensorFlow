# Development Guidelines

## Code Standards

### Emoji Policy

**STRICT NO-EMOJI POLICY**: This project maintains a professional, emoji-free codebase.

**Rules:**
- No emojis allowed in Python code, documentation, or configuration files
- Use descriptive text instead of emoji symbols
- Status indicators should use words like PASS/FAIL, SUCCESS/ERROR
- Documentation headers should be clear text without decorative symbols

**Examples of what NOT to do:**
```python
# BAD - Using emojis
print("[checkmark emoji] Test passed!")
print("[warning emoji] Error occurred!")
logger.info("[lock emoji] Security check complete")

# GOOD - Plain text
print("Test passed!")
print("Error occurred!")
logger.info("Security check complete")
```

**Enforcement:**
- Pre-commit hooks will automatically check for emojis
- CI/CD pipeline will fail if emojis are detected
- All pull requests are automatically scanned

### Code Quality Standards

1. **Formatting**: Use Black for Python code formatting
2. **Import Sorting**: Use isort with Black profile
3. **Linting**: Follow Flake8 rules with max line length of 100
4. **Type Checking**: Use MyPy for type hints where applicable
5. **Security**: Bandit security scanning on all commits

### Testing Requirements

1. **Security Tests**: All security features must have comprehensive tests
2. **Coverage**: Maintain high code coverage (>90% target)
3. **Performance**: Include performance benchmarks for critical paths
4. **Documentation**: All test categories must be documented

## Pre-commit Setup

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

This will automatically:
- Check for emojis and fail if found
- Format code with Black
- Sort imports with isort  
- Run Flake8 linting
- Perform security scanning with Bandit

## Manual Checks

Run emoji check manually:
```bash
python check_no_emojis.py
```

Run all quality checks:
```bash
tox -e lint
```

## CI/CD Pipeline

The GitHub Actions pipeline will:
1. **Emoji Check**: Fail immediately if emojis are found
2. **Security Tests**: Run comprehensive security validation
3. **Code Quality**: Verify formatting, linting, and type checking
4. **Performance Tests**: Validate performance benchmarks
5. **Integration Tests**: Full system validation

## Contributing

Before submitting any changes:
1. Run `python check_no_emojis.py` to verify no emojis
2. Run `tox -e lint` for code quality checks
3. Run `python test_runner.py --coverage` for test validation
4. Ensure all CI/CD checks pass

## Violations

If emojis are found:
- **Development**: Pre-commit hooks will prevent the commit
- **CI/CD**: Pipeline will fail with clear error messages
- **Pull Requests**: Will be automatically rejected until fixed

This ensures the codebase maintains professional standards and consistent output formatting.