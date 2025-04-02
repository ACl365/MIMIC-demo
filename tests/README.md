# MIMIC Demo Project Tests

This directory contains tests for the MIMIC demo project.

## Test Structure

The tests are organised as follows:

- `unit/`: Unit tests for individual components
  - `data/`: Tests for data processing modules
  - `features/`: Tests for feature extraction modules
  - `utils/`: Tests for utility modules

## Running Tests

You can run the tests using the provided `run_tests.py` script:

```bash
# Run all tests
python run_tests.py

# Run tests with coverage report
python run_tests.py --coverage

# Run specific test file or directory
python run_tests.py tests/unit/features/test_build_features.py
```

Alternatively, you can use pytest directly:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=term --cov-report=html

# Run specific test file or directory
pytest tests/unit/features/test_build_features.py
```

## Test Configuration

The test configuration is defined in `pytest.ini`. The configuration includes:

- Test discovery paths
- Test file naming patterns
- Coverage reporting options

## Writing Tests

When writing tests, follow these guidelines:

1. Test files should be named `test_*.py`
2. Test classes should be named `Test*`
3. Test methods should be named `test_*`
4. Use appropriate assertions for the type of test
5. Use mocks and fixtures to isolate the code being tested
6. Include docstrings to describe the purpose of each test

Example:

```python
def test_function_name():
    """
    Test description.
    """
    # Arrange
    input_data = ...

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_result
```

## Coverage Reports

Coverage reports are generated in HTML format in the `htmlcov/` directory. Open `htmlcov/index.html` in a web browser to view the coverage report.
