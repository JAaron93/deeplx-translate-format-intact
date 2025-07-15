# Testing Configuration

## Memory Limit Configuration

The integration test `test_11_memory_and_performance_monitoring` includes a configurable memory limit to prevent flaky test failures across different environments.

### Environment Variable

Set the `TEST_MEMORY_LIMIT_MB` environment variable to configure the memory limit:

```bash
# Set a custom memory limit of 750MB
export TEST_MEMORY_LIMIT_MB=750
pytest tests/test_complete_integration.py::TestCompletePhilosophyEnhancedIntegration::test_11_memory_and_performance_monitoring -v

# Use default limit (1000MB)
pytest tests/test_complete_integration.py::TestCompletePhilosophyEnhancedIntegration::test_11_memory_and_performance_monitoring -v
```

### Default Configuration

- **Default Memory Limit**: 1000MB (1GB)
- **Previous Hardcoded Limit**: 500MB (removed for better reliability)

### Usage Examples

```bash
# Strict memory testing
TEST_MEMORY_LIMIT_MB=500 pytest tests/test_complete_integration.py

# Generous memory testing for resource-constrained environments
TEST_MEMORY_LIMIT_MB=2000 pytest tests/test_complete_integration.py

# CI/CD environment with limited resources
TEST_MEMORY_LIMIT_MB=1500 pytest tests/test_complete_integration.py
```

This configuration helps prevent flaky tests due to varying memory usage across different environments while maintaining meaningful performance monitoring.
