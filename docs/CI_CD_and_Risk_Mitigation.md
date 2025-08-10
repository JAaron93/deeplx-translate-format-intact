# CI/CD and risk mitigation notes for philosophy-enhanced path

This document summarizes how CI is set up to validate the new philosophy-enhanced translation path, along with rollback and monitoring guidance.

- Pytest asyncio support
  - The test runner is configured for asyncio via pytest.ini: `asyncio_mode = auto`.
  - Dev/CI dependencies include pytest-asyncio in requirements-dev.txt.
  - The default CI job installs requirements.txt (if present) and requirements-dev.txt and runs `pytest`.

- Targeted load test (optional)
  - tests/test_translate_content_philosophy_load.py provides a focused load test for the philosophy path.
  - It validates:
    - Preserved ordering of outputs
    - Bounded concurrency via settings.translation_concurrency_limit
    - Reasonable runtime under moderate load
  - The CI workflow defines a separate optional job that runs only the load tests when triggered manually or on a schedule. Load tests are marked with `@pytest.mark.load`.

- Rollback plan
  - Changes are localized to the philosophy path. To revert behavior, restore core/translation_handler.py to the prior revision. This will remove the per-item concurrency path and re-enable previous behavior.
  - No database migrations or external schema changes are required for rollback.

- Monitoring and observability (optional)
  - Consider adding in-process counters and logs:
    - per-item failures (count when an item falls back to original text)
    - skipped empties (count of whitespace-only or empty strings bypassed)
    - max observed concurrency per page
  - These can be surfaced via logging or exposed as metrics (e.g., Prometheus) from the service layer.
  - The load test includes simple assertions that can serve as early signals during CI.

- Additional CI/CD recommendations
  - Fail fast on test flakiness using `--maxfail=1` and consider `pytest-rerunfailures` for non-deterministic external services.
  - Parallelize non-dependent tests with `pytest-xdist -n auto` if the suite becomes slow, avoiding load-marked tests in the parallel run.

