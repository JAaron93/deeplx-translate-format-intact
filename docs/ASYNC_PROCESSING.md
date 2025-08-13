# Async Processing Architecture

This document outlines the asynchronous processing path used by the project.

## Concurrency Model

- CPU-bound work: ProcessPoolExecutor
  - PDF to image conversion
  - Image optimization for OCR
- IO-bound work: asyncio tasks and TaskGroup (Python 3.11+)
  - OCR service requests (rate-limited)
  - Layout-aware translation with batching
- Request-level concurrency: asyncio.Semaphore
- Rate limiting: token-bucket aligned with the OCR provider

## Key Components

- `services/async_document_processor.AsyncDocumentProcessor`
  - Orchestrates the async pipeline
  - Methods:
    - `process_document(request, on_progress)`
- Token Bucket: `_TokenBucket` (simple refill each time acquire is called)

## Configuration

- `max_concurrent_requests` (default 4): maximum concurrent documents
- `translation_batch_size` (default 100): max blocks per translation call
- `translation_concurrency` (default 4): concurrent translation batches
- `ocr_rate_capacity` (default 2): token bucket capacity for OCR
- `ocr_rate_per_sec` (default 1.0): token refill rate per second

## Progress Events

- `validated`
- `converted`
- `ocr`
- `translated`
- `reconstructed`

## Testing

- `tests/test_async_document_processor.py` covers:
  - Translation batching (calls split into predictable batch sizes)
  - Token bucket invocation (acquire called once per document)
  - Request-level concurrency cap (no more than N active operations)

## Notes

- The async orchestrator is designed as a drop-in alternative to the sync path
  for services that benefit from concurrency. It avoids CPU starvation by
  offloading heavy work to a small process pool while keeping IO operations on
  the event loop.
