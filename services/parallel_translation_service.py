"""
Parallel translation service for high-performance document processing.

This module implements parallelized translation capabilities using asyncio,
aiohttp, and intelligent rate limiting to efficiently process large documents
while respecting API constraints.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession, ClientTimeout

# Import will be done locally to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class ParallelTranslationConfig:
    """Configuration for parallel translation operations."""

    # Concurrency settings
    max_concurrent_requests: int = 10
    max_requests_per_second: float = 5.0
    batch_size: int = 50

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0

    # Timeout settings
    request_timeout: float = 30.0
    total_timeout: float = 300.0

    # Rate limiting
    rate_limit_window: float = 1.0
    burst_allowance: int = 2

    @classmethod
    def from_config(cls) -> 'ParallelTranslationConfig':
        """Create configuration from environment/config settings."""
        import os
        return cls(
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            max_requests_per_second=float(os.getenv("MAX_REQUESTS_PER_SECOND", "5.0")),
            batch_size=int(os.getenv("TRANSLATION_BATCH_SIZE", "50")),
            max_retries=int(os.getenv("TRANSLATION_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("TRANSLATION_RETRY_DELAY", "1.0")),
            backoff_multiplier=float(os.getenv("TRANSLATION_BACKOFF_MULTIPLIER", "2.0")),
            request_timeout=float(os.getenv("TRANSLATION_REQUEST_TIMEOUT", "30.0")),
            total_timeout=float(os.getenv("TRANSLATION_TOTAL_TIMEOUT", "300.0")),
            rate_limit_window=float(os.getenv("TRANSLATION_RATE_LIMIT_WINDOW", "1.0")),
            burst_allowance=int(os.getenv("TRANSLATION_BURST_ALLOWANCE", "2")),
        )


@dataclass
class TranslationTask:
    """Individual translation task."""

    text: str
    source_lang: str
    target_lang: str
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationResult:
    """Result of a translation task."""

    task_id: str
    original_text: str
    translated_text: str
    success: bool
    error: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress tracking for batch operations."""

    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100.0

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining time in seconds."""
        if self.completed_tasks == 0 or self.elapsed_time < 0.001:
            return 0.0

        rate = self.completed_tasks / self.elapsed_time
        remaining_tasks = self.total_tasks - self.completed_tasks
        return remaining_tasks / rate if rate > 0 else 0.0


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, max_requests_per_second: float, burst_allowance: int = 2):
        self.max_requests_per_second = max_requests_per_second
        self.burst_allowance = burst_allowance
        self.tokens = max_requests_per_second + burst_allowance
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.max_requests_per_second + self.burst_allowance,
                self.tokens + elapsed * self.max_requests_per_second
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.max_requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ParallelLingoTranslator:
    """High-performance parallel translator using Lingo API."""

    def __init__(self, api_key: str, config: Optional[ParallelTranslationConfig] = None, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("LINGO_API_KEY is required")

        self.api_key = api_key
        self.base_url = base_url or "https://api.lingo.dev/v1/translate"
        self.config = config or ParallelTranslationConfig.from_config()

        # Rate limiting
        self.rate_limiter = RateLimiter(
            self.config.max_requests_per_second,
            self.config.burst_allowance
        )

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Session will be created when needed
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> ClientSession:
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(
                total=self.config.total_timeout,
                sock_read=self.config.request_timeout
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "PDF-Translator-Parallel/2.0",
            }

            self._session = ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(
                    limit=self.config.max_concurrent_requests * 2,
                    limit_per_host=self.config.max_concurrent_requests,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
            )

        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _translate_single_with_retry(self, task: TranslationTask) -> TranslationResult:
        """Translate a single text with retry logic."""
        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Concurrency control
                async with self.semaphore:
                    session = await self._ensure_session()

                    payload = {
                        "text": task.text,
                        "source": task.source_lang.lower(),
                        "target": task.target_lang.lower(),
                    }

                    async with session.post(self.base_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            translated_text = self._extract_translation(result)

                            return TranslationResult(
                                task_id=task.task_id,
                                original_text=task.text,
                                translated_text=translated_text,
                                success=True,
                                retry_count=attempt,
                                processing_time=time.time() - start_time,
                                metadata=task.metadata
                            )

                        if response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get('Retry-After', 1))
                            await asyncio.sleep(retry_after)
                            last_error = f"Rate limited (429), retry after {retry_after}s"
                            continue

                        # Handle other HTTP errors
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"

                        # Don't retry on client errors (4xx except 429)
                        if 400 <= response.status < 500 and response.status != 429:
                            break

            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except aiohttp.ClientError as e:
                last_error = f"Client error: {e}"
            except Exception as e:  # pylint: disable=broad-except
                last_error = f"Unexpected error: {e}"

            # Wait before retry (exponential backoff)
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.backoff_multiplier ** attempt)
                await asyncio.sleep(delay)

        # All retries failed
        return TranslationResult(
            task_id=task.task_id,
            original_text=task.text,
            translated_text=task.text,  # Return original on failure
            success=False,
            error=last_error,
            retry_count=self.config.max_retries,
            processing_time=time.time() - start_time,
            metadata=task.metadata
        )

    def _extract_translation(self, response_data: Dict[str, Any]) -> str:
        """Extract translation from API response."""
        if "translation" in response_data:
            return response_data["translation"]
        if "text" in response_data:
            return response_data["text"]

        logger.warning("Unexpected Lingo response format: %s", response_data)
        raise ValueError("Invalid response format")

    async def translate_batch_parallel(
        self,
        tasks: List[TranslationTask],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> List[TranslationResult]:
        """Translate multiple texts in parallel with progress tracking."""
        if not tasks:
            return []

        # Initialize progress tracking
        progress = BatchProgress(total_tasks=len(tasks))

        if progress_callback:
            progress_callback(progress)

        # Create semaphore for progress updates
        progress_lock = asyncio.Lock()

        async def translate_with_progress(task: TranslationTask) -> TranslationResult:
            """Translate task and update progress."""
            result = await self._translate_single_with_retry(task)

            async with progress_lock:
                if result.success:
                    progress.completed_tasks += 1
                else:
                    progress.failed_tasks += 1

                if progress_callback:
                    progress_callback(progress)

            return result

        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[translate_with_progress(task) for task in tasks],
            return_exceptions=True
        )

        # Handle any exceptions that weren't caught
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Unhandled exception for task %s: %s", tasks[i].task_id, result)
                final_results.append(TranslationResult(
                    task_id=tasks[i].task_id,
                    original_text=tasks[i].text,
                    translated_text=tasks[i].text,
                    success=False,
                    error=str(result),
                    processing_time=0.0,
                    metadata=tasks[i].metadata
                ))
            else:
                final_results.append(result)

        return final_results

    async def translate_texts_parallel(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> List[str]:
        """Translate a list of texts in parallel (simplified interface)."""
        if not texts:
            return []

        # Create tasks
        tasks = [
            TranslationTask(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                task_id=f"task_{i}",
                metadata={"index": i}
            )
            for i, text in enumerate(texts)
        ]

        # Execute parallel translation
        results = await self.translate_batch_parallel(tasks, progress_callback)

        # Extract translated texts in original order
        # Extract translated texts in original order
        translated_texts = [""] * len(texts)
        for result in results:
            index = result.metadata.get("index")
            if index is not None and 0 <= index < len(texts):
                translated_texts[index] = result.translated_text
            else:
                logger.warning("Invalid or missing index in result metadata: %s", result.task_id)
        return translated_texts

    async def translate_document_parallel(
        self,
        content: Dict[str, Any],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Dict[str, Any]:
        """Translate document content in parallel."""
        # Extract text blocks for translation
        text_blocks = self._extract_text_blocks(content)

        if not text_blocks:
            return content

        # Create translation tasks
        tasks = []
        for i, (block_id, text) in enumerate(text_blocks):
            if text.strip():  # Only translate non-empty texts
                tasks.append(TranslationTask(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    task_id=block_id,
                    metadata={"block_id": block_id, "index": i}
                ))

        # Execute parallel translation
        results = await self.translate_batch_parallel(tasks, progress_callback)

        # Apply translations back to content
        translated_content = self._apply_translations_to_content(content, results)

        return translated_content

    def _extract_text_blocks(self, content: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract text blocks from document content for translation."""
        text_blocks = []

        # Extract from pages
        for page_num, page_data in content.get("pages", {}).items():
            if isinstance(page_data, dict):
                for block_id, block_text in page_data.items():
                    if isinstance(block_text, str) and block_text.strip():
                        text_blocks.append((f"page_{page_num}_{block_id}", block_text))

        # Extract from layouts if available
        for i, layout in enumerate(content.get("layouts", [])):
            if hasattr(layout, "text") and layout.text.strip():
                text_blocks.append((f"layout_{i}", layout.text))

        return text_blocks

    def _apply_translations_to_content(
        self,
        content: Dict[str, Any],
        results: List[TranslationResult]
    ) -> Dict[str, Any]:
        """Apply translation results back to document content."""
        import copy

        # Create translation mapping
        translations = {result.task_id: result.translated_text for result in results}

        # Apply to content structure
        translated_content = copy.deepcopy(content)

        # Apply to pages
        if "pages" in translated_content:
            for page_num, page_data in translated_content["pages"].items():
                if isinstance(page_data, dict):
                    for block_id, block_text in page_data.items():
                        translation_key = f"page_{page_num}_{block_id}"
                        if translation_key in translations:
                            page_data[block_id] = translations[translation_key]

        # Apply to layouts
        if "layouts" in translated_content:
            for i, layout in enumerate(translated_content["layouts"]):
                translation_key = f"layout_{i}"
                if translation_key in translations and hasattr(layout, "text"):
                    layout.text = translations[translation_key]

        return translated_content


class ParallelTranslationService:
    """High-level service for parallel document translation."""

    def __init__(self, api_key: str, config: Optional[ParallelTranslationConfig] = None):
        self.api_key = api_key
        self.config = config or ParallelTranslationConfig.from_config()
        self._translator: Optional[ParallelLingoTranslator] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._translator = ParallelLingoTranslator(self.api_key, self.config)
        await self._translator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._translator:
            await self._translator.__aexit__(exc_type, exc_val, exc_tb)

    async def translate_large_document(
        self,
        content: Dict[str, Any],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Dict[str, Any]:
        """Translate large document with optimal parallel processing."""
        if not self._translator:
            raise RuntimeError("Service not initialized. Use async context manager.")

        return await self._translator.translate_document_parallel(
            content, source_lang, target_lang, progress_callback
        )

    async def translate_batch_texts(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> List[str]:
        """Translate batch of texts with parallel processing."""
        if not self._translator:
            raise RuntimeError("Service not initialized. Use async context manager.")

        return await self._translator.translate_texts_parallel(
            texts, source_lang, target_lang, progress_callback
        )
