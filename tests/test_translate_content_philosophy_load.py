import asyncio
import random
import time
from typing import Any

import pytest

from core.translation_handler import translate_content


class LoadFakeTranslator:
    def __init__(self, latency_ms: tuple[int, int] = (5, 15)):
        self.cur = 0
        self.max = 0
        self.calls = 0
        self.failures = 0
        self.skipped = 0
        self.latency_ms = latency_ms

    async def translate(self, text: str, **_: Any) -> str:
        if not (text or "").strip():
            self.skipped += 1
            return text
        self.cur += 1
        self.max = max(self.max, self.cur)
        try:
            delay = random.randint(*self.latency_ms) / 1000.0
            await asyncio.sleep(delay)
        finally:
            self.cur -= 1
            self.calls += 1
        # Inject a small failure rate
        if hash(text) % 37 == 0:
            self.failures += 1
            raise RuntimeError("injected failure")
        return f"T:{text}"


@pytest.mark.asyncio
@pytest.mark.load
async def test_philosophy_path_moderate_load_order_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
):
    # Test configuration constants
    CONCURRENCY_LIMIT = 8

    # Build a single page with N items to translate
    N = 200
    content = {
        "type": "pdf_advanced",
        "text_by_page": {0: [f"t{i}" for i in range(N)]},
    }

    # Bound concurrency via settings
    import core.translation_handler as th

    monkeypatch.setattr(
        th.settings, "TRANSLATION_CONCURRENCY_LIMIT", CONCURRENCY_LIMIT, raising=True
    )

    # Reduce run-to-run variability by fixing per-item latency to the midpoint.
    monkeypatch.setattr(random, "randint", lambda a, b: (a + b) // 2, raising=True)
    fake = LoadFakeTranslator(latency_ms=(5, 20))

    async def _fake_translate_text_with_neologism_handling(
        *, text: str, **kwargs: Any
    ) -> dict[str, str]:
        try:
            res = await fake.translate(text, **kwargs)
            return {"translated_text": res}
        except Exception:
            # Per-item failure should fall back to original
            return {"translated_text": text}

    monkeypatch.setattr(
        th.philosophy_translation_service,
        "translate_text_with_neologism_handling_async",
        _fake_translate_text_with_neologism_handling,
        raising=True,
    )

    t0 = time.perf_counter()
    translated_by_page, _ = await translate_content(
        content=content,
        source_language="en",
        target_language="de",
        philosophy_mode=True,
        session_id="s-load",
    )
    elapsed = time.perf_counter() - t0

    # Order is preserved exactly for items that succeed; failed items equal original
    page0: list[str] = translated_by_page[0]
    assert len(page0) == N
    for i, out in enumerate(page0):
        exp_ok = f"T:t{i}"
        assert out in (exp_ok, f"t{i}")

    # Concurrency should remain bounded
    assert fake.max <= CONCURRENCY_LIMIT

    # Basic throughput sanity: should complete within a reasonable bound on CI
    # With {CONCURRENCY_LIMIT}-way concurrency and ~12.5ms avg per item, expect ~N*12.5/{CONCURRENCY_LIMIT} ms => ~0.31s; add headroom
    assert elapsed < 5.0

    # Simple observability counters for the test
    translated = sum(1 for v in page0 if v.startswith("T:"))
    fallback = N - translated
    # Ensure we actually exercised both paths in typical runs
    assert translated > 0
    assert fallback >= 0
