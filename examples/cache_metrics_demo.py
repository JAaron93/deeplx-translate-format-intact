"""Demonstration of cache metrics and connection tracking system."""

import asyncio
import os
import sys
import time

# Add parent directory to path for imports (if not already present)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules with error handling
try:
    from examples.performance_optimization_examples import (
        get_active_connections,
        get_cache_hit_rate,
        get_cache_stats,
        instrument_cache,
        track_async_connection,
        track_connection,
    )
except ImportError as e:
    print(f"❌ Error: Could not import required modules from parent directory: {e}")
    print("   Make sure you're running this script from the correct location.")
    print(f"   Expected parent directory: {parent_dir}")
    print(f"   Current working directory: {os.getcwd()}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during import: {e}")
    sys.exit(1)


# Closure-based cache to avoid clobbering decorator state
_cache = {}


def example_cache_function(key: str) -> str:
    """Example function with proper cache miss semantics.

    This function uses closure-based caching and implements the correct pattern:
    - First call: Raises KeyError (miss)
    - Subsequent calls: Returns cached value (hit)
    - Manually increments hit counter on cache hits to align with decorator
    """
    # Check if result is in our closure cache
    if key in _cache:
        # This is a cache hit - manually increment hit counter to align with decorator
        from examples.performance_optimization_examples import _metrics_collector

        _metrics_collector.increment_cache_hit()
        return _cache[key]

    # Cache miss - compute result, store it, raise KeyError for decorator
    time.sleep(0.1)
    result = f"processed_{key}"
    _cache[key] = result
    raise KeyError(f"Cache miss for key: {key}")


# Apply the decorator
example_cache_function = instrument_cache(example_cache_function)


# Example of instrumenting a simple dict-backed cache (swap in LRU in real use)
class ExampleService:
    """Example service with cache instrumentation."""

    def __init__(self):
        self._cache = {}

    @instrument_cache
    def get_cached_data(self, key: str) -> str:
        """Get data with cache instrumentation."""
        # This method owns the cache lookup; the decorator only instruments hits/misses.
        if key in self._cache:
            return self._cache[key]
        # Simulate cache miss - could populate cache here
        # For demo purposes, just raise KeyError
        raise KeyError(f"Key {key} not found")

    def set_cached_data(self, key: str, value: str):
        """Set data in cache."""
        self._cache[key] = value


async def demonstrate_metrics():
    """Demonstrate the metrics system in action."""
    print("=== Cache Metrics and Connection Tracking Demo ===\n")

    # Initialize example service
    service = ExampleService()

    # Set some initial data
    service.set_cached_data("key1", "value1")
    service.set_cached_data("key2", "value2")

    print("1. Testing cache hit rate tracking:")
    print("   Initial cache stats:", get_cache_stats())

    # Test cache hits
    try:
        result1 = service.get_cached_data("key1")
        print(f"   Cache hit for 'key1': {result1}")
    except KeyError:
        print("   Cache miss for 'key1'")

    try:
        result2 = service.get_cached_data("key2")
        print(f"   Cache hit for 'key2': {result2}")
    except KeyError:
        print("   Cache miss for 'key2'")

    # Test cache miss
    try:
        result3 = service.get_cached_data("key3")
        print(f"   Cache hit for 'key3': {result3}")
    except KeyError:
        print("   Cache miss for 'key3'")

    print("   Cache stats after operations:", get_cache_stats())
    print(f"   Cache hit rate: {get_cache_hit_rate():.2%}")

    print("\n2. Testing connection tracking:")
    print("   Initial active connections:", get_active_connections())

    # Simulate database connections
    with track_connection():
        print("   Database connection 1 acquired")
        print("   Active connections:", get_active_connections())

        with track_connection():
            print("   Database connection 2 acquired")
            print("   Active connections:", get_active_connections())

        print("   Database connection 2 released")
        print("   Active connections:", get_active_connections())

    print("   Database connection 1 released")
    print("   Active connections:", get_active_connections())

    print("\n3. Testing async connection tracking:")
    print("   Initial active connections:", get_active_connections())

    async with track_async_connection():
        print("   Async connection 1 acquired")
        print("   Active connections:", get_active_connections())

        async with track_async_connection():
            print("   Async connection 2 acquired")
            print("   Active connections:", get_active_connections())

        print("   Async connection 2 released")
        print("   Active connections:", get_active_connections())

    print("   Async connection 1 released")
    print("   Active connections:", get_active_connections())

    print("\n4. Final metrics summary:")
    print("   Active connections:", get_active_connections())
    print("   Cache stats:", get_cache_stats())
    print(f"   Cache hit rate: {get_cache_hit_rate():.2%}")


def demonstrate_sync_metrics():
    """Demonstrate synchronous metrics usage."""
    print("\n=== Synchronous Metrics Demo ===\n")

    # Test the instrumented cache function
    print("Testing instrumented cache function:")

    # First call (cache miss)
    try:
        result = example_cache_function("test_key")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Second call (cache hit)
    try:
        result = example_cache_function("test_key")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    print("   Cache stats:", get_cache_stats())


if __name__ == "__main__":
    # Run synchronous demo
    demonstrate_sync_metrics()

    # Run async demo
    asyncio.run(demonstrate_metrics())
