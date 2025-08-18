"""Demonstration of cache metrics and connection tracking system."""

import asyncio
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.performance_optimization_examples import (
    get_cache_stats,
    get_cache_hit_rate,
    get_active_connections,
    instrument_cache,
    track_async_connection,
    track_connection,
)


# Example cache that can be instrumented
@instrument_cache
def example_cache_function(key: str) -> str:
    """Example function with cache instrumentation."""
    # Simulate some computation
    time.sleep(0.1)
    return f"processed_{key}"


# Example of instrumenting an existing LRU cache
class ExampleService:
    """Example service with cache instrumentation."""

    def __init__(self):
        self._cache = {}

    @instrument_cache
    def get_cached_data(self, key: str) -> str:
        """Get data with cache instrumentation."""
        if key in self._cache:
            return self._cache[key]
        # Simulate cache miss
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
    print("   Cache hit rate:", f"{get_cache_hit_rate():.2%}")
    print("   Active connections:", get_active_connections())
    print("   Cache stats:", get_cache_stats())


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
