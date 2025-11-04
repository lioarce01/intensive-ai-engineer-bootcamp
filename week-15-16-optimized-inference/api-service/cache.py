"""
Caching Layer for vLLM API
===========================

Implements Redis-based caching for:
- Response caching (frequent identical queries)
- Prefix caching (common prompt prefixes)
- TTL-based expiration

Installation:
    pip install redis

Setup Redis:
    docker run -d -p 6379:6379 redis:alpine
"""

import redis
import json
import logging
from typing import Optional, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages Redis caching for LLM responses.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_connections: int = 50,
    ):
        """
        Initialize cache manager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_connections: Maximum connection pool size
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                max_connections=max_connections,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")

            # Initialize statistics
            self.hits = 0
            self.misses = 0

        except redis.ConnectionError as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            logger.warning("Running without cache")
            self.redis_client = None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response.

        Args:
            key: Cache key

        Returns:
            Cached response dict or None
        """
        if not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(key)
            if cached:
                self.hits += 1
                logger.debug(f"Cache HIT: {key[:16]}...")
                return json.loads(cached)
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {key[:16]}...")
                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """
        Set cached response with TTL.

        Args:
            key: Cache key
            value: Response dictionary
            ttl: Time-to-live in seconds (default 1 hour)
        """
        if not self.redis_client:
            return

        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key[:16]}... (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def delete(self, key: str):
        """
        Delete cached response.

        Args:
            key: Cache key
        """
        if not self.redis_client:
            return

        try:
            self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key[:16]}...")

        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    def clear_all(self):
        """
        Clear all cached responses.
        """
        if not self.redis_client:
            return

        try:
            self.redis_client.flushdb()
            logger.info("🗑️  Cache cleared")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.redis_client:
            return {
                "status": "disabled",
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
            }

        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        try:
            info = self.redis_client.info()
            return {
                "status": "active",
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_keys": self.redis_client.dbsize(),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "status": "error",
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }

    def close(self):
        """
        Close Redis connection.
        """
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


class PrefixCache:
    """
    Implements prefix caching for common prompt beginnings.

    This is useful when many prompts share the same prefix
    (e.g., system instructions, few-shot examples).
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix_namespace = "prefix:"

    def get_prefix_key(self, prefix: str) -> str:
        """
        Generate cache key for prefix.
        """
        prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
        return f"{self.prefix_namespace}{prefix_hash}"

    def cache_prefix(self, prefix: str, kv_cache: Any, ttl: int = 7200):
        """
        Cache KV states for a prefix.

        Args:
            prefix: Common prompt prefix
            kv_cache: Key-value cache states
            ttl: Time-to-live (default 2 hours)
        """
        key = self.get_prefix_key(prefix)
        # Note: In practice, you'd serialize the KV cache appropriately
        self.cache.set(key, {"prefix": prefix, "kv_cache": str(kv_cache)}, ttl)

    def get_prefix_cache(self, prefix: str) -> Optional[Any]:
        """
        Retrieve cached KV states for prefix.
        """
        key = self.get_prefix_key(prefix)
        cached = self.cache.get(key)
        if cached:
            return cached.get("kv_cache")
        return None


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Cache Manager Demo")
    print("=" * 60)

    # Initialize cache
    cache = CacheManager()

    # Test data
    test_key = "test_prompt_123"
    test_response = {
        "generated_text": "This is a test response.",
        "num_tokens": 10,
        "inference_time_ms": 45.2,
    }

    # Set cache
    print("\n1. Setting cache entry...")
    cache.set(test_key, test_response, ttl=60)

    # Get cache
    print("\n2. Retrieving cache entry...")
    cached = cache.get(test_key)
    print(f"   Cached response: {cached}")

    # Stats
    print("\n3. Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Miss
    print("\n4. Testing cache miss...")
    result = cache.get("nonexistent_key")
    print(f"   Result: {result}")

    # Final stats
    print("\n5. Final statistics:")
    stats = cache.get_stats()
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")

    # Cleanup
    cache.close()
    print("\n✅ Cache demo completed!")
