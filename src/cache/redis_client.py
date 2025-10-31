"""Redis caching client for high-speed data access."""

import json
import pickle
from typing import Any, Optional, Union

import pandas as pd
import redis
from redis.exceptions import ConnectionError, RedisError

from utils.config import settings
from utils.logger import logger


class RedisCache:
    """
    Redis caching client with automatic serialization.

    Features:
    - Automatic serialization (JSON, pickle, pandas)
    - TTL management
    - Connection pooling
    - Graceful degradation
    - Cache statistics
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = None,
        max_connections: int = 10,
    ):
        """
        Initialize Redis cache client.

        Args:
            redis_url: Redis connection URL (default from settings)
            default_ttl: Default TTL in seconds (default from settings)
            max_connections: Maximum connections in pool
        """
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl or settings.cache_ttl
        self.max_connections = max_connections

        # Connection pool for better performance
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=max_connections,
            decode_responses=False,  # Handle binary data
        )

        self._client: Optional[redis.Redis] = None
        self._is_available = True

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0

        logger.info(
            "redis_cache_initialized",
            redis_url=self._mask_url(self.redis_url),
            default_ttl=self.default_ttl,
        )

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
            # Test connection
            try:
                self._client.ping()
                self._is_available = True
                logger.info("redis_connection_established")
            except (ConnectionError, RedisError) as e:
                self._is_available = False
                logger.error("redis_connection_failed", error=str(e))

        return self._client

    def _mask_url(self, url: str) -> str:
        """Mask password in Redis URL for logging."""
        if "@" in url:
            parts = url.split("@")
            return f"redis://***@{parts[1]}"
        return url

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: str = "auto",
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = default_ttl, 0 = no expiry)
            serializer: Serialization method ('auto', 'json', 'pickle', 'pandas')

        Returns:
            True if successful, False otherwise
        """
        if not self._is_available:
            logger.warning("redis_unavailable", operation="set", key=key)
            return False

        try:
            # Determine serialization method
            if serializer == "auto":
                if isinstance(value, pd.DataFrame):
                    serializer = "pandas"
                elif isinstance(value, (dict, list, str, int, float, bool)):
                    serializer = "json"
                else:
                    serializer = "pickle"

            # Serialize value
            if serializer == "json":
                serialized = json.dumps(value)
            elif serializer == "pandas":
                serialized = pickle.dumps(value)
            elif serializer == "pickle":
                serialized = pickle.dumps(value)
            else:
                raise ValueError(f"Unknown serializer: {serializer}")

            # Set with TTL
            ttl_seconds = self.default_ttl if ttl is None else ttl
            if ttl_seconds > 0:
                self.client.setex(key, ttl_seconds, serialized)
            else:
                self.client.set(key, serialized)

            logger.debug(
                "cache_set",
                key=key,
                ttl=ttl_seconds,
                serializer=serializer,
                size_bytes=len(serialized) if isinstance(serialized, bytes) else len(serialized.encode()),
            )

            return True

        except (ConnectionError, RedisError) as e:
            self.errors += 1
            self._is_available = False
            logger.error("cache_set_failed", key=key, error=str(e))
            return False

    def get(
        self,
        key: str,
        serializer: str = "auto",
    ) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            serializer: Deserialization method ('auto', 'json', 'pickle', 'pandas')

        Returns:
            Cached value or None if not found
        """
        if not self._is_available:
            self.misses += 1
            return None

        try:
            serialized = self.client.get(key)

            if serialized is None:
                self.misses += 1
                logger.debug("cache_miss", key=key)
                return None

            # Deserialize value
            if serializer == "auto":
                # Try JSON first, fall back to pickle
                try:
                    value = json.loads(serialized)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # SECURITY NOTE: pickle.loads() should only be used with trusted data
                    # This cache is internal and only stores data from our own application
                    value = pickle.loads(serialized)  # nosec B301
            elif serializer == "json":
                value = json.loads(serialized)
            elif serializer in ("pandas", "pickle"):
                # SECURITY NOTE: pickle.loads() should only be used with trusted data
                # This cache is internal and only stores data from our own application
                value = pickle.loads(serialized)  # nosec B301
            else:
                raise ValueError(f"Unknown serializer: {serializer}")

            self.hits += 1
            logger.debug("cache_hit", key=key)

            return value

        except (ConnectionError, RedisError) as e:
            self.errors += 1
            self.misses += 1
            self._is_available = False
            logger.error("cache_get_failed", key=key, error=str(e))
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self._is_available:
            return False

        try:
            result = self.client.delete(key)
            logger.debug("cache_delete", key=key, deleted=bool(result))
            return bool(result)

        except (ConnectionError, RedisError) as e:
            self.errors += 1
            logger.error("cache_delete_failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self._is_available:
            return False

        try:
            return bool(self.client.exists(key))

        except (ConnectionError, RedisError) as e:
            self.errors += 1
            logger.error("cache_exists_failed", key=key, error=str(e))
            return False

    def flush(self, pattern: Optional[str] = None) -> int:
        """
        Flush cache keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "market:*"), None for all keys

        Returns:
            Number of keys deleted
        """
        if not self._is_available:
            return 0

        try:
            if pattern:
                keys = self.client.keys(pattern)
                if keys:
                    count = self.client.delete(*keys)
                    logger.info("cache_flushed", pattern=pattern, count=count)
                    return count
                return 0
            else:
                self.client.flushdb()
                logger.warning("cache_flushed_all")
                return -1  # Unknown count

        except (ConnectionError, RedisError) as e:
            self.errors += 1
            logger.error("cache_flush_failed", pattern=pattern, error=str(e))
            return 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "is_available": self._is_available,
        }

        # Get Redis info if available
        if self._is_available:
            try:
                info = self.client.info("stats")
                stats["redis_stats"] = {
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            except (ConnectionError, RedisError):
                pass

        return stats

    def reset_stats(self):
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        logger.info("cache_stats_reset")

    def ping(self) -> bool:
        """
        Ping Redis to check connection.

        Returns:
            True if Redis is available, False otherwise
        """
        try:
            self.client.ping()
            self._is_available = True
            return True
        except (ConnectionError, RedisError) as e:
            self._is_available = False
            logger.error("redis_ping_failed", error=str(e))
            return False

    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("redis_connection_closed")


# Cache key builders
class CacheKeys:
    """Standard cache key patterns."""

    @staticmethod
    def market_price(symbol: str) -> str:
        """Cache key for current market price."""
        return f"market:price:{symbol}"

    @staticmethod
    def historical_data(symbol: str, start: str, end: str, interval: str) -> str:
        """Cache key for historical data."""
        return f"historical:{symbol}:{start}:{end}:{interval}"

    @staticmethod
    def indicator(symbol: str, indicator: str, params: str = "") -> str:
        """Cache key for technical indicator."""
        return f"indicators:{symbol}:{indicator}:{params}"

    @staticmethod
    def prediction(symbol: str, model: str) -> str:
        """Cache key for model prediction."""
        return f"prediction:{symbol}:{model}"

    @staticmethod
    def model_active(model_name: str) -> str:
        """Cache key for active model version."""
        return f"model:active:{model_name}"
