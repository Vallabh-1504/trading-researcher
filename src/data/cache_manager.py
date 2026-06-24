import aiosqlite
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

class SQLiteCacheManager:
    """
    Handles asynchronous SQLite caching for all network-bound data fetching.
    Ensures we don't spam APIs or get rate-limited during development/testing.
    """

    def __init__(self, db_path: str = "data/cache.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    async def initialize(self) -> None:
        """Creates the cache table if it does not exist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        key TEXT PRIMARY KEY,
                        data TEXT,
                        timestamp DATETIME
                    )
                """)
                await db.commit()

            logger.debug(f"Cache initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize SQLite cache: {e}")
            raise

    async def get(self, key: str, expiry_days: int = 1) -> Optional[str]:
        """Retrieves cached data if it exists and is strictly within the expiry window."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT data, timestamp FROM api_cache WHERE key = ?", 
                    (key,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        cached_time = datetime.fromisoformat(row[1])
                        if datetime.now() - cached_time < timedelta(days=expiry_days):
                            logger.info(f"Cache HIT for key: {key}")
                            return row[0]
                        else:
                            logger.info(f"Cache EXPIRED for key: {key}")
                            return None
                        
            logger.info(f"Cache MISS for key: {key}")
            return None
        
        except Exception as e:
            logger.error(f"Error reading from cache for key {key}: {e}")
            return None
        
    async def set(self, key: str, data: str) -> None:
        """Upserts data into the cache with the current timestamp."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO api_cache (key, data, timestamp) VALUES (?, ?, ?)",
                    (key, data, datetime.now().isoformat())
                )
                await db.commit()
            logger.debug(f"Data successfully cached for key: {key}")
        except Exception as e:
            logger.error(f"Error writing to cache for key {key}: {e}")
            raise