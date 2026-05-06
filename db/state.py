import asyncpg
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseState:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool and create tables if needed."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
        
        async with self.pool.acquire() as conn:
            # Create processed_urls table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_urls (
                    url TEXT PRIMARY KEY,
                    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_urls_processed_at 
                ON processed_urls(processed_at)
            """)
            
        logger.info("Database state initialized")

    async def is_processed(self, url: str) -> bool:
        """Check if a URL has already been processed."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM processed_urls WHERE url = $1",
                url
            )
            return result is not None

    async def mark_processed(self, url: str):
        """Mark a URL as processed."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO processed_urls (url, processed_at) VALUES ($1, $2) "
                "ON CONFLICT (url) DO NOTHING",
                url,
                datetime.utcnow()
            )

    async def mark_processed_batch(self, urls: list[str]):
        """Mark multiple URLs as processed in a single transaction."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        if not urls:
            return
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for url in urls:
                    await conn.execute(
                        "INSERT INTO processed_urls (url, processed_at) VALUES ($1, $2) "
                        "ON CONFLICT (url) DO NOTHING",
                        url,
                        datetime.utcnow()
                    )

    async def cleanup_old(self, days: int = 7):
        """Remove processed URLs older than specified days."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM processed_urls WHERE processed_at < $1",
                cutoff_date
            )
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Cleaned up {deleted_count} old processed URLs")
            return deleted_count

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
