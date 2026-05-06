import asyncpg
import uuid
from datetime import datetime
from typing import Optional
import json
import logging
import re

from ai import PerspectiveResult

logger = logging.getLogger(__name__)


class DatabaseClient:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
        logger.info("Database client initialized")

    async def _get_or_create_category(self, category_name: str) -> int:
        """Get category ID by name, or create it if it doesn't exist."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            # Try to get existing category
            category_id = await conn.fetchval(
                "SELECT id FROM categories WHERE slug = $1",
                category_name.lower()
            )
            
            if category_id:
                return category_id
            
            # Generate a unique ID for the new category
            new_id = await conn.fetchval(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM categories"
            )
            
            # Create the category
            slug = category_name.lower().replace(' ', '-')
            await conn.execute("""
                INSERT INTO categories (id, name, slug, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
            """,
            new_id,
            category_name,
            slug,
            datetime.utcnow(),
            datetime.utcnow()
            )
            
            logger.info(f"Created new category: {category_name} with ID {new_id}")
            return new_id

    async def save_article(self, result: PerspectiveResult, image_url: Optional[str] = None) -> Optional[str]:
        """Save article and perspectives to database."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Generate slug from topic
                    slug = self._generate_slug(result.topic)
                    
                    # Check if article with this slug already exists
                    existing = await conn.fetchval(
                        "SELECT id FROM articles WHERE slug = $1",
                        slug
                    )
                    if existing:
                        logger.info(f"Article with slug '{slug}' already exists, skipping")
                        return None
                    
                    # Get or create category
                    category_id = await self._get_or_create_category(result.category)
                    
                    # Build metadata
                    metadata = {"image_url": image_url, "description": result.description}
                    if image_url:
                        metadata["image_url"] = image_url
                    
                    # Insert article
                    article_id = await conn.fetchval("""
                        INSERT INTO articles (slug, original_url, topic, category_id, status, published_at, created_at, updated_at, metadata, tags)
                        VALUES ($1, $2, $3, $4, 'published', $5, $6, $7, $8, $9)
                        RETURNING id
                    """, 
                    slug,
                    result.left_source_url,  # Use left source as primary original URL
                    result.topic,
                    category_id,
                    datetime.utcnow(),
                    datetime.utcnow(),
                    datetime.utcnow(),
                    json.dumps(metadata),
                    result.tags
                    )
                    
                    # Insert left perspective
                    await conn.execute("""
                        INSERT INTO perspectives (article_id, lean, lean_score, headline, summary, body, source_name, source_url, created_at)
                        VALUES ($1, 'left', -5, $2, $3, $4, $5, $6, $7)
                    """,
                    article_id,
                    result.left_headline,
                    result.left_summary,
                    result.left_body,
                    result.left_source_name,
                    result.left_source_url,
                    datetime.utcnow()
                    )
                    
                    # Insert right perspective
                    await conn.execute("""
                        INSERT INTO perspectives (article_id, lean, lean_score, headline, summary, body, source_name, source_url, created_at)
                        VALUES ($1, 'right', 5, $2, $3, $4, $5, $6, $7)
                    """,
                    article_id,
                    result.right_headline,
                    result.right_summary,
                    result.right_body,
                    result.right_source_name,
                    result.right_source_url,
                    datetime.utcnow()
                    )
                    
                    logger.info(f"Successfully saved article '{result.topic}' with ID {article_id}")
                    return str(article_id)
                    
        except Exception as e:
            logger.error(f"Failed to save article to database: {e}")
            raise

    def _generate_slug(self, topic: str) -> str:
        """Generate a URL-friendly slug from topic."""
        # Convert to lowercase
        slug = topic.lower()
        # Replace spaces and special chars with hyphens
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s-]+', '-', slug)
        slug = slug.strip('-')
        # Add timestamp to ensure uniqueness
        timestamp = int(datetime.utcnow().timestamp())
        return f"{slug}-{timestamp}"

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database client connection pool closed")
