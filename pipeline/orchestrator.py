import asyncio
import logging
from typing import List
import json

import config
from scraper import RSSFetcher, ArticleScraper, RawArticle
from clustering import ArticleClusterer, Cluster
from ai import PerspectiveWriter, PerspectiveResult
from api import DatabaseClient
from db import DatabaseState

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self):
        self.rss_fetcher = RSSFetcher(max_age_hours=config.MAX_ARTICLE_AGE_HOURS)
        self.article_scraper = ArticleScraper(max_length=config.MAX_ARTICLE_TEXT_LENGTH)
        self.clusterer = ArticleClusterer(
            similarity_threshold=config.CLUSTERING_SIMILARITY_THRESHOLD
        )
        self.perspective_writer = PerspectiveWriter(
            provider=config.AI_PROVIDER,
            model=config.AI_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
            ollama_base_url=config.OLLAMA_BASE_URL,
            gemini_api_key=config.GEMINI_API_KEY
        )
        self.db_client = DatabaseClient(config.DATABASE_URL)
        self.db_state = DatabaseState(config.DATABASE_URL)

    async def initialize(self):
        """Initialize all database connections."""
        await self.db_client.initialize()
        await self.db_state.initialize()
        logger.info("Pipeline orchestrator initialized")

    async def run_pipeline(self):
        """Run the complete news pipeline."""
        logger.info("=" * 50)
        logger.info("Starting pipeline run")
        logger.info("=" * 50)
        
        try:
            # Step 1: Fetch all RSS feeds
            logger.info("Step 1: Fetching RSS feeds...")
            raw_articles = await self.rss_fetcher.fetch_all_sources(config.sources)
            logger.info(f"Fetched {len(raw_articles)} total articles")
            
            # Step 2: Filter out already processed URLs
            logger.info("Step 2: Filtering processed URLs...")
            unprocessed_articles = []
            for article in raw_articles:
                is_processed = await self.db_state.is_processed(article.url)
                if not is_processed:
                    unprocessed_articles.append(article)
            
            logger.info(f"Found {len(unprocessed_articles)} unprocessed articles")
            logger.info(f"Unprocessed articles: {unprocessed_articles}")
            
            if not unprocessed_articles:
                logger.info("No new articles to process")
                return
            
            # Step 3: Cluster articles
            logger.info("Step 3: Clustering articles...")
            clusters = self.clusterer.cluster_articles(unprocessed_articles)
            logger.info(f"Found {len(clusters)} valid clusters")
            logger.info(f"Clusters data: {clusters}")
            
            if not clusters:
                logger.info("No valid clusters found")
                return
            
            # Step 4: Process each cluster
            logger.info("Step 4: Processing clusters...")
            successful_posts = 0
            failed_clusters = 0
            
            for i, cluster in enumerate(clusters, 1):
                logger.info(f"Processing cluster {i}/{len(clusters)}...")
                
                try:
                    await self._process_cluster(cluster)
                    successful_posts += 1
                    logger.info(f"Successfully processed cluster {i}")
                except Exception as e:
                    logger.error(f"Failed to process cluster {i}: {e}")
                    failed_clusters += 1
                    continue
            
            # Step 5: Log summary
            logger.info("=" * 50)
            logger.info("Pipeline run completed")
            logger.info(f"Total clusters: {len(clusters)}")
            logger.info(f"Successfully posted: {successful_posts}")
            logger.info(f"Failed: {failed_clusters}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}", exc_info=True)
            raise

    async def _process_cluster(self, cluster: Cluster):
        """Process a single cluster: scrape, generate perspectives, save to DB."""
        left_article = cluster.left_articles[0]
        right_article = cluster.right_articles[0]
        
        # Scrape full article text
        logger.info(f"Scraping articles from {left_article.source_name} and {right_article.source_name}...")
        
        left_text_result = await self.article_scraper.scrape_article(
            left_article.url,
            fallback_summary=left_article.summary,
            source_name=left_article.source_name
        )
        
        right_text_result = await self.article_scraper.scrape_article(
            right_article.url,
            fallback_summary=right_article.summary,
            source_name=right_article.source_name
        )
        
        # Generate perspectives
        logger.info("Generating AI perspectives...")
        perspective_result = self.perspective_writer.generate_perspectives(
            cluster,
            left_text_result.text,
            right_text_result.text
        )
        
        if not perspective_result:
            raise Exception("Failed to generate perspectives")
        
        # Pick best image URL
        image_url = left_article.image_url or right_article.image_url or left_text_result.image_url or right_text_result.image_url
        
        # Save to database
        logger.info(f"Saving to database: {perspective_result.topic}")
        article_id = await self.db_client.save_article(perspective_result, image_url)
        
        if not article_id:
            raise Exception("Article already exists or failed to save")
        
        # Mark all URLs in cluster as processed
        all_urls = [a.url for a in cluster.left_articles + cluster.right_articles + cluster.centre_articles]
        await self.db_state.mark_processed_batch(all_urls)
        
        logger.info(f"Saved article with ID: {article_id}")

    async def close(self):
        """Close all connections."""
        await self.article_scraper.close()
        self.perspective_writer.close()
        await self.db_client.close()
        await self.db_state.close()
        logger.info("Pipeline orchestrator closed")


async def run_pipeline():
    """Convenience function to run the pipeline."""
    orchestrator = PipelineOrchestrator()
    try:
        await orchestrator.initialize()
        await orchestrator.run_pipeline()
    finally:
        await orchestrator.close()
