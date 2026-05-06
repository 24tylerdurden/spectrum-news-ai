import feedparser
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class RawArticle:
    title: str
    url: str
    summary: str
    image_url: Optional[str]
    source_name: str
    lean: str
    reliability: float
    published_at: datetime


class RSSFetcher:
    def __init__(self, max_age_hours: int = 48):
        self.max_age_hours = max_age_hours
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def fetch_all_sources(self, sources: list[dict]) -> List[RawArticle]:
        """Fetch articles from all RSS sources."""
        all_articles = []
        
        for source in sources:
            try:
                articles = await self._fetch_source(source)
                all_articles.extend(articles)
                logger.info(f"Fetched {len(articles)} articles from {source['name']}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {source['name']}: {e}")
                continue
        
        await self.client.aclose()
        return all_articles

    async def _fetch_source(self, source: dict) -> List[RawArticle]:
        """Fetch articles from a single RSS source."""
        response = await self.client.get(source["rss_url"])
        feed = feedparser.parse(response.content)
        
        articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_age_hours)
        
        for entry in feed.entries:
            try:
                published_at = self._parse_published_date(entry)
                
                # Skip articles older than max_age_hours
                if published_at < cutoff_time:
                    continue
                
                url = self._extract_url(entry)
                if not url:
                    continue
                
                title = entry.get('title', '').strip()
                if not title:
                    continue
                
                summary = self._extract_summary(entry)
                image_url = self._extract_image(entry, url)
                
                article = RawArticle(
                    title=title,
                    url=url,
                    summary=summary,
                    image_url=image_url,
                    source_name=source["name"],
                    lean=source["lean"],
                    reliability=source["reliability"],
                    published_at=published_at
                )
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error parsing entry from {source['name']}: {e}")
                continue
        
        return articles

    def _parse_published_date(self, entry) -> datetime:
        """Parse publication date from RSS entry."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        else:
            return datetime.utcnow()

    def _extract_url(self, entry) -> Optional[str]:
        """Extract URL from RSS entry."""
        if hasattr(entry, 'link'):
            return entry.link
        return None

    def _extract_summary(self, entry) -> str:
        """Extract summary/description from RSS entry."""
        if hasattr(entry, 'summary'):
            return entry.summary
        elif hasattr(entry, 'description'):
            return entry.description
        return ""

    def _extract_image(self, entry, article_url: str) -> Optional[str]:
        """Extract image URL from RSS entry."""
        # Try enclosure first
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').startswith('image/'):
                    return enclosure.get('href')
        
        # Try media:content
        if hasattr(entry, 'media_content') and entry.media_content:
            for media in entry.media_content:
                if media.get('medium') == 'image' or media.get('type', '').startswith('image/'):
                    return media.get('url')
        
        # Try to extract from summary HTML
        if hasattr(entry, 'summary'):
            soup = BeautifulSoup(entry.summary, 'lxml')
            img_tag = soup.find('img')
            if img_tag and img_tag.get('src'):
                return img_tag.get('src')
        
        return None
