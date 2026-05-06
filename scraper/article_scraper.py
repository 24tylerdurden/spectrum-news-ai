import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ArticleText:
    text: str
    image_url: Optional[str] = None


class ArticleScraper:
    def __init__(self, max_length: int = 3000):
        self.max_length = max_length
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    async def scrape_article(self, url: str, fallback_summary: str = "") -> ArticleText:
        """Scrape full article text from URL."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Remove ads and social sharing
            for element in soup.find_all(class_=re.compile(r'ad|share|social|comment', re.I)):
                element.decompose()
            
            # Try to find main article content
            article_text = self._extract_article_text(soup)
            
            # Clean up text
            article_text = self._clean_text(article_text)
            
            # Truncate if too long
            if len(article_text) > self.max_length:
                article_text = article_text[:self.max_length].rsplit(' ', 1)[0] + '...'
            
            # Extract og:image if not already found
            image_url = self._extract_og_image(soup)
            
            if not article_text:
                logger.warning(f"Could not extract text from {url}, using fallback")
                article_text = fallback_summary
            
            return ArticleText(text=article_text, image_url=image_url)
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}, using fallback summary")
            return ArticleText(text=fallback_summary)

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text using common patterns."""
        # Try common article containers
        selectors = [
            'article',
            '[class*="article-body"]',
            '[class*="story-details"]',
            '[class*="article-content"]',
            '[class*="story-content"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[id*="article-body"]',
            '[id*="story-content"]',
            'main',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Ensure it's substantial content
                    return text
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            return text
        
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def _extract_og_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract og:image meta tag."""
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image.get('content')
        
        # Try twitter:image
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            return twitter_image.get('content')
        
        return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
