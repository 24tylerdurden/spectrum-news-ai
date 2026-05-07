import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
import logging
import re
import random
import time
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ArticleText:
    text: str
    image_url: Optional[str] = None


class ArticleScraper:
    def __init__(self, max_length: int = 5000):
        self.max_length = max_length
        self.client = httpx.AsyncClient(
            timeout=30.0, 
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
        )

    async def scrape_article(self, url: str, fallback_summary: str = "", source_name: str = "") -> ArticleText:
        """Scrape full article text from URL."""
        # Add random delay to avoid being flagged as bot
        delay = random.uniform(1.5, 4.0)
        await asyncio.sleep(delay)
        
        try:
            # Add referer based on source
            referer = "https://www.google.com/"
            if "ndtv" in url:
                referer = "https://www.ndtv.com/"
            elif "zeenews" in url:
                referer = "https://zeenews.india.com/"
            elif "indianexpress" in url:
                referer = "https://indianexpress.com/"
            elif "timesofindia" in url:
                referer = "https://timesofindia.indiatimes.com/"
            elif "thehindu" in url:
                referer = "https://www.thehindu.com/"
            elif "swarajya" in url:
                referer = "https://swarajyamag.com/"
            elif "republicworld" in url:
                referer = "https://www.republicworld.com/"
            elif "opindia" in url:
                referer = "https://www.opindia.com/"
            elif "thewire" in url:
                referer = "https://thewire.in/"
            
            headers = {"Referer": referer}
            response = await self.client.get(url, headers=headers)
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
                # Enrich fallback with context
                article_text = self._enrich_fallback(fallback_summary, source_name)
            
            return ArticleText(text=article_text, image_url=image_url)
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}, using fallback summary")
            # Enrich fallback with context even on error
            enriched = self._enrich_fallback(fallback_summary, source_name)
            return ArticleText(text=enriched)

    def _enrich_fallback(self, summary: str, source_name: str) -> str:
        """Enrich RSS summary with context when full scraping fails."""
        if not summary:
            return f"Article from {source_name}. Full content unavailable due to access restrictions."
        
        # If summary is too short, note that it's truncated
        if len(summary) < 200:
            return f"{summary}\n\n[Note: Full article from {source_name} was unavailable for scraping. Using RSS summary.]"
        
        return f"{summary}\n\n[Source: {source_name}]"
    

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text using common patterns with better prioritization."""
        # Try specific Indian news site patterns first
        selectors = [
            # NDTV
            'div.story__content',
            'div.ins_storybody',
            # Times of India
            'div._s30J',
            'div.article-content',
            # Indian Express
            'div.story-details',
            'div.story_content',
            # Zee News
            'div.article-section',
            'div.story-content',
            # Generic patterns
            'article',
            '[class*="article-body"]',
            '[class*="story-details"]',
            '[class*="article-content"]',
            '[class*="story-content"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[class*="content-body"]',
            '[class*="articleText"]',
            '[id*="article-body"]',
            '[id*="story-content"]',
            '[id*="main-content"]',
            'main',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove any nested unwanted elements
                for unwanted in element.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'form']):
                    unwanted.decompose()
                
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 300:  # Ensure it's substantial content
                    return text
        
        # Fallback: get all paragraphs from main content areas
        content_areas = soup.find_all(['div', 'section'], class_=re.compile(r'content|story|article|body', re.I))
        if content_areas:
            for area in content_areas:
                paragraphs = area.find_all('p')
                if len(paragraphs) >= 3:  # At least 3 paragraphs
                    text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    if len(text) > 200:
                        return text
        
        # Last resort: all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
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
