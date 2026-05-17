import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# Environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PIPELINE_INTERVAL_HOURS = int(os.getenv("PIPELINE_INTERVAL_HOURS", "3"))

# AI Model Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")  # Options: anthropic, openai, ollama
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-4-20250514")  # Default model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# News sources with political leanings
sources = [
    # LEFT / CENTRE-LEFT
    {
        "name": "The Hindu",
        "rss_url": "https://www.thehindu.com/news/national/feednews/",
        "lean": "left",
        "reliability": 0.9
    },
    {
        "name": "NDTV",
        "rss_url": "https://feeds.feedburner.com/ndtvnews-india-news",
        "lean": "left",
        "reliability": 0.85
    },
    {
        "name": "Indian Express",
        "rss_url": "https://indianexpress.com/section/india/feed/",
        "lean": "left",
        "reliability": 0.85
    },
    {
        "name": "The Wire",
        "rss_url": "https://thewire.in/feed",
        "lean": "left",
        "reliability": 0.8
    },

    # RIGHT / CENTRE-RIGHT
    {
        "name": "Republic World",
        "rss_url": "https://www.republicworld.com/rss/india-news.xml",
        "lean": "right",
        "reliability": 0.75
    },
    {
        "name": "Swarajya",
        "rss_url": "https://swarajyamag.com/feed",
        "lean": "right",
        "reliability": 0.8
    },
    {
        "name": "Zee News",
        "rss_url": "https://zeenews.india.com/rss/india-national-news.xml",
        "lean": "right",
        "reliability": 0.75
    },
    {
        "name": "Opindia",
        "rss_url": "https://www.opindia.com/feed/",
        "lean": "right",
        "reliability": 0.7
    },

    # CENTRE (use for neutral summary context only)
    {
        "name": "Times of India",
        "rss_url": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
        "lean": "centre",
        "reliability": 0.8
    },
]

# Pipeline settings
MAX_ARTICLE_AGE_HOURS = 48
CLUSTERING_SIMILARITY_THRESHOLD = 0.60
MAX_ARTICLE_TEXT_LENGTH = 5000

# Logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
