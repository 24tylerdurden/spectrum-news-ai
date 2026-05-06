# Biased India - Automated News Pipeline

A Python-based automated news pipeline for an Indian political news aggregator that fetches news from multiple outlets, clusters same-story articles, classifies them by political lean, and generates structured left/right perspective articles via Claude AI.

## Features

- **RSS Feed Fetching**: Automatically fetches articles from 9 Indian news sources with different political leanings
- **Article Clustering**: Uses sentence-transformers to group similar stories together
- **AI-Powered Perspectives**: Generates left and right perspectives using Claude (claude-sonnet-4-20250514)
- **Direct Database Storage**: Saves articles and perspectives directly to PostgreSQL
- **Scheduled Execution**: Runs automatically every 3 hours (configurable) using APScheduler
- **Duplicate Prevention**: Tracks processed URLs to avoid duplicates
- **Error Handling**: Robust error handling with fallbacks and logging

## Tech Stack

- Python 3.11+
- feedparser — RSS parsing
- httpx — async HTTP requests
- beautifulsoup4 + lxml — article text scraping
- sentence-transformers — headline embedding + clustering
- anthropic — Claude API for perspective generation
- apscheduler — cron job scheduling
- postgresql + asyncpg — storing processed state
- python-dotenv — environment config
- Pillow + httpx — image fetching/validation

## Project Structure

```
news_pipeline/
├── main.py                  # entry point, starts scheduler
├── config.py                # env vars, source definitions
├── scraper/
│   ├── rss_fetcher.py       # fetch + parse RSS feeds
│   └── article_scraper.py   # scrape full article text from URL
├── clustering/
│   └── clusterer.py         # embed headlines, group same stories
├── ai/
│   └── perspective_writer.py # Claude API calls, prompt + parse
├── pipeline/
│   └── orchestrator.py      # main flow: fetch→cluster→write→post
├── api/
│   └── client.py            # save articles to database
├── db/
│   └── state.py             # track processed URLs, avoid duplicates
└── requirements.txt
```

## Installation

### Using Docker (Recommended)

1. Clone the repository and navigate to the project directory:
```bash
cd indian-biased-ai
```

2. Copy the example environment file and configure it:
```bash
cp .env.example .env
```

3. Edit `.env` with your configuration:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/biased_india
PIPELINE_INTERVAL_HOURS=3
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=biased_india
```

4. Build and run with Docker Compose:
```bash
docker-compose up -d
```

This will:
- Build the Docker image for the news pipeline
- Start a PostgreSQL database container
- Initialize the database schema using `init.sql`
- Start the news pipeline scheduler
- Persist database data in a Docker volume

5. View logs:
```bash
docker-compose logs -f news-pipeline
```

6. Stop the containers:
```bash
docker-compose down
```

### Local Installation

1. Clone the repository and navigate to the project directory:
```bash
cd indian-biased-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the example environment file and configure it:
```bash
cp .env.example .env
```

4. Edit `.env` with your configuration:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
PIPELINE_INTERVAL_HOURS=3
```

## Database Setup

Ensure your PostgreSQL database has the following tables (as provided):

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE oauth_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    provider_user_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, provider_user_id)
);

CREATE TABLE categories (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(255) UNIQUE NOT NULL,
    original_url TEXT,
    topic VARCHAR(255) NOT NULL,
    category_id BIGINT REFERENCES categories(id),
    status VARCHAR(20) DEFAULT 'draft',
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[],
    metadata JSONB
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token TEXT NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS perspectives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    lean VARCHAR(20) NOT NULL CHECK (lean IN ('left', 'right', 'center', 'neutral')),
    lean_score SMALLINT CHECK (lean_score BETWEEN -10 AND 10),
    headline VARCHAR(512) NOT NULL,
    summary TEXT NOT NULL,
    body TEXT,
    source_name VARCHAR(255),
    source_url TEXT,
    sentiment VARCHAR(20) CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

The pipeline will automatically create a `processed_urls` table to track processed articles.

## Usage

### Docker Usage

#### Start the Pipeline with Scheduler
```bash
docker-compose up -d
```

#### View Logs
```bash
docker-compose logs -f news-pipeline
```

#### Run Once Immediately (Docker)
```bash
docker-compose run --rm news-pipeline python main.py --run-now
```

#### Stop the Pipeline
```bash
docker-compose down
```

### Local Usage

#### Start the Scheduler
Run the pipeline with the scheduler (runs every 3 hours by default):
```bash
python main.py
```

#### Run Once Immediately
Execute a single pipeline run immediately:
```bash
python main.py --run-now
```

## News Sources

The pipeline fetches from the following Indian news sources:

**Left / Centre-Left:**
- The Hindu (reliability: 0.9)
- NDTV (reliability: 0.85)
- Indian Express (reliability: 0.85)
- The Wire (reliability: 0.8)

**Right / Centre-Right:**
- Republic World (reliability: 0.75)
- Swarajya (reliability: 0.8)
- Zee News (reliability: 0.75)
- Opindia (reliability: 0.7)

**Centre:**
- Times of India (reliability: 0.8)

## Pipeline Flow

1. **Fetch**: RSS feeds from all sources are fetched
2. **Filter**: Articles older than 48 hours are skipped, already-processed URLs are filtered out
3. **Cluster**: Headlines are embedded and clustered using cosine similarity (>0.82 threshold)
4. **Validate**: Only clusters with at least one left and one right article are kept
5. **Scrape**: Full article text is scraped for the best left and right articles in each cluster
6. **Generate**: Claude AI generates structured left/right perspectives
7. **Save**: Articles and perspectives are saved to the database
8. **Track**: All URLs in processed clusters are marked as processed

## Logging

Logs are written to `logs/pipeline.log` with rotation (10MB max, 5 backups). Logs are also printed to the console.

## Error Handling

- RSS fetch failures: Logged and skipped, pipeline continues
- Article scraping failures: Falls back to RSS summary, never crashes
- Claude JSON parsing failures: Retries once with stricter prompt, then skips cluster
- Database failures: Logged and written to `failed_articles.jsonl` for manual review
- The pipeline never lets one bad article crash the entire run

## Configuration

Edit `config.py` to customize:
- News sources
- Maximum article age (default: 48 hours)
- Clustering similarity threshold (default: 0.82)
- Maximum article text length (default: 3000 chars)
- Logging paths

## License

MIT License
