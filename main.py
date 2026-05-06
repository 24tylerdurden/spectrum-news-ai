#!/usr/bin/env python3
import asyncio
import sys
import logging
import os
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from pipeline import run_pipeline


def setup_logging():
    """Configure logging with rotation."""
    # Create logs directory if it doesn't exist
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


async def run_now():
    """Run the pipeline immediately."""
    logging.info("Running pipeline now...")
    try:
        await run_pipeline()
        logging.info("Pipeline run completed successfully")
    except Exception as e:
        logging.error(f"Pipeline run failed: {e}", exc_info=True)
        sys.exit(1)


def start_scheduler():
    """Start the APScheduler for periodic runs."""
    scheduler = AsyncIOScheduler()
    
    # Schedule pipeline to run every N hours
    scheduler.add_job(
        run_pipeline,
        'interval',
        hours=config.PIPELINE_INTERVAL_HOURS,
        id='news_pipeline',
        name='News Pipeline',
        replace_existing=True
    )
    
    logging.info(f"Scheduler started. Pipeline will run every {config.PIPELINE_INTERVAL_HOURS} hours.")
    
    try:
        scheduler.start()
        logging.info("Scheduler is running. Press Ctrl+C to exit.")
        
        # Keep the script running
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_forever()
        
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutting down scheduler...")
        scheduler.shutdown()
    except Exception as e:
        logging.error(f"Scheduler error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    setup_logging()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--run-now':
            asyncio.run(run_now())
        else:
            print("Usage: python main.py [--run-now]")
            print("  --run-now: Run the pipeline once immediately")
            sys.exit(1)
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
