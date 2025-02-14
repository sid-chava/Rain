"""
Main entry point for the data ingestion system.
"""

from data.ingestion import DataIngestionPipeline
from data.scheduler import DataIngestionScheduler

def ingest_financial_news():
    pipeline = DataIngestionPipeline()
    urls = [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
        # Add more URLs
    ]
    num_chunks = pipeline.ingest_web_content(urls)
    print(f"Processed {num_chunks} chunks from web content")

def ingest_newsletters():
    pipeline = DataIngestionPipeline()
    num_chunks = pipeline.ingest_emails("path/to/email/directory")
    print(f"Processed {num_chunks} chunks from newsletters")

def main():
    scheduler = DataIngestionScheduler()
    
    # Add jobs with their intervals (in minutes)
    scheduler.add_job(ingest_financial_news, interval_minutes=60)  # Every hour
    scheduler.add_job(ingest_newsletters, interval_minutes=360)    # Every 6 hours
    
    # Start the scheduler
    scheduler.start()

if __name__ == "__main__":
    main() 