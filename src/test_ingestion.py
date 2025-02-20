from data.ingestion import DataIngestionPipeline

def test_web_ingestion():
    pipeline = DataIngestionPipeline()
    
    # Test URLs (financial news/analysis)
    urls = [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
        "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240307~58209047ea.en.html",
        # Add more URLs as needed
    ]
    
    # Add metadata for better tracking
    metadata = {
        "source_type": "financial_news",
        "category": "monetary_policy"
    }
    
    try:
        num_chunks = pipeline.ingest_web_content(urls, metadata)
        print(f"Successfully processed {num_chunks} chunks from {len(urls)} URLs")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")

if __name__ == "__main__":
    test_web_ingestion()
