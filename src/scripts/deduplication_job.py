"""
Script to handle document ingestion and deduplication in Supabase.
Runs as a cron job to periodically clean up duplicate entries.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Set
import hashlib
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

from ..data.ingestion import DataIngestionPipeline

class DeduplicationJob:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Supabase client
        self.supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        # Initialize ingestion pipeline
        self.pipeline = DataIngestionPipeline()

    def compute_content_hash(self, content: str) -> str:
        """Compute a hash of the document content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    def fetch_documents(self, days_ago: int = 7) -> List[Dict]:
        """Fetch documents from the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days_ago)
        
        response = self.supabase_client.table("documents").select(
            "id", "content", "metadata", "created_at"
        ).gte("created_at", cutoff_date.isoformat()).execute()
        
        return response.data

    def deduplicate_documents(self, days_ago: int = 7) -> int:
        """
        Remove duplicate documents based on content hash.
        Returns the number of duplicates removed.
        """
        documents = self.fetch_documents(days_ago)
        content_hashes: Dict[str, List[str]] = {}  # hash -> list of document IDs
        duplicates: Set[str] = set()
        
        # Group documents by content hash
        for doc in documents:
            content_hash = self.compute_content_hash(doc["content"])
            if content_hash not in content_hashes:
                content_hashes[content_hash] = []
            content_hashes[content_hash].append(doc["id"])
            
            # If we found a duplicate, add all but the first document to duplicates set
            if len(content_hashes[content_hash]) > 1:
                duplicates.update(content_hashes[content_hash][1:])
        
        if duplicates:
            # Delete duplicate documents
            self.supabase_client.table("documents").delete().in_("id", list(duplicates)).execute()
        
        return len(duplicates)

    def run_ingestion_and_dedup(self):
        """Run the complete ingestion and deduplication process."""
        try:
            # First run ingestion pipeline
            print("Starting document ingestion...")
            
            # Example: Ingest from Gmail
            email_chunks = self.pipeline.ingest_gmail(
                credentials_path='credentials.json',
                query="newer_than:7d",
                max_results=100
            )
            print(f"Ingested {email_chunks} email chunks")
            
            # Run deduplication
            print("Starting deduplication process...")
            duplicates_removed = self.deduplicate_documents(days_ago=7)
            print(f"Removed {duplicates_removed} duplicate documents")
            
            return {
                "success": True,
                "email_chunks_ingested": email_chunks,
                "duplicates_removed": duplicates_removed
            }
            
        except Exception as e:
            print(f"Error in ingestion/deduplication job: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    job = DeduplicationJob()
    result = job.run_ingestion_and_dedup()
    print(f"Job completed with result: {result}") 