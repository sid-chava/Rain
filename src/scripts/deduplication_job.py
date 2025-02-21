"""
Script to handle document ingestion and deduplication in Supabase.
Runs as a cron job to periodically clean up duplicate entries.
"""

import os
import time
import signal
import resource
from datetime import datetime, timedelta
from typing import List, Dict, Set
import hashlib
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

from ..data.ingestion import DataIngestionPipeline

class TimeoutError(Exception):
    """Raised when a function times out"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class DeduplicationJob:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Set resource limits
        # Limit CPU time to 300 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        # Limit memory to 1GB
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        
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
        try:
            cutoff_date = datetime.now() - timedelta(days=days_ago)
            
            response = self.supabase_client.table("documents").select(
                "id", "content", "metadata", "created_at"
            ).gte("created_at", cutoff_date.isoformat()).execute()
            
            return response.data
        except Exception as e:
            print(f"Error fetching documents: {str(e)}")
            return []

    def deduplicate_documents(self, days_ago: int = 7, batch_size: int = 100) -> int:
        """
        Remove duplicate documents based on content hash.
        Uses batching to prevent memory spikes.
        Returns the number of duplicates removed.
        """
        try:
            documents = self.fetch_documents(days_ago)
            content_hashes: Dict[str, List[str]] = {}
            duplicates: Set[str] = set()
            processed = 0
            total_duplicates = 0
            
            # Process documents in batches
            for doc in documents:
                try:
                    content_hash = self.compute_content_hash(doc["content"])
                    if content_hash not in content_hashes:
                        content_hashes[content_hash] = []
                    content_hashes[content_hash].append(doc["id"])
                    
                    # If we found a duplicate, add all but the first document to duplicates set
                    if len(content_hashes[content_hash]) > 1:
                        duplicates.update(content_hashes[content_hash][1:])
                    
                    processed += 1
                    
                    # When batch size is reached, delete duplicates and clear memory
                    if len(duplicates) >= batch_size:
                        if duplicates:
                            self.supabase_client.table("documents").delete().in_("id", list(duplicates)).execute()
                            total_duplicates += len(duplicates)
                            duplicates.clear()
                            content_hashes.clear()
                        
                        # Add a small delay between batches to prevent CPU spikes
                        time.sleep(1)
                        
                    print(f"Processed {processed}/{len(documents)} documents. Found {total_duplicates} duplicates so far.")
                    
                except Exception as e:
                    print(f"Error processing document {doc.get('id')}: {str(e)}")
                    continue
            
            # Process any remaining duplicates
            if duplicates:
                self.supabase_client.table("documents").delete().in_("id", list(duplicates)).execute()
                total_duplicates += len(duplicates)
            
            return total_duplicates
            
        except Exception as e:
            print(f"Error in deduplication process: {str(e)}")
            return 0

    def run_ingestion_and_dedup(self):
        """Run the complete ingestion and deduplication process."""
        try:
            # Set up timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # Set 10-minute timeout
            
            results = {
                "success": False,
                "email_chunks_ingested": 0,
                "duplicates_removed": 0,
                "errors": []
            }
            
            try:
                # First run ingestion pipeline
                print("Starting document ingestion...")
                
                # Example: Ingest from Gmail using the Gmail API
                try:
                    email_chunks = self.pipeline.ingest_gmail(
                        credentials_path='credentials.json',
                        query="newer_than:7d",
                        max_results=100
                    )
                    print(f"Ingested {email_chunks} email chunks")
                    results["email_chunks_ingested"] = email_chunks
                except Exception as e:
                    error_msg = f"Error during Gmail ingestion: {str(e)}"
                    print(error_msg)
                    results["errors"].append(error_msg)
                
                # Run deduplication with batching
                print("Starting deduplication process...")
                duplicates_removed = self.deduplicate_documents(days_ago=7, batch_size=100)
                print(f"Removed {duplicates_removed} duplicate documents")
                results["duplicates_removed"] = duplicates_removed
                
                results["success"] = True
                
            except TimeoutError:
                error_msg = "Job timed out after 10 minutes"
                print(error_msg)
                results["errors"].append(error_msg)
            
            # Clear the alarm
            signal.alarm(0)
            
            return results
            
        except Exception as e:
            error_msg = f"Critical error in job: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "email_chunks_ingested": 0,
                "duplicates_removed": 0,
                "errors": [error_msg]
            }

if __name__ == "__main__":
    job = DeduplicationJob()
    result = job.run_ingestion_and_dedup()
    print(f"Job completed with result: {result}") 