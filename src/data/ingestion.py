"""
Data ingestion pipeline for processing various document sources and storing them in the vector database.
"""

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from supabase import create_client

from .gmail_loader import GmailLoader

class DataIngestionPipeline:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Supabase client
        self.supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def process_and_store_documents(
        self, 
        documents: List[Document], 
        source_type: str,
        metadata: Optional[dict] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Process documents and store them in the vector store with metadata"""
        results = {
            "chunks_processed": 0,
            "chunks_stored": 0,
            "errors": []
        }
        
        try:
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            results["chunks_processed"] = len(splits)
            
            # Process in batches
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                try:
                    # Add metadata to each split
                    for split in batch:
                        split.metadata["source_type"] = source_type
                        split.metadata["timestamp"] = datetime.now().isoformat()
                        if metadata:
                            split.metadata.update(metadata)
                    
                    # Store batch in vector store
                    self.vector_store.add_documents(batch)
                    results["chunks_stored"] += len(batch)
                    
                    # Progress update
                    print(f"Stored {results['chunks_stored']}/{results['chunks_processed']} chunks")
                    
                    # Small delay between batches
                    time.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"Error processing batch {i//batch_size}: {str(e)}"
                    print(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in document processing: {str(e)}"
            print(error_msg)
            results["errors"].append(error_msg)
            return results

    def ingest_gmail(self, 
                    credentials_path: str = 'credentials.json',
                    query: str = "newer_than:7d",
                    max_results: int = 100,
                    metadata: Optional[dict] = None) -> int:
        """Load and process emails from Gmail using the Gmail API
        
        Args:
            credentials_path: Path to the Gmail API credentials file
            query: Gmail search query (default: emails from last 7 days)
            max_results: Maximum number of emails to fetch
            metadata: Additional metadata to add to the documents
            
        Returns:
            Number of chunks processed and stored
        """
        try:
            # Initialize Gmail loader
            gmail_loader = GmailLoader(credentials_path=credentials_path)
            
            # Load emails with progress reporting
            print(f"Fetching up to {max_results} emails matching query: {query}")
            email_docs = gmail_loader.load(query=query, max_results=max_results)
            print(f"Fetched {len(email_docs)} emails")
            
            # Process and store documents
            results = self.process_and_store_documents(
                email_docs, 
                "gmail", 
                metadata,
                batch_size=50
            )
            
            if results["errors"]:
                print(f"Encountered {len(results['errors'])} errors during processing:")
                for error in results["errors"]:
                    print(f"  - {error}")
            
            return results["chunks_stored"]
            
        except Exception as e:
            print(f"Error in Gmail ingestion: {str(e)}")
            return 0 