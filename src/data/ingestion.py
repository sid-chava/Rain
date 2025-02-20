"""
Data ingestion pipeline for processing various document sources and storing them in the vector database.
"""

import os
from datetime import datetime
from typing import List, Optional

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
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
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
        metadata: Optional[dict] = None
    ) -> int:
        """Process documents and store them in the vector store with metadata"""
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Add metadata to each split
        for split in splits:
            split.metadata["source_type"] = source_type
            split.metadata["timestamp"] = datetime.now().isoformat()
            if metadata:
                split.metadata.update(metadata)
        
        # Store in vector store
        self.vector_store.add_documents(splits)
        
        return len(splits)

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
        # Initialize Gmail loader
        gmail_loader = GmailLoader(credentials_path=credentials_path)
        
        # Load emails
        email_docs = gmail_loader.load(query=query, max_results=max_results)
        
        # Process and store documents
        return self.process_and_store_documents(email_docs, "gmail", metadata) 