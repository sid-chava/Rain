"""
Gmail API integration for fetching and processing emails.
"""

import os
import base64
from typing import List, Optional
from datetime import datetime, timedelta
from email import message_from_bytes

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_core.documents import Document

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailLoader:
    """Loader that uses Gmail API to fetch emails."""
    
    def __init__(self, credentials_path: str = 'credentials.json'):
        """Initialize the Gmail loader.
        
        Args:
            credentials_path: Path to the credentials.json file from Google Cloud Console
        """
        self.credentials_path = credentials_path
        self.creds = None
        
    def _get_credentials(self) -> Credentials:
        """Get valid user credentials from storage or user.
        
        Returns:
            Credentials, the obtained credential.
        """
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                print("Starting OAuth flow on port 8080")
                self.creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())
                
        return self.creds
    
    def _build_service(self):
        """Build and return Gmail API service."""
        creds = self._get_credentials()
        return build('gmail', 'v1', credentials=creds)
    
    def _parse_email(self, message) -> Optional[Document]:
        """Parse email message into a Document.
        
        Args:
            message: Gmail API message resource
            
        Returns:
            Document containing email content and metadata
        """
        try:
            msg_data = message['payload']
            
            # Get headers
            headers = {header['name']: header['value'] 
                      for header in msg_data['headers']}
            
            # Get body
            if 'parts' in msg_data:
                # Multipart message
                parts = msg_data['parts']
                body = ''
                for part in parts:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8')
                        break
            else:
                # Single part message
                body = base64.urlsafe_b64decode(
                    msg_data['body']['data']).decode('utf-8')
            
            # Create metadata
            metadata = {
                'source': 'gmail',
                'subject': headers.get('Subject', ''),
                'from': headers.get('From', ''),
                'to': headers.get('To', ''),
                'date': headers.get('Date', ''),
                'message_id': message['id']
            }
            
            return Document(
                page_content=f"Subject: {metadata['subject']}\n\n{body}",
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error parsing email {message['id']}: {str(e)}")
            return None
    
    def load(self, 
            query: str = "newer_than:7d",
            max_results: int = 100) -> List[Document]:
        """Load emails from Gmail matching the search query.
        
        Args:
            query: Gmail search query (default: emails from last 7 days)
            max_results: Maximum number of emails to fetch
            
        Returns:
            List of Documents containing email content and metadata
        """
        service = self._build_service()
        
        # Get list of messages
        results = service.users().messages().list(
            userId='me', 
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        documents = []
        
        for message in messages:
            # Get full message details
            msg = service.users().messages().get(
                userId='me', 
                id=message['id'],
                format='full'
            ).execute()
            
            doc = self._parse_email(msg)
            if doc:
                documents.append(doc)
        
        return documents 