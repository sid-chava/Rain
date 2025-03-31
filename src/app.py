import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from langchain_core.prompts import ChatPromptTemplate
import time
from postgrest.exceptions import APIError
from datetime import datetime, timedelta
from typing import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain import hub
import requests
import pandas as pd
from fredapi import Fred

# FRED API setup
FRED_API_KEY = st.secrets["api_keys"].get("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

def get_market_data(days=7):
    """Fetch market data from FRED API"""
    if not FRED_API_KEY:
        st.warning("FRED API key not found in secrets. Please add it to continue.")
        return None
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # FRED Series IDs
    series = {
        'VIX': 'VIXCLS',  # VIX
        '10Y': 'DGS10',   # 10-Year Treasury
        '2Y': 'DGS2',     # 2-Year Treasury
        '3M': 'DTB3',     # 3-Month Treasury
    }
    
    market_data = {}
    for name, series_id in series.items():
        try:
            data = fred.get_series(
                series_id,
                start_date=start_date,
                end_date=end_date
            )
            if not data.empty:
                df = pd.DataFrame(data, columns=['close'])
                df.index.name = 'date'
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date']).dt.date
                market_data[name] = df
                
        except Exception as e:
            st.error(f"Error fetching data for {name}: {str(e)}")
            
    return market_data

# Set environment variables from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["api_keys"]["OPENAI_API_KEY"]
os.environ["SUPABASE_URL"] = st.secrets["supabase"]["SUPABASE_URL"]
os.environ["SUPABASE_SERVICE_KEY"] = st.secrets["supabase"]["SUPABASE_SERVICE_KEY"]


# Initialize Supabase client
supabase_url = st.secrets["supabase"]["SUPABASE_URL"]
supabase_key = st.secrets["supabase"]["SUPABASE_SERVICE_KEY"]
supabase_client = create_client(supabase_url, supabase_key)

# Initialize LLM and embeddings
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize vector store
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Initialize prompts
qa_prompt = hub.pull("rlm/rag-prompt")

# Create a report-focused prompt template
report_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert macro and volatility analyst with deep experience in global markets, monetary policy, and risk analysis. When analyzing markets and responding to queries:

First, display a "MARKET SNAPSHOT" section at the very top with:
- Current VIX level and its daily change
- Treasury yield levels (10Y, 2Y, 3M) and their daily changes
- Key yield curve spreads (10Y-2Y, 10Y-3M)

Then list the dates of all newsletters you are analyzing in chronological order.

After that, proceed with your analysis, including:

- Start with key market metrics and indicators:
  * Treasury yields and yield curve dynamics
  * Credit spreads and financial conditions
  * Currency movements and cross-asset correlations
  * Commodity prices and trends

- Provide detailed analysis of:
  * Central bank policies and their market implications (Primarily the Fed)
  * Geopolitical risks and their potential market impact
  * Positioning data and market sentiment indicators
  * Systematic flows and technical factors
  * Cross-asset relationships and regime changes

- Structure your responses with:
  * Clear executive summary highlighting key points
  * Detailed analysis backed by specific data points
  * Forward-looking scenarios and their probabilities
  * Specific risks to the current market narrative
  * Actionable trading implications

Base your analysis on the provided context, but incorporate your broad market knowledge where relevant. Be specific rather than ambivalent - represent the views in the context. Maintain a professional, analytical tone and clearly distinguish between facts and opinions."""),
    ("human", "Generate a detailed market analysis with specific implications for our portfolio positions:\n\nContext: {context}")
])

# Define state for processing
class State(TypedDict):
    query: str
    context: List[dict]
    answer: str
    processed_docs: int
    total_docs: int

# Streamlit UI
st.set_page_config(page_title="Market Analysis Generator", layout="wide")
st.title("Market Analysis Generator")

# Add About section and disclaimer
st.markdown("""
### About
This tool helps minimize time spent scrolling through daily newsletters by automatically extracting and analyzing key market insights. 
The tool will become more comprehensive as more newsletters are indexed over the coming days.

⚠️ **Note**: If you encounter an API error, please wait a few moments and try again. This is usually due to temporary rate limiting.

---
""")

# Add sidebar controls
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Document Settings")
    num_docs = st.slider("Number of documents to retrieve", min_value=3, max_value=30, value=3)
    
    st.subheader("Date Range")
    date_options = ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    date_range = st.selectbox("Select time range", date_options, index=1)

def get_date_filter(date_range):
    now = datetime.now()
    if date_range == "Last 24 hours":
        return now - timedelta(days=1)
    elif date_range == "Last 7 days":
        return now - timedelta(days=7)
    elif date_range == "Last 30 days":
        return now - timedelta(days=30)
    return None

# Function to safely retrieve documents
def safe_similarity_search(date_filter, k=3):
    try:
        # Create date filter for Supabase
        filter_dict = {}
        if date_filter:
            # Use timestamp range for more precise filtering
            start_of_day = date_filter.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = date_filter.replace(hour=23, minute=59, second=59, microsecond=999999)
            filter_dict = {
                "created_at": {
                    "gte": start_of_day.isoformat(),
                    "lte": end_of_day.isoformat()
                }
            }
        
        # Add debug print
        st.write(f"Attempting to retrieve {k} documents with filter: {filter_dict}")
        
        # Single vector search with filter
        docs = vector_store.similarity_search(
            "What are the latest market updates and economic indicators?",
            k=k,
            filter=filter_dict
        )
        
        # Debug print for retrieved documents
        st.write(f"Retrieved {len(docs)} documents")
        
        # Group documents by source with better error handling
        docs_by_source = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown Source')
            if not source:
                source = 'Unknown Source'
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
            
            # Debug print for document metadata
            st.write(f"Document metadata: {doc.metadata}")
        
        return docs, docs_by_source
    except APIError as e:
        st.error(f"API Error during document retrieval: {str(e)}")
        raise e
    except Exception as e:
        st.error(f"Unexpected error during document retrieval: {str(e)}")
        raise e

def process_documents(docs):
    """Process a batch of documents and update the progress bar"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch market data
    market_data = get_market_data(days=7)
    
    # Prepare market data summary
    market_summary = []
    if market_data:
        # Format VIX data
        if 'VIX' in market_data and not market_data['VIX'].empty:
            vix_data = market_data['VIX']
            latest_vix = vix_data.iloc[-1]['close']
            prev_vix = vix_data.iloc[-2]['close'] if len(vix_data) > 1 else latest_vix
            vix_change = latest_vix - prev_vix
            market_summary.append(f"VIX: {latest_vix:.2f} ({vix_change:+.2f})")

        # Format Treasury yields
        yields = {}
        for tenor in ['10Y', '2Y', '3M']:
            if tenor in market_data and not market_data[tenor].empty:
                yield_data = market_data[tenor]
                latest_yield = yield_data.iloc[-1]['close']
                prev_yield = yield_data.iloc[-2]['close'] if len(yield_data) > 1 else latest_yield
                yield_change = latest_yield - prev_yield
                yields[tenor] = latest_yield
                market_summary.append(f"{tenor} Treasury: {latest_yield:.3f}% ({yield_change:+.3f})")
        
        # Calculate and add yield curve spreads
        if '10Y' in yields and '2Y' in yields:
            spread_10y2y = yields['10Y'] - yields['2Y']
            market_summary.append(f"10Y-2Y Spread: {spread_10y2y:.3f}%")
        if '10Y' in yields and '3M' in yields:
            spread_10y3m = yields['10Y'] - yields['3M']
            market_summary.append(f"10Y-3M Spread: {spread_10y3m:.3f}%")
    
    # Prepare context with market data
    context_with_dates = [
        f"Current Date: {current_date}\n\n"
        f"MARKET SNAPSHOT ({datetime.now().strftime('%Y-%m-%d %H:%M')} ET):\n"
        + "\n".join(market_summary) + "\n\n"
        "Analysis based on the following sources:\n"
    ]
    
    for doc in docs:
        date_str = ""
        if doc.metadata.get('published_at'):
            date_str = f"[Source Date: {doc.metadata['published_at'][:10]}]\n"
        context_with_dates.append(f"{date_str}{doc.page_content}")
    
    context = "\n\n".join(context_with_dates)
    messages = report_prompt.invoke({"context": context})
    return llm.stream(messages)

# Add tabs at the top level
tab1, tab2, tab3 = st.tabs(["Market Analysis", "Q&A", "Sources"])

# Q&A Interface in tab2
with tab2:
    user_question = st.text_input("Ask a question about the market:", placeholder="e.g., Which energy company laid off 20% of its employees recently?")
    if st.button("Get Answer"):
        with st.spinner("Searching for relevant information..."):
            # Get date filter
            date_filter = get_date_filter(date_range)
            
            # Create date filter for Supabase
            filter_dict = {}
            if date_filter:
                filter_dict = {
                    "created_at": {
                        "gte": date_filter.isoformat(),
                        "lte": datetime.now().isoformat()
                    }
                }
            
            # Get documents for the specific question with date filtering
            qa_docs = vector_store.similarity_search(
                user_question,
                k=num_docs,
                filter=filter_dict
            )
            
            # Prepare context and generate answer
            current_date = datetime.now().strftime("%Y-%m-%d")
            qa_context_parts = [f"Current Date: {current_date}\n\nAnalysis based on the following sources:\n"]
            
            for doc in qa_docs:
                date_str = ""
                if doc.metadata.get('published_at'):
                    date_str = f"[Source Date: {doc.metadata['published_at'][:10]}]\n"
                qa_context_parts.append(f"{date_str}{doc.page_content}")
            
            qa_context = "\n\n".join(qa_context_parts)
            qa_messages = qa_prompt.invoke({
                "context": qa_context,
                "question": user_question
            })
            
            # Display the answer with streaming
            answer_container = st.empty()
            current_answer = ""
            for chunk in llm.stream(qa_messages):
                if chunk.content is not None:
                    current_answer += chunk.content
                    answer_container.markdown(current_answer)
            
            # Show sources used for the answer
            st.markdown("### Sources Used")
            try:
                for i, doc in enumerate(qa_docs, 1):
                    try:
                        source = doc.metadata.get('source', 'Unknown Source')
                        date = ''
                        if isinstance(doc.metadata.get('published_at'), str):
                            date = doc.metadata['published_at'][:10]
                        elif isinstance(doc.metadata.get('timestamp'), str):
                            date = doc.metadata['timestamp'][:10]
                        else:
                            date = 'No date'
                            
                        with st.expander(f"Source {i} - {source} ({date})"):
                            st.markdown(f"**Source:** {source}")
                            st.markdown(f"**Date:** {date}")
                            st.markdown("**Content:**")
                            st.markdown(doc.page_content)
                    except Exception as e:
                        st.error(f"Error displaying source {i}: {str(e)}")
            except Exception as e:
                st.error(f"Error processing sources: {str(e)}")

# Main report generation
if st.button("Generate Analysis", key="generate_analysis"):
    try:
        # Initialize the progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get date filter
        date_filter = get_date_filter(date_range)
        
        with st.spinner("Retrieving documents..."):
            # Get documents using the notebook's query
            docs, docs_by_source = safe_similarity_search(
                date_filter,
                k=num_docs
            )
            
            status_text.text(f"Processing {len(docs)} documents from {len(docs_by_source)} sources...")
            
            with tab1:
                report_container = st.empty()
                
                # Stream the report generation
                current_content = ""
                for chunk in process_documents(docs):
                    if chunk.content is not None:
                        current_content += chunk.content
                        report_container.markdown(current_content)
            
            with tab3:
                st.markdown("### Document Sources")
                try:
                    for source, source_docs in docs_by_source.items():
                        with st.expander(f"{source} ({len(source_docs)} documents)"):
                            for i, doc in enumerate(source_docs, 1):
                                try:
                                    date = ''
                                    if isinstance(doc.metadata.get('published_at'), str):
                                        date = doc.metadata['published_at'][:10]
                                    elif isinstance(doc.metadata.get('timestamp'), str):
                                        date = doc.metadata['timestamp'][:10]
                                    else:
                                        date = 'No date'
                                        
                                    st.markdown(f"#### Document {i} ({date})")
                                    st.markdown(doc.page_content)
                                    st.markdown("---")
                                except Exception as e:
                                    st.error(f"Error displaying document {i}: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing source documents: {str(e)}")
            
            st.success("Analysis generated successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try again with different settings or contact support if the issue persists.") 