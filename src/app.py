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

- Start with key market metrics and indicators:
  * VIX, MOVE, and other volatility measures
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


Base your analysis primarily on the provided context, but incorporate your broad market knowledge where relevant. Be specific rather than ambivalent - represent the views in the context. Maintain a professional, analytical tone and clearly distinguish between facts and opinions."""),
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
        # Use the exact query that worked in the notebook
        docs = vector_store.similarity_search(
            "What are the latest market updates and economic indicators?",
            k=k
        )
        
        # Apply date filtering if needed
        if date_filter:
            docs = [
                doc for doc in docs 
                if doc.metadata.get('timestamp') and 
                datetime.fromisoformat(doc.metadata['timestamp']) >= date_filter
            ]
        
        # Group documents by source
        docs_by_source = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown Source')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        return docs, docs_by_source
    except APIError as e:
        raise e

def process_documents(docs):
    """Process a batch of documents and update the progress bar"""
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = report_prompt.invoke({"context": context})
    return llm.stream(messages)

# Add tabs at the top level
tab1, tab2, tab3 = st.tabs(["Market Analysis", "Q&A", "Sources"])

# Q&A Interface in tab2
with tab2:
    user_question = st.text_input("Ask a question about the market:", placeholder="e.g., Which energy company laid off 20% of its employees recently?")
    if st.button("Get Answer"):
        with st.spinner("Searching for relevant information..."):
            # Get documents for the specific question
            qa_docs = vector_store.similarity_search(
                user_question,
                k=num_docs
            )
            
            # Prepare context and generate answer
            qa_context = "\n\n".join(doc.page_content for doc in qa_docs)
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
            for i, doc in enumerate(qa_docs, 1):
                with st.expander(f"Source {i} - {doc.metadata.get('source', 'Unknown Source')}"):
                    if 'timestamp' in doc.metadata:
                        st.caption(f"Date: {doc.metadata['timestamp'][:10]}")
                    st.markdown(doc.page_content)

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
                st.markdown(f"Found {len(docs)} documents from {len(docs_by_source)} different sources")
                
                # Display documents grouped by source
                for source, source_docs in docs_by_source.items():
                    with st.expander(f"{source} ({len(source_docs)} documents)"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"#### Document {i}")
                            # Show timestamp if available
                            if 'timestamp' in doc.metadata:
                                st.caption(f"Date: {doc.metadata['timestamp'][:10]}")
                            
                            # Show content
                            st.markdown("**Content:**")
                            st.markdown(doc.page_content)
                            
                            # Show metadata in a collapsible section
                            with st.expander("View Metadata"):
                                st.json(doc.metadata)
                            
                            if i < len(source_docs):
                                st.markdown("---")  # Add separator between documents
            
            st.success("Analysis generated successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try again with different settings or contact support if the issue persists.") 