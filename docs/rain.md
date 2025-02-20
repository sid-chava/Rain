# Product Requirements Document (PRD) for Personal Hedge Fund Agent System

## Overview
- **Product Name:** Personal Hedge Fund Agent System
- **Objective:**  
  Build an agent-driven system that monitors macro trends, manages portfolio positions, processes live news, and calculates risk. The system will generate daily reports and timely recommendations for portfolio adjustments, leveraging LLM-driven insights.

## Objectives
- Aggregate diverse data sources (newsletters, free news feeds, portfolio data).
- Analyze macroeconomic indicators, market sentiment, and risk metrics.
- Provide actionable recommendations with minimal delay.
- Operate without reliance on paid API subscriptions using open-source tools.

## Scope
- **In Scope:**  
  - Data ingestion from newsletters, scraped news, and portfolio updates.
  - Development of separate agents for macro analysis, position management, live news processing, and risk management.
  - Scheduled reporting and alert system.
- **Out of Scope:**  
  - Real-time trade execution.
  - Direct integration with live trading accounts.

## Features

### 1. Data Ingestion & Storage
- **Data Sources:**  
  - Email newsletters and documents.  
  - Free scraped news data.  
  - User-provided portfolio positions and earnings.
- **Processing:**  
  - Implement a Retrieval-Augmented Generation (RAG) system with two main components:
    1. **Indexing Pipeline:**
       - Document loading using LangChain's DocumentLoaders
       - Text splitting with RecursiveCharacterTextSplitter (1000 char chunks, 200 char overlap)
       - Embedding generation using OpenAI embeddings
       - Vector storage in FAISS or similar
    2. **Retrieval & Generation Pipeline:**
       - Similarity search to find relevant documents
       - Context-aware prompting using LangChain's RAG prompts
       - LLM-based answer generation with source tracking
- **Key Requirements:**  
  - Automated pipelines for data extraction, transformation, and loading
  - Secure and scalable storage with scheduled updates and indexing
  - Efficient retrieval with source attribution for generated insights

### 2. Macro Agent
- **Functionality:**  
  - Analyze macroeconomic trends from ingested documents and news.
  - Monitor capital, debt, and equity market changes, plus regulatory shifts.
  - Update analysis with current portfolio data.
- **Key Requirements:**  
  - Integration with an LLM for contextual analysis.
  - Regular (e.g., daily) processing with urgent alerts on significant events.
  - Ability to ingest custom inputs (portfolio updates).

### 3. Position Manager
- **Functionality:**  
  - Monitor and track current portfolio positions.
  - Simulate and assess recommended changes.
  - Maintain historical performance records for backtesting.
- **Key Requirements:**  
  - Integration with the data ingestion system to fetch up-to-date portfolio data.
  - Simulation module to test potential adjustments.
  - Clear interfaces for feedback and manual overrides.

### 4. Live News Agent
- **Functionality:**  
  - Continuously scrape and index free news sources.
  - Perform sentiment analysis and anomaly detection.
  - Issue real-time or near-real-time alerts for market-impacting events.
- **Key Requirements:**  
  - Robust web scraping framework using open-source libraries (e.g., BeautifulSoup, Scrapy).
  - Lightweight sentiment analysis, possibly augmented by LLM insights.
  - Customizable alert thresholds and notification systems.

### 5. Risk Manager
- **Functionality:**  
  - Calculate key risk metrics (e.g., VaR, exposure limits).
  - Monitor overall portfolio risk in relation to market conditions.
  - Generate periodic risk reports.
- **Key Requirements:**  
  - Predefined risk calculation functions (can be scripted in Python).
  - Integration with both portfolio and macro data sources.
  - Alerting mechanism for risk threshold breaches.

### 6. Scheduler & Reporting
- **Functionality:**  
  - Aggregate outputs from all agents into a central daily report.
  - Schedule and dispatch alerts for urgent market events.
  - Provide a dashboard for real-time monitoring and historical analysis.
- **Key Requirements:**  
  - Cron-based scheduling or similar task management.
  - Centralized dashboard with clear visualization of recommendations and risk metrics.
  - Customizable report formats and notification settings.

## Architecture & Integration

- **Modular Design:**  
  - Each agent (Macro, Position Manager, Live News, Risk Manager) operates independently using LangGraph for orchestration
  - Communication via RESTful APIs or message queues
- **Data Flow:**  
  1. **Ingestion Layer:** 
     - Data collection from various sources
     - Document splitting and embedding generation
     - Storage in vector database with metadata preservation
  2. **Processing Layer:** 
     - LLM-driven analysis using RAG for context-aware processing
     - Domain-specific functions for each agent
     - State management using LangGraph for complex workflows
  3. **Reporting Layer:** 
     - Aggregated outputs with source attribution
     - Scheduled reports and alerts
- **Technology Stack:**  
  - **Core Framework:** LangChain for RAG implementation
  - **Orchestration:** LangGraph for agent workflows
  - **Data Storage:** FAISS for vector search
  - **LLM Integration:** Groq or similar for generation
  - **Embeddings:** OpenAI text-embedding-3-large
  - **Web Scraping:** BeautifulSoup, Scrapy
  - **Scheduling:** Celery, cron jobs

## Implementation Roadmap

1. **Phase 1: Setup & Data Ingestion**
   - Set up vector database and RAG system.
   - Develop automated pipelines for newsletters, documents, and news scraping.
   - Validate data ingestion and indexing process.

2. **Phase 2: Agent Development**
   - **Macro Agent:**  
     - Integrate LLM for macro trend analysis.
     - Develop routines to update with portfolio data.
   - **Position Manager:**  
     - Build portfolio tracking and simulation modules.
   - **Live News Agent:**  
     - Implement web scraping and sentiment analysis.
     - Establish real-time alert system.
   - **Risk Manager:**  
     - Script basic risk calculation functions.
     - Integrate with macro and portfolio data.

3. **Phase 3: Integration & Testing**
   - Integrate agents using APIs/message queues.
   - Develop a central dashboard for monitoring and reporting.
   - Conduct unit, integration, and backtesting of recommendations.
   - Iterate based on feedback and performance metrics.

4. **Phase 4: Deployment & Monitoring**
   - Deploy system on a secure, scalable platform.
   - Set up scheduling for daily reports and urgent alerts.
   - Monitor system performance, refine modules, and implement improvements.

## Success Metrics
- **Accuracy:** Timely and correct recommendations.
- **Risk Management:** Reduction in exposure and adherence to risk thresholds.
- **User Engagement:** Positive feedback from daily reports and alerts.
- **System Reliability:** High uptime and smooth integration between agents.

## Risks & Mitigations
- **Data Quality Issues:**  
  - Implement robust error handling and validation in data ingestion.
- **LLM Misinterpretations:**  
  - Combine LLM outputs with rule-based checks and simulations.
- **Integration Complexity:**  
  - Modular design to isolate issues and allow independent troubleshooting.
- **Cost Constraints:**  
  - Use open-source tools and free data sources to minimize operational costs.

## Future Considerations
- Expand capabilities to include semi-real-time trade execution.
- Enhance risk modeling using machine learning techniques.
- Incorporate more advanced user feedback loops and adaptive algorithms.