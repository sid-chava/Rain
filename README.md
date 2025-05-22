2/13/2025
Can clone, but will need a supabase db, open ai key, and gmail key. 

Currently my supabase db has a table called `messages` with the following columns:
- id
- role
- content
- created_at

Pipeline:
Pulls emails from Gmail API - notebook cell to pull backwards
- gmail_loader.py - pulls the emails
- ingestion.py - splits/processes emails
- Cron job runs on AWS ec2 every 6 hours to pull new emails in

UI
Simple streamlit interface with daily macroeconomic report based on newsletters. RAG q&a box 
- used langchain for rag setup, openai embedding-3 small for vector embeddings, and supabase vector store
- app.py -  constains the system prompt and similarity search function (cosine similarity)

Auxillary
- deduplication_job.py runs on aws ec2 every 6 hours to ingest new and clean any potential duplicates out. 
- notebook testenvironment.ipynb is where I worked out the intial working version of this
- docs are in reference to the tutorial I looked at from langchain



4/15/25
had shut the live version of this down because supabase compute was getting expensive.
Adapted a version of this script and set up for a friend who worked at a hedge fund
