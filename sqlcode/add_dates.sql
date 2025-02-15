-- Add timestamp columns if they don't exist
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS created_at timestamp with time zone DEFAULT current_timestamp,
ADD COLUMN IF NOT EXISTS published_at timestamp with time zone,
ADD COLUMN IF NOT EXISTS updated_at timestamp with time zone DEFAULT current_timestamp;

-- Create index on published_at if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_documents_published_at ON documents(published_at);

-- Migrate existing timestamp data from metadata to published_at
UPDATE documents 
SET published_at = (metadata->>'timestamp')::timestamp with time zone
WHERE published_at IS NULL 
AND metadata->>'timestamp' IS NOT NULL;

-- Drop and recreate the match_documents function
DROP FUNCTION IF EXISTS match_documents;

CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(3072),
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}',
  start_date timestamp with time zone DEFAULT null,
  end_date timestamp with time zone DEFAULT null
) RETURNS TABLE (
  id uuid,
  content text,
  metadata jsonb,
  similarity float,
  published_at timestamp with time zone
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity,
    documents.published_at
  FROM documents
  WHERE documents.metadata @> filter
    AND (start_date IS NULL OR documents.published_at >= start_date)
    AND (end_date IS NULL OR documents.published_at <= end_date)
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Update existing rows to have created_at and updated_at values
UPDATE documents 
SET created_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE created_at IS NULL; 