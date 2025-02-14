-- Drop existing table and function if they exist
drop function if exists match_documents;
drop table if exists documents;

-- Enable the vector extension if not already enabled
create extension if not exists vector;

-- Create the documents table with UUID
create table documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(3072)
);

-- Create the match_documents function
create or replace function match_documents (
  query_embedding vector(3072),
  match_count int default 10,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;