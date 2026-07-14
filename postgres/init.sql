-- DDL schema script for PageRAG database tables.
-- This script can be used to set up the schemas in a clean PostgreSQL database.
-- Note: PageRAG SQLAlchemy models also create these tables automatically.

-- Create Papers Table
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    year INT,
    keywords TEXT,
    doi TEXT,
    citation_count INT DEFAULT 0
);

-- Create Pages Table
CREATE TABLE IF NOT EXISTS pages (
    id SERIAL PRIMARY KEY,
    paper_id INT NOT NULL,
    page_no INT NOT NULL,
    text TEXT NOT NULL,
    CONSTRAINT fk_paper
        FOREIGN KEY(paper_id) 
        REFERENCES papers(id)
        ON DELETE CASCADE
);

-- Create Queries Table
CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Evaluations Table
CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    query_id INT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    latency_ms FLOAT NOT NULL,
    citation_accuracy FLOAT NOT NULL,
    hallucination_score FLOAT NOT NULL,
    CONSTRAINT fk_query
        FOREIGN KEY(query_id) 
        REFERENCES queries(id)
        ON DELETE CASCADE
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_pages_paper_id ON pages(paper_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_query_id ON evaluations(query_id);
