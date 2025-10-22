-- Create the nuq schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS nuq;

-- Create the queue_scrape table
CREATE TABLE IF NOT EXISTS nuq.queue_scrape (
  id SERIAL PRIMARY KEY,
  url VARCHAR(2048) NOT NULL,
  mode VARCHAR(50) DEFAULT 'scrape',
  options JSONB,
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create other necessary tables
CREATE TABLE IF NOT EXISTS nuq.queue_crawl (
  id SERIAL PRIMARY KEY,
  url VARCHAR(2048) NOT NULL,
  options JSONB,
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nuq.queue_map (
  id SERIAL PRIMARY KEY,
  url VARCHAR(2048) NOT NULL,
  options JSONB,
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_queue_scrape_status ON nuq.queue_scrape(status);
CREATE INDEX IF NOT EXISTS idx_queue_crawl_status ON nuq.queue_crawl(status);
CREATE INDEX IF NOT EXISTS idx_queue_map_status ON nuq.queue_map(status);
