import os
from dotenv import load_dotenv
from scrape import scrape_tesla_sec_filings
from embeddings import process_and_store_documents
from elastic_ingest import ElasticsearchIngestor, create_es_client
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_ingestion_pipeline():
    """Run the complete ingestion pipeline"""
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    required_vars = ['HF_TOKEN', 'ELASTIC_ENDPOINT', 'ELASTIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    # Create ES client
    es_client = create_es_client()
    if not es_client:
        logging.error("Failed to create Elasticsearch client")
        return
        
    try:
        # Step 1: Scrape SEC filings
        logging.info("Starting SEC filings scraping...")
        scrape_tesla_sec_filings()
        logging.info("SEC filings scraping completed!")
        
        # Step 2: Process documents and create embeddings
        logging.info("Starting document processing...")
        process_and_store_documents()
        logging.info("Document processing completed!")
        
        # Step 3: Ingest embeddings into Elasticsearch
        logging.info("Starting Elasticsearch ingestion...")
        ingestor = ElasticsearchIngestor(es_client)
        ingestor.ingest_embeddings()
        logging.info("Elasticsearch ingestion completed!")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
    finally:
        es_client.close()

if __name__ == "__main__":
    run_ingestion_pipeline()