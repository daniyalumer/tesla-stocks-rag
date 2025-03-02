import os
from dotenv import load_dotenv
from scrape import scrape_tesla_sec_filings
from embeddings import process_and_store_documents
from elastic_ingest import ingest_embeddings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to run the entire pipeline"""
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    required_vars = ['HF_TOKEN', 'ELASTIC_ENDPOINT', 'ELASTIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    # Step 1: Scrape SEC filings
    logging.info("Starting SEC filings scraping...")
    try:
     #   scrape_tesla_sec_filings()
        logging.info("SEC filings scraping completed!")
    except Exception as e:
        logging.error(f"Error during scraping: {str(e)}")
        return
        
    # Step 2: Process documents and create embeddings
    logging.info("Starting document processing and embedding generation...")
    try:
        process_and_store_documents()
        logging.info("Document processing and embedding generation completed!")
    except Exception as e:
        logging.error(f"Error during document processing: {str(e)}")
        return
    
    # Step 3: Ingest embeddings into Elasticsearch
    logging.info("Starting Elasticsearch ingestion...")
    try:
        ingest_embeddings()
        logging.info("Elasticsearch ingestion completed!")
    except Exception as e:
        logging.error(f"Error during Elasticsearch ingestion: {str(e)}")

if __name__ == "__main__":
    main()