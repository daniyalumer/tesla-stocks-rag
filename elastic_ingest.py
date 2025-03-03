import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ElasticsearchIngestor:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.index_name = "tesla_filings"

    def create_index(self) -> bool:
        """Create index with proper mappings"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(
                    index=self.index_name,
                    mappings={
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384,
                                "similarity": "cosine"
                            },
                            "file_name": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "processed_date": {"type": "date"}
                        }
                    }
                )
                logging.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def index_document(self, document: dict) -> bool:
        """Index a single document with retry logic"""
        try:
            self.es.index(index=self.index_name, document=document)
            return True
        except Exception as e:
            logging.error(f"Error indexing document: {str(e)}")
            raise

    def ingest_embeddings(self) -> None:
        """Ingest stored embeddings into Elasticsearch"""
        embeddings_dir = "tesla_sec_filings_embeddings"
        
        if not os.path.exists(embeddings_dir):
            logging.error(f"Directory {embeddings_dir} does not exist!")
            return
        
        if not self.create_index():
            return
        
        embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.json')]
        
        for filename in tqdm(embedding_files, desc="Ingesting embeddings"):
            file_path = os.path.join(embeddings_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    documents = json.load(f)
                
                successful_docs = 0
                for doc in documents:
                    try:
                        if self.index_document(doc):
                            successful_docs += 1
                    except Exception:
                        continue
                        
                logging.info(f"Successfully indexed {successful_docs}/{len(documents)} documents from {filename}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue

def create_es_client() -> Optional[Elasticsearch]:
    """Create Elasticsearch client"""
    try:
        es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            request_timeout=120,
            verify_certs=False,
            ssl_show_warn=False,
        )
        return es
    except Exception as e:
        logging.error(f"Error creating Elasticsearch client: {str(e)}")
        return None

