import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

def create_es_client():
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
        print(f"Error creating Elasticsearch client: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def index_document(es, index_name, document):
    """Index a document with retry logic"""
    try:
        es.index(index=index_name, document=document)
        return True
    except Exception as e:
        print(f"Error indexing document: {str(e)}")
        raise  # Raise the error to trigger retry

def ingest_embeddings():
    """Ingest stored embeddings into Elasticsearch"""
    embeddings_dir = "tesla_sec_filings_embeddings"
    index_name = "tesla_filings"
    
    # Check if embeddings directory exists
    if not os.path.exists(embeddings_dir):
        print(f"Directory {embeddings_dir} does not exist!")
        return
    
    # Initialize Elasticsearch client
    es = create_es_client()
    if not es:
        return
    
    # Create index if it doesn't exist
    if not es.indices.exists(index=index_name):
        try:
            es.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,  # BGE-small embedding dimension
                            "similarity": "cosine"
                        },
                        "file_name": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "processed_date": {"type": "date"}
                    }
                }
            )
            print(f"Created index: {index_name}")
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return
    
    # Process each embedding file
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.json')]
    
    for filename in tqdm(embedding_files, desc="Ingesting embeddings"):
        file_path = os.path.join(embeddings_dir, filename)
        
        try:
            # Load embeddings from JSON file
            with open(file_path, 'r') as f:
                documents = json.load(f)
            
            # Index each document
            successful_docs = 0
            for doc in documents:
                try:
                    if index_document(es, index_name, doc):
                        successful_docs += 1
                except Exception as e:
                    continue
                    
            print(f"Successfully indexed {successful_docs}/{len(documents)} documents from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    ingest_embeddings()