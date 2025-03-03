import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SearchEngine:
    def __init__(self, es_client: Elasticsearch, model_name: str = 'all-MiniLM-L6-v2'):
        self.es = es_client
        self.model = SentenceTransformer(model_name)
        self.index_name = "tesla_filings"

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using KNN search
        """
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            
            # Construct KNN query
            knn_query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding.tolist(),
                    "k": k,
                    "num_candidates": 100
                },
                "_source": ["content", "file_name", "chunk_index"]
            }
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=knn_query
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'content': hit['_source']['content'],
                    'file_name': hit['_source']['file_name'],
                    'chunk_index': hit['_source']['chunk_index'],
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

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

def main():
    """Interactive search interface"""
    # Load environment variables
    load_dotenv()
    
    # Create ES client
    es_client = create_es_client()
    if not es_client:
        return
    
    # Initialize search engine
    search_engine = SearchEngine(es_client)
    
    print("\nTesla SEC Filings Search")
    print("Enter 'quit' to exit")
    
    try:
        while True:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'quit':
                break
                
            if not query:
                continue
                
            print("\nSearching...")
            results = search_engine.search(query)
            
            if not results:
                print("No results found.")
                continue
                
            print("\nTop results:")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"File: {result['file_name']}")
                print(f"Chunk: {result['chunk_index']}")
                print("-" * 40)
                print(result['content'])
                print("-" * 80)
    finally:
        es_client.close()

if __name__ == "__main__":
    main()