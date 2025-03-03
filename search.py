import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from huggingface_hub import InferenceClient
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
        # Initialize Mixtral client
        self.llm_client = InferenceClient(token=os.getenv("HF_TOKEN"))

    def _rank_results_with_llm(self, query: str, results: List[Dict]) -> str:
        """Use Mixtral to rank and select most relevant result"""
        prompt = f"""Given the user query: '{query}'
        
        And these document chunks from Tesla's SEC filings:
        
        {'-' * 80}
        {'\n\n'.join([f'Document {i+1}:\n{r["content"]}' for i, r in enumerate(results)])}
        {'-' * 80}
        
        First, analyze which document chunk is most relevant to answering the query.
        Then, provide a clear, concise answer based on that document.
        Format your response as:
        Most Relevant Document: <document number>
        Answer: <your answer based on the document>
        """
        
        try:
            response = self.llm_client.text_generation(
                prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=512,
                temperature=0.1
            )
            return response
            
        except Exception as e:
            logging.error(f"Error in LLM ranking: {str(e)}")
            return None

    def search(self, query: str, k: int = 5) -> Dict:
        """
        Search for similar documents using KNN search and LLM ranking
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
            
            # Use LLM to rank and explain results
            if results:
                llm_response = self._rank_results_with_llm(query, results)
                return {
                    'vector_results': results,
                    'llm_analysis': llm_response
                }
            
            return {'vector_results': [], 'llm_analysis': None}
            
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return {'vector_results': [], 'llm_analysis': None}

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
