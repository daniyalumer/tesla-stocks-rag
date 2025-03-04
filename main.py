import os
from dotenv import load_dotenv
from elastic_ingest import create_es_client
from search import SearchEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_search_interface():
    """Run the interactive search interface"""
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
        # Initialize search engine
        search_engine = SearchEngine(es_client)
        
        # Interactive search loop
        print("\nTesla SEC Filings Search")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'quit':
                break
                
            if not query:
                continue
                
            print("\nSearching...")
            search_results = search_engine.search(query)
            
            if not search_results['vector_results']:
                print("No results found.")
                continue
            
            # First show initial vector search results
            print("\nInitial Vector Search Results:")
            print("=" * 80)
            for i, result in enumerate(search_results['vector_results'], 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"File: {result['file_name']}")
                print(f"Chunk: {result['chunk_index']}")
                print("-" * 40)
                print(result['content'])
                print("-" * 80)
            
            # Then display LLM reranked analysis
            if search_results['llm_analysis']:
                print("\nLLM Reranked Analysis:")
                print("=" * 80)
                print(search_results['llm_analysis'])
                print("=" * 80)
                
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
    finally:
        es_client.close()

if __name__ == "__main__":
    run_search_interface()