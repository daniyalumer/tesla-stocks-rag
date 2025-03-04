from flask import Flask, request, jsonify
from dotenv import load_dotenv
from elastic_ingest import create_es_client
from search import SearchEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Create ES client and SearchEngine as global objects
es_client = create_es_client()
if es_client:
    search_engine = SearchEngine(es_client)
else:
    raise RuntimeError("Failed to create Elasticsearch client")

@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint that accepts JSON queries
    Request body format: {"query": "your search query"}
    """
    try:
        # Get query from request body
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body'
            }), 400
            
        query = data['query']
        
        # Perform search
        search_results = search_engine.search(query)
        
        # Format response
        response = {
            'vector_results': [
                {
                    'score': result['score'],
                    'file_name': result['file_name'],
                    'chunk_index': result['chunk_index'],
                    'content': result['content']
                }
                for result in search_results['vector_results']
            ],
            'llm_analysis': search_results['llm_analysis']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.teardown_appcontext
def cleanup(exc):
    """Clean up resources when the app context ends"""
    es_client.close()

if __name__ == '__main__':
    app.run(debug=True, port=5000)