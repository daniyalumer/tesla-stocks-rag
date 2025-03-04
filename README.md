# Tesla SEC Filings RAG System

A Retrieval Augmented Generation (RAG) system for Tesla's SEC filings, combining vector search with LLM-powered analysis. Provides both CLI and REST API interfaces.

## Project Overview

This project creates an intelligent search system for Tesla's SEC filings through:
1. SEC filings download from Tesla's investor relations website
2. PDF processing and text chunking
3. Local embedding generation using sentence-transformers
4. Vector storage in Elasticsearch
5. Semantic search with LLM-powered result analysis

## Setup

### Prerequisites
- Python 3.12+
- Pipenv
- Elasticsearch instance
- HuggingFace API token (for LLM inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tesla-stock-rag.git
cd tesla-stock-rag
```

2. Install dependencies:
```bash
pipenv install
```

3. Create `.env` file:
```plaintext
HF_TOKEN=your_huggingface_token
ELASTIC_ENDPOINT=your_elasticsearch_endpoint
ELASTIC_API_KEY=your_elasticsearch_api_key
```

## Project Structure

```plaintext
tesla-stock-rag/
├── api.py                    # REST API interface
├── ingest_pipeline.py        # Data ingestion pipeline
├── main.py                   # CLI search interface
├── scrape.py                # SEC filings downloader
├── embeddings.py            # PDF processing and embeddings
├── search.py                # Search engine implementation
├── elastic_ingest.py        # Elasticsearch operations
├── tesla_sec_filings/       # Downloaded PDF files
└── tesla_sec_filings_embeddings/ # Generated embeddings
```

## Features

### Data Processing Pipeline
- **PDF Processing**: PyPDF2 for text extraction
- **Text Chunking**: Smart text splitting with configurable size
- **Embedding Generation**: Local processing using sentence-transformers
- **Vector Storage**: Elasticsearch with dense vector support

### Search Capabilities
- **Semantic Search**: Vector similarity using cosine distance
- **LLM Analysis**: Mixtral-8x7B powered result reranking
- **Dual Interfaces**: CLI and REST API

### Architecture
- **Modular Design**: Separate components for each function
- **Dependency Injection**: Efficient resource management
- **Error Handling**: Comprehensive retry logic
- **Progress Tracking**: Real-time status updates

## Usage

### Data Ingestion
First-time setup to process and store documents:
```bash
pipenv shell
python ingest_pipeline.py
```

### CLI Search Interface
Run the interactive command-line interface:
```bash
python main.py
```

### REST API
Start the API server:
```bash
python api.py
```

#### API Endpoints

**Search Endpoint**
- URL: `/search`
- Method: `POST`
- Request Body:
```json
{
    "query": "What were Tesla's revenues in 2024?"
}
```
- Response Format:
```json
{
    "vector_results": [
        {
            "score": 0.8207,
            "file_name": "tsla-20241231.pdf",
            "chunk_index": 342,
            "content": "...",
        }
    ],
    "llm_analysis": "..."
}
```

#### Example API Usage

Using curl:
```bash
curl -X POST \
  http://localhost:5000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "What were Tesla'\''s total revenues in 2024?"}'
```

Using Python:
```python
import requests

response = requests.post(
    'http://localhost:5000/search',
    json={'query': 'What were Tesla\'s total revenues in 2024?'}
)
results = response.json()
print(results['llm_analysis'])
```

### Component-wise Execution
Run individual components as needed:
```bash
python scrape.py         # Download SEC filings
python embeddings.py     # Generate embeddings
python elastic_ingest.py # Ingest into Elasticsearch
```

## Technical Details

### Models
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Search Results**: Top 5 similar documents with analysis

### Configuration
- **Chunk Size**: 1000 characters (adjustable)
- **File Types**: PDF documents
- **Storage**: 
  - Embeddings: Local JSON files
  - Search Index: Elasticsearch
- **API Port**: 5000 (default)

### Error Handling
- API failure retries
- Detailed logging
- Progress preservation
- Resource cleanup
- Input validation

## Development

### Adding Features
1. Create new module
2. Update pipeline or interfaces
3. Add environment variables if needed

### Testing
Run the test suite:
```bash
python -m pytest
```

### Code Style
- Type hints
- Docstrings
- PEP 8 compliance
- Comprehensive error handling

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## License

MIT License - See LICENSE file