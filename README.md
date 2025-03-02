# Tesla SEC Filings RAG System

A Retrieval Augmented Generation (RAG) system that processes Tesla's SEC filings using embeddings and Elasticsearch for advanced search and analysis capabilities.

## Project Overview

This project creates a searchable database of Tesla's SEC filings by:
1. Downloading SEC filings from Tesla's investor relations website
2. Processing PDFs into text chunks
3. Generating embeddings using HuggingFace's Inference API
4. Storing both raw text and embeddings in Elasticsearch

## Setup

### Prerequisites
- Python 3.8+
- Pipenv
- Elasticsearch instance
- HuggingFace API token

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

3. Create `.env` file with required credentials:
```plaintext
HF_TOKEN=your_huggingface_token
ELASTIC_ENDPOINT=your_elasticsearch_endpoint
ELASTIC_API_KEY=your_elasticsearch_api_key
```

## Project Structure

```
tesla-stock-rag/
├── main.py              # Main pipeline orchestration
├── scrape.py           # SEC filings downloader
├── embeddings.py       # PDF processing and embedding generation
├── elastic_ingest.py   # Elasticsearch ingestion
├── tesla_sec_filings/  # Downloaded PDF files
└── tesla_sec_filings_embeddings/  # Generated embeddings
```

## Features

- **PDF Processing**: Uses PyPDF2 for reliable text extraction
- **Text Chunking**: Intelligent text splitting for optimal processing
- **Embedding Generation**: Uses sentence-transformers model via HuggingFace API
- **Error Handling**: Comprehensive retry logic and error reporting
- **Progress Tracking**: Real-time processing status with tqdm
- **Modular Design**: Separate modules for scraping, processing, and storage

## Usage

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Run the complete pipeline:
```bash
python main.py
```

Or run individual components:
```bash
python scrape.py        # Only download SEC filings
python embeddings.py    # Only process PDFs and generate embeddings
python elastic_ingest.py # Only ingest into Elasticsearch
```

## Configuration

### Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Optimized for: Semantic search and similarity matching

### Chunking Parameters
- Default chunk size: 1000 characters
- Adjustable via `chunk_size` parameter in `create_chunks()`

## Error Handling

The system includes:
- Automatic retries for API failures
- Detailed error logging
- Progress preservation (skip already processed files)
- Input validation and error reporting

## Development

### Adding New Features

1. Create a new module for your feature
2. Update `main.py` to include your module
3. Add any new environment variables to `.env`

### Testing

Run tests with:
```bash
python -m pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

