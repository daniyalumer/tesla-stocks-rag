import os
import PyPDF2
import json
import logging
import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Get embedding using SentenceTransformer locally"""
    try:
        # Generate embedding
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        return None

def read_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + ' '
        return text.strip()
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {str(e)}")
        return None

def create_chunks(text, chunk_size=1000):
    """Split text into chunks of approximately chunk_size characters"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_and_store_documents():
    """Process PDFs and store embeddings locally"""
    pdf_dir = "tesla_sec_filings"
    embeddings_dir = "tesla_sec_filings_embeddings"
    
    # Ensure directories exist
    if not os.path.exists(pdf_dir):
        logging.error(f"Directory {pdf_dir} does not exist!")
        return
        
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Process each PDF file
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(pdf_dir, filename)
        output_path = os.path.join(embeddings_dir, f"{os.path.splitext(filename)[0]}_embeddings.json")
        
        # Skip if already processed
        if os.path.exists(output_path):
            logging.info(f"Skipping {filename} - already processed")
            continue
            
        # Extract text from PDF
        text = read_pdf(file_path)
        if not text:
            continue
        
        # Create chunks
        chunks = create_chunks(text)
        logging.info(f"Created {len(chunks)} chunks from {filename}")
        
        # Store chunks and their embeddings
        documents = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = get_embedding(chunk)
                if embedding is None:
                    continue
                
                # Prepare document
                doc = {
                    "content": chunk,
                    "embedding": embedding.tolist(),
                    "file_name": filename,
                    "chunk_index": i,
                    "processed_date": datetime.datetime.now().isoformat()
                }
                documents.append(doc)
                
            except Exception as e:
                logging.error(f"Failed to process chunk {i} from {filename}: {str(e)}")
                continue
        
        # Save to JSON file
        if documents:
            try:
                with open(output_path, 'w') as f:
                    json.dump(documents, f)
                logging.info(f"Saved {len(documents)} embeddings for {filename}")
            except Exception as e:
                logging.error(f"Error saving embeddings for {filename}: {str(e)}")

if __name__ == "__main__":
    process_and_store_documents()