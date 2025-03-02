import os
import PyPDF2
import json
import logging
import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Initialize HuggingFace client
client = InferenceClient(token=os.getenv('HF_TOKEN'))
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Changed model

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding(text):
    """Get embedding using HuggingFace Inference API with retries"""
    try:
        response = client.feature_extraction(
            text,
            model=MODEL_ID
        )
        return response
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        raise  # Raise the error to trigger retry

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
        print(f"Error reading PDF {file_path}: {str(e)}")
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
        print(f"Directory {pdf_dir} does not exist!")
        return
        
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Process each PDF file
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(pdf_dir, filename)
        output_path = os.path.join(embeddings_dir, f"{os.path.splitext(filename)[0]}_embeddings.json")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {filename} - already processed")
            continue
            
        # Extract text from PDF
        text = read_pdf(file_path)
        if not text:
            continue
        
        # Create chunks
        chunks = create_chunks(text)
        print(f"Created {len(chunks)} chunks from {filename}")
        
        # Store chunks and their embeddings
        documents = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding with retry logic
                embedding = get_embedding(chunk)
                
                # Prepare document
                doc = {
                    "content": chunk,
                    "embedding": embedding.tolist(),  # Convert numpy array to list
                    "file_name": filename,
                    "chunk_index": i,
                    "processed_date": datetime.datetime.now().isoformat()
                }
                documents.append(doc)
                
            except Exception as e:
                print(f"Failed to process chunk {i} from {filename}: {str(e)}")
                continue
        
        # Save to JSON file
        if documents:
            try:
                with open(output_path, 'w') as f:
                    json.dump(documents, f)
                print(f"Saved {len(documents)} embeddings for {filename}")
            except Exception as e:
                print(f"Error saving embeddings for {filename}: {str(e)}")
