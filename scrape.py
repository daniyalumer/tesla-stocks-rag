import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

def create_download_directory():
    download_dir = "tesla_sec_filings"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return download_dir

def download_pdf(url, filename, download_dir, page_num):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Add page number to filename
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_page{page_num}{ext}"
        
        filepath = os.path.join(download_dir, new_filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded: {new_filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def scrape_tesla_sec_filings():
    base_url = "https://ir.tesla.com/sec-filings"
    download_dir = create_download_directory()
    
    for page in range(1, 30):  # 131 pages total
        print(f"Processing page {page}")
        
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}#list_id=tcl-list-1&page={page}"
            
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all PDF links
            filing_links = soup.find_all('a', href=lambda x: x and x.endswith('.pdf'))
            
            for link in filing_links:
                pdf_url = urljoin(base_url, link['href'])
                filename = pdf_url.split('/')[-1]
                
                # Download the PDF with page number
                download_pdf(pdf_url, filename, download_dir, page)
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(1)
                
        except Exception as e:
            print(f"Error processing page {page}: {str(e)}")
            continue
        
        # Add a delay between pages
        time.sleep(2)
