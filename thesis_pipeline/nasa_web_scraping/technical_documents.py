import os
import json
import pandas as pd
import fitz  # PyMuPDF
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple, Set
import re
from dateutil.parser import parse as parse_date
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from browser_utils import create_chrome_driver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# THIS SCRIPT ONLY USES THE UPPER CLASS TO DOWNLOAD THE PDFs FROM NASA TECHNICAL REPORTS SERVER
# THE LOWER CLASS IS USED TO PROCESS THE PDFs AND EXTRACT THE TEXT FROM THEM BUT IT IS NOT CURRENTLY BEING USED

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PDFDownloader:
    """Class to download PDFs from NASA Technical Reports Server."""
    
    def __init__(
        self,
        base_url,
        output_dir="NTRS_PDFS_CONFERENCE_GLOBAL",
        max_docs=None,
        start_from=0,
        headless=True,
        max_pages=80,
        max_workers=4,
    ):
        """
        Args:
            max_docs: 最大下载数；None 表示不限制（下载全部可用，直到翻页结束）
            max_pages: 最多爬取搜索结果页数（每页约 100 条）
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.start_from = start_from
        self.headless = headless
        self.max_pages = max_pages
        self.max_workers = max(1, min(max_workers or 1, 8))
        self.logger = logging.getLogger(__name__)
        self.base_domain = "https://ntrs.nasa.gov"
        self.progress_file = os.path.join(output_dir, "download_progress.json")
        self._progress_lock = Lock()

        os.makedirs(output_dir, exist_ok=True)
    
    def load_progress(self):
        """Load download progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    return progress.get('documents_downloaded', 0), progress.get('page_num', 0)
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
        return self.start_from, self.start_from // 100
    
    def save_progress(self, documents_downloaded, page_num):
        """Save download progress to file."""
        try:
            progress = {
                'documents_downloaded': documents_downloaded,
                'page_num': page_num,
                'timestamp': datetime.now().isoformat()
            }
            with self._progress_lock:
                with open(self.progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def download_file(self, url, filename):
        """Download a file using requests. Returns (success, skipped_existing)."""
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            self.logger.info(f"Skipping existing file: {os.path.basename(filename)}")
            return True, True
            
        try:
            # If URL is relative, make it absolute
            if url.startswith('/'):
                url = f"{self.base_domain}{url}"
                
            self.logger.info(f"Downloading from: {url}")
            
            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Check if it's a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type and not url.endswith('.pdf'):
                self.logger.warning(f"Downloaded file may not be a PDF. Content-Type: {content_type}")
            
            # Save the file
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was downloaded and is not empty
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return True, False

            self.logger.error(f"File download failed or file is empty: {filename}")
            return False, False
                
        except Exception as e:
            self.logger.error(f"Error downloading file {url}: {e}")
            return False, False
    
    def _extract_download_links(self, driver, page_num: int):
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        download_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if '/downloads/' in href and '.pdf' in href:
                card = a_tag.find_parent('div', class_='search-result-card')
                title = "Unknown"
                if card:
                    title_tag = card.find('h3', class_='title')
                    if title_tag:
                        title = title_tag.text.strip()
                download_links.append((title, href))

        self.logger.info(f"Found {len(download_links)} download links on page {page_num+1}")

        if not download_links:
            self.logger.info("Trying direct Selenium approach to find download links")
            download_buttons = driver.find_elements(By.CSS_SELECTOR, "a[title='Download Document']")
            self.logger.info(f"Found {len(download_buttons)} download buttons with Selenium")

            for button in download_buttons:
                try:
                    href = button.get_attribute('href')
                    if href and '/downloads/' in href and '.pdf' in href:
                        card = button.find_element(By.XPATH, "./ancestor::div[contains(@class, 'search-result-card')]")
                        title = "Unknown"
                        if card:
                            try:
                                title_elem = card.find_element(By.CSS_SELECTOR, "h3.title")
                                title = title_elem.text.strip()
                            except Exception:
                                pass
                        download_links.append((title, href))
                except Exception as e:
                    self.logger.warning(f"Error extracting link from button: {e}")

        if not download_links:
            with open(f"page_{page_num+1}_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            self.logger.warning(f"No download links found on page {page_num+1}. Page source saved for debugging.")

        return download_links

    def _build_filename(self, title: str, href: str) -> str:
        if '/downloads/' in href:
            url_filename = href.split('/downloads/')[-1].split('?')[0]
            url_filename = url_filename.replace('%20', ' ')
            return os.path.join(self.output_dir, url_filename)
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:100]
        return os.path.join(self.output_dir, f"{safe_title}.pdf")

    def _download_task(self, item):
        title, href = item
        filename = self._build_filename(title, href)
        success, skipped = self.download_file(href, filename)
        return item, filename, success, skipped

    def download_pdfs(self):
        """Download PDFs from NASA Technical Reports Server."""
        documents_downloaded, page_num = self.load_progress()
        existing_pdfs = [
            f for f in os.listdir(self.output_dir)
            if f.endswith('.pdf') and os.path.getsize(os.path.join(self.output_dir, f)) > 0
        ]
        self.logger.info(
            f"Resume: {len(existing_pdfs)} PDFs already on disk (will skip duplicates)"
        )
        self.logger.info(f"Starting download from document {documents_downloaded} (page {page_num+1})")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            driver = None
            try:
                driver = create_chrome_driver(headless=self.headless)

                while page_num < self.max_pages:
                    if self.max_docs is not None and documents_downloaded >= self.max_docs:
                        break

                    page_url = f"{self.base_url}&page=%7B%22size%22:100,%22from%22:{page_num*100}%7D&sort=%7B%22field%22:%22published%22,%22order%22:%22desc%22%7D"
                    self.logger.info(f"Processing page {page_num+1}, URL: {page_url}")

                    driver.get(page_url)
                    time.sleep(10)

                    download_links = self._extract_download_links(driver, page_num)

                    if download_links:
                        remaining = download_links
                        if documents_downloaded < self.start_from:
                            skip_n = min(self.start_from - documents_downloaded, len(remaining))
                            documents_downloaded += skip_n
                            remaining = remaining[skip_n:]

                        if self.max_docs is not None:
                            remaining_slots = self.max_docs - documents_downloaded
                            if remaining_slots <= 0:
                                break
                            remaining = remaining[:remaining_slots]

                        if remaining:
                            self.logger.info(
                                f"Downloading {len(remaining)} PDFs with {self.max_workers} workers..."
                            )
                            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                                futures = [executor.submit(self._download_task, item) for item in remaining]
                                for future in as_completed(futures):
                                    item, filename, success, skipped = future.result()
                                    title, href = item
                                    if success:
                                        documents_downloaded += 1
                                        action = "Skipped existing" if skipped else "Downloaded"
                                        self.logger.info(
                                            f"{action} {documents_downloaded}/{self.max_docs or 'all'}: "
                                            f"{os.path.basename(filename)}"
                                        )
                                    else:
                                        self.logger.warning(f"Failed to download: {title} -> {href}")

                            self.save_progress(documents_downloaded, page_num)

                    page_num += 1
                    self.save_progress(documents_downloaded, page_num)
                    time.sleep(1)

                break

            except Exception as e:
                self.logger.error(f"Error during download process: {e}")
                retry_count += 1
                self.logger.info(f"Retrying ({retry_count}/{max_retries})...")
                self.save_progress(documents_downloaded, page_num)
                time.sleep(10)

            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

        self.logger.info(f"Download complete. Downloaded {documents_downloaded} documents.")
        pdf_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pdf')]
        self.logger.info(f"Found {len(pdf_files)} PDF files in the output directory")
        return len(pdf_files)
    
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("nasa_pdf_processing.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # NASA Technical Reports Server URL - use the exact URL you provided
    nasa_url = "https://ntrs.nasa.gov/search?stiTypeDetails=Conference%20Paper"
    
    # Get the starting document number from command line or use default
    import sys
    start_from = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
    
    # Step 1: Download PDFs to NTRS_PDFS folder, resuming from where we left off
    PDFDownloader(nasa_url, output_dir="NTRS_PDFS_CONFERENCE_GLOBAL", max_docs=10000, start_from=start_from)
    

if __name__ == "__main__":
    main()