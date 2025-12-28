#!/usr/bin/env python3
"""
Download all MRI DICOM slices from the Visible Human Dataset
https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_DICOM/MRI/
"""

import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import argparse


class VisibleHumanDownloader:
    def __init__(self, base_url, output_dir="visible_human_mri"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.downloaded_files = 0
        
    def parse_directory(self, url):
        """Parse an Apache-style directory listing"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and not href.startswith('?') and href != '../':
                    links.append(href)
            
            return links
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return []
    
    def download_file(self, url, local_path):
        """Download a single file with progress bar"""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=local_path.name, leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            self.downloaded_files += 1
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def crawl_and_download(self, url, local_dir):
        """Recursively crawl directories and download files"""
        print(f"\nCrawling: {url}")
        
        links = self.parse_directory(url)
        
        for link in links:
            full_url = urljoin(url, link)
            
            # If it's a directory (ends with /)
            if link.endswith('/'):
                subdir_name = link.rstrip('/')
                subdir_path = local_dir / subdir_name
                subdir_path.mkdir(parents=True, exist_ok=True)
                
                # Recursively download subdirectory
                self.crawl_and_download(full_url, subdir_path)
            
            # If it's a file
            else:
                local_file = local_dir / link
                
                # Skip if already downloaded
                if local_file.exists():
                    print(f"Skipping (exists): {link}")
                    continue
                
                print(f"Downloading: {link}")
                self.download_file(full_url, local_file)
                
                # Be nice to the server
                time.sleep(0.5)
    
    def download_all(self):
        """Start the download process"""
        print(f"Starting download from: {self.base_url}")
        print(f"Saving to: {self.output_dir.absolute()}")
        print("-" * 60)
        
        self.crawl_and_download(self.base_url, self.output_dir)
        
        print("\n" + "=" * 60)
        print(f"Download complete! Total files downloaded: {self.downloaded_files}")
        print(f"Files saved to: {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Download MRI DICOM slices from Visible Human Dataset'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='visible_human_mri',
        help='Output directory for downloaded files (default: visible_human_mri)'
    )
    parser.add_argument(
        '--url', '-u',
        default='https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_DICOM/MRI/',
        help='Base URL to download from'
    )
    
    args = parser.parse_args()
    
    downloader = VisibleHumanDownloader(
        base_url=args.url,
        output_dir=args.output_dir
    )
    
    downloader.download_all()


if __name__ == "__main__":
    main()
