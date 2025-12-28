#!/usr/bin/env python3
"""
Advanced downloader for Visible Human MRI DICOM Dataset
Features: parallel downloads, resume capability, detailed progress tracking
"""

import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import argparse
import hashlib
import json


class AdvancedVisibleHumanDownloader:
    def __init__(self, base_url, output_dir="visible_human_mri", 
                 max_workers=4, resume=True):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.resume = resume
        
        # Track progress
        self.manifest_file = self.output_dir / "download_manifest.json"
        self.manifest = self.load_manifest()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research/Educational Download Bot)'
        })
        
    def load_manifest(self):
        """Load download manifest for resume capability"""
        if self.resume and self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {"downloaded": {}, "failed": {}}
    
    def save_manifest(self):
        """Save download manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def parse_directory(self, url):
        """Parse Apache-style directory listing"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            items = {'files': [], 'dirs': []}
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and not href.startswith('?') and href != '../':
                    if href.endswith('/'):
                        items['dirs'].append(href.rstrip('/'))
                    else:
                        items['files'].append(href)
            
            return items
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return {'files': [], 'dirs': []}
    
    def collect_all_files(self, url, base_path=""):
        """Recursively collect all file URLs"""
        all_files = []
        items = self.parse_directory(url)
        
        # Add files in current directory
        for filename in items['files']:
            file_url = urljoin(url, filename)
            relative_path = os.path.join(base_path, filename)
            all_files.append((file_url, relative_path))
        
        # Recursively process subdirectories
        for dirname in items['dirs']:
            subdir_url = urljoin(url, dirname + '/')
            subdir_path = os.path.join(base_path, dirname)
            all_files.extend(self.collect_all_files(subdir_url, subdir_path))
        
        return all_files
    
    def download_file(self, url, relative_path):
        """Download a single file"""
        local_path = self.output_dir / relative_path
        
        # Check if already downloaded
        if str(relative_path) in self.manifest['downloaded']:
            return {'status': 'skipped', 'path': relative_path}
        
        try:
            # Create directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with streaming
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Write file
            with open(local_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # Update manifest
            self.manifest['downloaded'][str(relative_path)] = {
                'url': url,
                'size': local_path.stat().st_size,
                'timestamp': time.time()
            }
            
            return {'status': 'success', 'path': relative_path, 'size': total_size}
            
        except Exception as e:
            self.manifest['failed'][str(relative_path)] = str(e)
            return {'status': 'failed', 'path': relative_path, 'error': str(e)}
    
    def download_all(self):
        """Download all files with parallel processing"""
        print(f"Collecting file list from: {self.base_url}")
        all_files = self.collect_all_files(self.base_url)
        
        print(f"\nFound {len(all_files)} files")
        
        # Filter out already downloaded if resuming
        if self.resume:
            files_to_download = [
                (url, path) for url, path in all_files 
                if str(path) not in self.manifest['downloaded']
            ]
            print(f"Already downloaded: {len(all_files) - len(files_to_download)}")
            print(f"To download: {len(files_to_download)}")
        else:
            files_to_download = all_files
        
        if not files_to_download:
            print("\nAll files already downloaded!")
            return
        
        print(f"\nStarting download with {self.max_workers} parallel workers...")
        print("-" * 60)
        
        # Download with progress bar
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_file, url, path): (url, path)
                for url, path in files_to_download
            }
            
            with tqdm(total=len(files_to_download), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['status'] == 'success':
                        success_count += 1
                        pbar.set_postfix({'✓': success_count, '✗': failed_count})
                    elif result['status'] == 'failed':
                        failed_count += 1
                        pbar.set_postfix({'✓': success_count, '✗': failed_count})
                        tqdm.write(f"Failed: {result['path']} - {result.get('error', 'Unknown error')}")
                    else:
                        skipped_count += 1
                    
                    pbar.update(1)
                    
                    # Save manifest periodically
                    if (success_count + failed_count) % 10 == 0:
                        self.save_manifest()
        
        # Final save
        self.save_manifest()
        
        # Summary
        print("\n" + "=" * 60)
        print("Download Summary:")
        print(f"  Successfully downloaded: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped (already exists): {skipped_count}")
        print(f"  Total files: {len(all_files)}")
        print(f"\nFiles saved to: {self.output_dir.absolute()}")
        print(f"Manifest saved to: {self.manifest_file}")
        print("=" * 60)
        
        if failed_count > 0:
            print("\nFailed downloads saved in manifest. Run again to retry.")


def main():
    parser = argparse.ArgumentParser(
        description='Download MRI DICOM slices from Visible Human Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Download with defaults
  %(prog)s -o my_data -w 8                    # 8 parallel downloads
  %(prog)s --no-resume                         # Fresh download
        """
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='visible_human_mri',
        help='Output directory (default: visible_human_mri)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel downloads (default: 4)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from previous download'
    )
    parser.add_argument(
        '--url', '-u',
        default='https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Additional-Head-Images/MR_CT_DICOM/MRI/',
        help='Base URL (default: Visible Human MRI)'
    )
    
    args = parser.parse_args()
    
    downloader = AdvancedVisibleHumanDownloader(
        base_url=args.url,
        output_dir=args.output_dir,
        max_workers=args.workers,
        resume=not args.no_resume
    )
    
    try:
        downloader.download_all()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        downloader.save_manifest()
        print("Progress saved. Run again to resume.")


if __name__ == "__main__":
    main()
