"""
Bulk Video Download Script
Usage: python bulk_download.py
"""

import requests
import os
import time
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
API_URL = "http://localhost:8000"  # Change to your deployed URL
DOWNLOAD_DIR = Path("downloaded_videos")
DOWNLOAD_DIR.mkdir(exist_ok=True)


class BulkDownloader:
    def __init__(self, api_url, download_dir):
        self.api_url = api_url
        self.download_dir = Path(download_dir)
        
    def get_all_videos(self):
        """Get list of all available videos"""
        try:
            response = requests.get(f"{self.api_url}/api/list-videos")
            response.raise_for_status()
            return response.json()["videos"]
        except Exception as e:
            print(f"Error fetching video list: {e}")
            return []
    
    def download_video(self, job_id, filename):
        """Download a single video"""
        try:
            print(f"Downloading: {filename}")
            
            response = requests.get(
                f"{self.api_url}/api/download/{job_id}",
                stream=True
            )
            response.raise_for_status()
            
            # Save file
            file_path = self.download_dir / filename
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {filename}")
            return {"status": "success", "filename": filename}
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return {"status": "error", "filename": filename, "error": str(e)}
    
    def download_all(self, max_workers=3):
        """Download all videos concurrently"""
        videos = self.get_all_videos()
        
        if not videos:
            print("No videos found to download")
            return
        
        print(f"\nFound {len(videos)} videos to download")
        print(f"Downloading to: {self.download_dir.absolute()}\n")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.download_video, 
                    video["job_id"], 
                    video["filename"]
                ): video for video in videos
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        
        print(f"\n{'='*50}")
        print(f"Download Complete!")
        print(f"✅ Success: {success}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Location: {self.download_dir.absolute()}")
        print(f"{'='*50}")
        
        return results
    
    def download_by_csv_result(self, csv_results):
        """Download videos from bulk generation results"""
        print(f"Downloading {len(csv_results)} videos from bulk generation\n")
        
        results = []
        for result in csv_results:
            if result["status"] == "success":
                job_id = result["job_id"]
                name = result["name"].replace(" ", "_")
                filename = f"{name}.mp4"
                
                download_result = self.download_video(job_id, filename)
                results.append(download_result)
            else:
                print(f"⚠️ Skipping {result['name']} (generation failed)")
        
        return results


def generate_and_download_bulk(video_path, csv_path, api_url=API_URL):
    """Generate bulk videos and download them all"""
    
    print("Starting bulk generation...")
    
    # Step 1: Upload and generate
    with open(video_path, 'rb') as video_file, \
         open(csv_path, 'rb') as csv_file:
        
        response = requests.post(
            f"{api_url}/api/generate-bulk",
            files={
                'video': video_file,
                'csv_file': csv_file
            },
            data={
                'duration': 10,
                'scroll_speed': 100,
                'video_size': 300,
                'video_position': 'bottom-right'
            }
        )
    
    if response.status_code != 200:
        print(f"Error generating videos: {response.text}")
        return
    
    results = response.json()
    print(f"\nGeneration complete: {results['total']} videos")
    
    # Step 2: Download all
    downloader = BulkDownloader(api_url, DOWNLOAD_DIR)
    downloader.download_by_csv_result(results['results'])


def download_all_videos(api_url=API_URL):
    """Download all available videos"""
    downloader = BulkDownloader(api_url, DOWNLOAD_DIR)
    downloader.download_all(max_workers=5)


def download_specific_videos(job_ids, api_url=API_URL):
    """Download specific videos by job IDs"""
    downloader = BulkDownloader(api_url, DOWNLOAD_DIR)
    
    for job_id in job_ids:
        filename = f"{job_id}.mp4"
        downloader.download_video(job_id, filename)


# =====================================================
# CLI Interface
# =====================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("BULK VIDEO DOWNLOADER")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python bulk_download.py all                    - Download all videos")
        print("  python bulk_download.py generate VIDEO CSV     - Generate & download")
        print("  python bulk_download.py ids ID1 ID2 ID3        - Download specific IDs")
        print("\nExamples:")
        print("  python bulk_download.py all")
        print("  python bulk_download.py generate my_video.mp4 leads.csv")
        print("  python bulk_download.py ids abc123 def456")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "all":
        # Download all available videos
        download_all_videos()
        
    elif command == "generate" and len(sys.argv) >= 4:
        # Generate bulk and download
        video_path = sys.argv[2]
        csv_path = sys.argv[3]
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        
        generate_and_download_bulk(video_path, csv_path)
        
    elif command == "ids" and len(sys.argv) >= 3:
        # Download specific job IDs
        job_ids = sys.argv[2:]
        download_specific_videos(job_ids)
        
    else:
        print("Invalid command. Use 'all', 'generate', or 'ids'")
        sys.exit(1)