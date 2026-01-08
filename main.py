Python 3.14.2 (v3.14.2:df793163d58, Dec  5 2025, 12:18:06) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
"""
Personalized Video Generator Backend
Requirements: pip install fastapi uvicorn python-multipart playwright opencv-python-headless moviepy pillow pandas
Also run: playwright install chromium
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import shutil
from pathlib import Path
from typing import Optional
import csv
import json
import os

# Configuration for production
PORT = int(os.getenv("PORT", 8000))

# Video processing
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Web scraping
from playwright.async_api import async_playwright

app = FastAPI(title="Personalized Video Generator API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir.mkdir(exist_ok=True)


class VideoGenerator:
    def __init__(self):
        self.screenshot_width = 1920
        self.screenshot_height = 1080
        
    async def capture_website_scroll(self, url: str, duration: int = 10, 
                                     scroll_speed: int = 100) -> list:
        """Capture scrolling website screenshots"""
        screenshots = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                viewport={"width": self.screenshot_width, 
                         "height": self.screenshot_height}
            )
            
            try:
                # Navigate to website
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)  # Let page settle
                
                # Get page height
                page_height = await page.evaluate("document.body.scrollHeight")
                
                # Calculate frames needed
                fps = 30
                total_frames = duration * fps
                scroll_increment = scroll_speed / fps
                
                current_scroll = 0
                for frame in range(total_frames):
                    # Take screenshot
                    screenshot_path = TEMP_DIR / f"frame_{frame:04d}.png"
                    await page.screenshot(path=str(screenshot_path))
                    screenshots.append(str(screenshot_path))
                    
                    # Scroll down
                    current_scroll += scroll_increment
                    if current_scroll > page_height - self.screenshot_height:
                        current_scroll = 0  # Loop back to top
                    
                    await page.evaluate(f"window.scrollTo(0, {current_scroll})")
                    await asyncio.sleep(1/fps)
                
            except Exception as e:
                print(f"Error capturing website: {e}")
                raise
            finally:
                await browser.close()
        
        return screenshots
    
    def create_circular_mask(self, size: tuple) -> np.ndarray:
        """Create a circular mask for the video overlay"""
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size[0], size[1]) // 2
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
    def overlay_video_circular(self, background_frame, video_frame, 
                              position: tuple, size: tuple) -> np.ndarray:
        """Overlay video in a circular frame on background"""
        # Resize video frame
        video_resized = cv2.resize(video_frame, size)
        
        # Create circular mask
        mask = self.create_circular_mask(size)
        
        # Apply mask to video frame
        video_circular = cv2.bitwise_and(video_resized, video_resized, mask=mask)
        
        # Create border (glow effect)
        border_mask = cv2.circle(
            np.zeros((size[1], size[0], 3), dtype=np.uint8),
            (size[0]//2, size[1]//2),
            min(size[0], size[1])//2,
            (0, 255, 255),  # Cyan border
            8
        )
        
        # Composite onto background
        x, y = position
        h, w = size[1], size[0]
        
        # Ensure we don't go out of bounds
        if x + w > background_frame.shape[1]:
            w = background_frame.shape[1] - x
        if y + h > background_frame.shape[0]:
            h = background_frame.shape[0] - y
            
        roi = background_frame[y:y+h, x:x+w]
        
        # Blend circular video
        mask_3channel = cv2.cvtColor(mask[:h, :w], cv2.COLOR_GRAY2BGR)
        video_area = cv2.bitwise_and(video_circular[:h, :w], mask_3channel)
        background_area = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3channel))
        blended = cv2.add(video_area, background_area)
        
        # Add border
        blended = cv2.add(blended, border_mask[:h, :w])
        
        background_frame[y:y+h, x:x+w] = blended
        
        return background_frame
    
    async def generate_video(self, video_path: str, website_url: str,
                           output_path: str, duration: int = 10,
                           scroll_speed: int = 100,
                           video_size: tuple = (300, 300),
                           video_position: str = "bottom-right") -> str:
        """Generate the final personalized video"""
        
        print(f"Starting video generation for {website_url}")
        
        # Step 1: Capture website scrolling
        print("Capturing website screenshots...")
        screenshots = await self.capture_website_scroll(
            website_url, duration, scroll_speed
        )
        
        # Step 2: Load user video
        print("Loading user video...")
        video_clip = VideoFileClip(video_path)
        video_duration = min(video_clip.duration, duration)
        
        # Step 3: Process frames
        print("Compositing video...")
        output_frames = []
        fps = 30
        
        # Calculate position
        if video_position == "bottom-right":
            pos_x = self.screenshot_width - video_size[0] - 50
            pos_y = self.screenshot_height - video_size[1] - 50
        elif video_position == "bottom-left":
            pos_x = 50
            pos_y = self.screenshot_height - video_size[1] - 50
        elif video_position == "top-right":
            pos_x = self.screenshot_width - video_size[0] - 50
            pos_y = 50
        else:  # top-left
            pos_x = 50
            pos_y = 50
        
        for i, screenshot_path in enumerate(screenshots):
            # Load background
            bg_frame = cv2.imread(screenshot_path)
            
            # Get corresponding video frame
            time = i / fps
            if time < video_duration:
                video_frame = video_clip.get_frame(time)
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                
                # Overlay video
                final_frame = self.overlay_video_circular(
                    bg_frame, video_frame, (pos_x, pos_y), video_size
                )
            else:
                final_frame = bg_frame
            
            output_frames.append(final_frame)
        
        # Step 4: Create final video
        print("Creating final video file...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                             (self.screenshot_width, self.screenshot_height))
        
        for frame in output_frames:
            out.write(frame)
        
        out.release()
        video_clip.close()
        
        # Cleanup temp files
        for screenshot in screenshots:
            try:
                os.remove(screenshot)
            except:
                pass
        
        print(f"Video generated successfully: {output_path}")
        return output_path


# Initialize generator
generator = VideoGenerator()


@app.get("/")
async def root():
    return {"message": "Personalized Video Generator API", "status": "running"}


@app.post("/api/generate-video")
async def generate_video(
    video: UploadFile = File(...),
    website_url: str = Form(...),
    duration: int = Form(10),
    scroll_speed: int = Form(100),
    video_size: int = Form(300),
    video_position: str = Form("bottom-right")
):
    """
    Generate a personalized video
    
    Parameters:
    - video: Video file (MP4, MOV, etc.)
    - website_url: Target website URL
    - duration: Video duration in seconds (default: 10)
    - scroll_speed: Scroll speed in pixels (default: 100)
    - video_size: Size of circular video overlay (default: 300)
    - video_position: Position of video (bottom-right, bottom-left, top-right, top-left)
    """
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded video
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Output path
    output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")
    
    try:
        # Generate video
        result = await generator.generate_video(
            video_path=str(video_path),
            website_url=website_url,
            output_path=output_path,
            duration=duration,
            scroll_speed=scroll_speed,
            video_size=(video_size, video_size),
            video_position=video_position
        )
        
        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "download_url": f"/api/download/{job_id}",
            "message": "Video generated successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-bulk")
async def generate_bulk(
    video: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    duration: int = Form(10),
    scroll_speed: int = Form(100),
    video_size: int = Form(300),
    video_position: str = Form("bottom-right")
):
    """
    Generate videos for multiple prospects from CSV
    
    CSV format:
    name,website_url
    John Doe,https://example.com
    Jane Smith,https://another-site.com
    """
    
    batch_id = str(uuid.uuid4())
    
    # Save video
    video_path = UPLOAD_DIR / f"{batch_id}_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Parse CSV
    csv_content = await csv_file.read()
    csv_text = csv_content.decode('utf-8')
    csv_reader = csv.DictReader(csv_text.splitlines())
    
    results = []
    
    for row in csv_reader:
        name = row.get('name', 'Unknown')
        website_url = row.get('website_url')
        
        if not website_url:
            continue
        
        job_id = str(uuid.uuid4())
        output_path = str(OUTPUT_DIR / f"{job_id}_{name.replace(' ', '_')}.mp4")
        
        try:
            await generator.generate_video(
                video_path=str(video_path),
                website_url=website_url,
                output_path=output_path,
                duration=duration,
                scroll_speed=scroll_speed,
                video_size=(video_size, video_size),
                video_position=video_position
            )
            
            results.append({
                "name": name,
                "website": website_url,
                "status": "success",
                "job_id": job_id,
                "download_url": f"/api/download/{job_id}"
            })
        except Exception as e:
            results.append({
                "name": name,
                "website": website_url,
                "status": "error",
                "error": str(e)
            })
    
    return JSONResponse({
        "batch_id": batch_id,
        "total": len(results),
        "results": results
    })


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Download generated video"""
    
    # Find video file
    video_files = list(OUTPUT_DIR.glob(f"{job_id}*"))
    
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_files[0],
        media_type="video/mp4",
        filename=video_files[0].name
    )

... 
... @app.get("/api/list-videos")
... async def list_videos():
...     """List all generated videos"""
...     
...     videos = []
...     for video_path in OUTPUT_DIR.glob("*.mp4"):
...         videos.append({
...             "filename": video_path.name,
...             "job_id": video_path.stem.split('_')[0],
...             "size_mb": round(video_path.stat().st_size / 1024 / 1024, 2)
...         })
...     
...     return {"videos": videos}
... 
... 
... @app.delete("/api/cleanup")
... async def cleanup():
...     """Clean up old files"""
...     
...     deleted = {"uploads": 0, "outputs": 0, "temp": 0}
...     
...     for file in UPLOAD_DIR.glob("*"):
...         file.unlink()
...         deleted["uploads"] += 1
...     
...     for file in OUTPUT_DIR.glob("*"):
...         file.unlink()
...         deleted["outputs"] += 1
...     
...     for file in TEMP_DIR.glob("*"):
...         file.unlink()
...         deleted["temp"] += 1
...     
...     return {"message": "Cleanup complete", "deleted": deleted}
... 
... 
... if __name__ == "__main__":
   	import uvicorn
    	uvicorn.run(app, host="0.0.0.0", port=PORT)
