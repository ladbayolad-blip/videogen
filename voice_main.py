"""
Voice-Only Personalized Video Generator
Upload audio file instead of video - generates waveform animation + scrolling website

Additional requirements:
pip install librosa pydub soundfile
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import uuid
import shutil
from pathlib import Path
import numpy as np
import cv2

# Audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment

# Web scraping (reuse from previous)
from playwright.async_api import async_playwright
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageSequenceClip

app = FastAPI(title="Voice-Only Video Generator API")

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


class VoiceVideoGenerator:
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
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)
                
                page_height = await page.evaluate("document.body.scrollHeight")
                
                fps = 30
                total_frames = duration * fps
                scroll_increment = scroll_speed / fps
                
                current_scroll = 0
                for frame in range(total_frames):
                    screenshot_path = TEMP_DIR / f"frame_{frame:04d}.png"
                    await page.screenshot(path=str(screenshot_path))
                    screenshots.append(str(screenshot_path))
                    
                    current_scroll += scroll_increment
                    if current_scroll > page_height - self.screenshot_height:
                        current_scroll = 0
                    
                    await page.evaluate(f"window.scrollTo(0, {current_scroll})")
                    await asyncio.sleep(1/fps)
                
            finally:
                await browser.close()
        
        return screenshots
    
    def create_waveform_visualization(self, audio_path: str, size: tuple, 
                                     fps: int = 30) -> list:
        """Create animated waveform visualization from audio"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Generate frames
        frames = []
        total_frames = int(duration * fps)
        samples_per_frame = len(y) // total_frames
        
        width, height = size
        
        for i in range(total_frames):
            # Create frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get audio segment for this frame
            start_idx = i * samples_per_frame
            end_idx = start_idx + samples_per_frame
            segment = y[start_idx:end_idx] if end_idx < len(y) else y[start_idx:]
            
            # Draw waveform
            if len(segment) > 0:
                # Downsample to width
                step = max(1, len(segment) // width)
                waveform = segment[::step][:width]
                
                # Normalize
                waveform = waveform / np.max(np.abs(waveform) + 1e-6)
                
                # Draw
                for x, amp in enumerate(waveform):
                    y_pos = int(height / 2 + amp * height / 2.5)
                    cv2.line(frame, (x, height // 2), (x, y_pos), 
                            (0, 255, 255), 2)  # Cyan color
            
            # Add circular border
            center = (width // 2, height // 2)
            radius = min(width, height) // 2 - 10
            cv2.circle(frame, center, radius, (0, 255, 255), 8)
            
            # Add audio icon in center
            cv2.putText(frame, "🎤", (width // 2 - 40, height // 2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            frames.append(frame)
        
        return frames
    
    def create_spectrum_visualization(self, audio_path: str, size: tuple,
                                     fps: int = 30, style: str = "bars") -> list:
        """Create animated frequency spectrum visualization"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        frames = []
        total_frames = int(duration * fps)
        hop_length = 512
        
        # Compute spectrogram
        D = librosa.stft(y, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        width, height = size
        
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get spectrum for this frame
            frame_idx = int(i * S_db.shape[1] / total_frames)
            if frame_idx >= S_db.shape[1]:
                frame_idx = S_db.shape[1] - 1
            
            spectrum = S_db[:, frame_idx]
            
            if style == "bars":
                # Bar visualization
                num_bars = 50
                bar_width = width // num_bars
                spectrum_resampled = np.interp(
                    np.linspace(0, len(spectrum), num_bars),
                    np.arange(len(spectrum)),
                    spectrum
                )
                
                # Normalize
                spectrum_norm = (spectrum_resampled - spectrum_resampled.min()) / \
                               (spectrum_resampled.max() - spectrum_resampled.min() + 1e-6)
                
                for j, val in enumerate(spectrum_norm):
                    bar_height = int(val * height * 0.8)
                    x1 = j * bar_width
                    x2 = x1 + bar_width - 2
                    y1 = height - bar_height
                    
                    # Gradient color based on height
                    color = (
                        int(255 * val),
                        int(255 * (1 - val)),
                        255
                    )
                    
                    cv2.rectangle(frame, (x1, y1), (x2, height), color, -1)
            
            elif style == "circular":
                # Circular spectrum
                center = (width // 2, height // 2)
                num_lines = 100
                max_radius = min(width, height) // 2 - 20
                
                spectrum_resampled = np.interp(
                    np.linspace(0, len(spectrum), num_lines),
                    np.arange(len(spectrum)),
                    spectrum
                )
                
                spectrum_norm = (spectrum_resampled - spectrum_resampled.min()) / \
                               (spectrum_resampled.max() - spectrum_resampled.min() + 1e-6)
                
                for j, val in enumerate(spectrum_norm):
                    angle = (j / num_lines) * 2 * np.pi
                    line_length = int(val * max_radius * 0.5)
                    
                    x1 = int(center[0] + np.cos(angle) * max_radius * 0.5)
                    y1 = int(center[1] + np.sin(angle) * max_radius * 0.5)
                    x2 = int(center[0] + np.cos(angle) * (max_radius * 0.5 + line_length))
                    y2 = int(center[1] + np.sin(angle) * (max_radius * 0.5 + line_length))
                    
                    color = (0, int(255 * val), 255)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add border
            cv2.circle(frame, (width // 2, height // 2), 
                      min(width, height) // 2 - 5, (0, 255, 255), 8)
            
            frames.append(frame)
        
        return frames
    
    def overlay_visualization(self, background_frame, vis_frame,
                            position: tuple) -> np.ndarray:
        """Overlay visualization on background"""
        x, y = position
        h, w = vis_frame.shape[:2]
        
        # Ensure within bounds
        if x + w > background_frame.shape[1]:
            w = background_frame.shape[1] - x
        if y + h > background_frame.shape[0]:
            h = background_frame.shape[0] - y
        
        # Create mask for non-black pixels
        gray = cv2.cvtColor(vis_frame[:h, :w], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract ROI
        roi = background_frame[y:y+h, x:x+w]
        
        # Black out area in ROI
        bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
        
        # Take only region of visualization
        vis_part = cv2.bitwise_and(vis_frame[:h, :w], vis_frame[:h, :w], mask=mask)
        
        # Blend
        dst = cv2.add(bg_part, vis_part)
        background_frame[y:y+h, x:x+w] = dst
        
        return background_frame
    
    async def generate_voice_video(self, audio_path: str, website_url: str,
                                  output_path: str, visualization_style: str = "waveform",
                                  video_size: tuple = (300, 300),
                                  video_position: str = "bottom-right") -> str:
        """Generate video with audio visualization + scrolling website"""
        
        print(f"Starting voice video generation for {website_url}")
        
        # Get audio duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000  # Convert to seconds
        duration = min(duration, 30)  # Cap at 30 seconds
        
        # Step 1: Capture website
        print("Capturing website...")
        screenshots = await self.capture_website_scroll(
            website_url, int(duration), 100
        )
        
        # Step 2: Create audio visualization
        print(f"Creating {visualization_style} visualization...")
        if visualization_style == "waveform":
            vis_frames = self.create_waveform_visualization(
                audio_path, video_size, fps=30
            )
        elif visualization_style == "spectrum":
            vis_frames = self.create_spectrum_visualization(
                audio_path, video_size, fps=30, style="bars"
            )
        elif visualization_style == "circular":
            vis_frames = self.create_spectrum_visualization(
                audio_path, video_size, fps=30, style="circular"
            )
        else:
            vis_frames = self.create_waveform_visualization(
                audio_path, video_size, fps=30
            )
        
        # Step 3: Composite frames
        print("Compositing video...")
        output_frames = []
        
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
            bg_frame = cv2.imread(screenshot_path)
            
            # Get corresponding visualization frame
            vis_idx = min(i, len(vis_frames) - 1)
            vis_frame = vis_frames[vis_idx]
            
            # Overlay
            final_frame = self.overlay_visualization(
                bg_frame, vis_frame, (pos_x, pos_y)
            )
            
            output_frames.append(final_frame)
        
        # Step 4: Create video
        print("Creating video file...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = str(TEMP_DIR / f"temp_{uuid.uuid4()}.mp4")
        out = cv2.VideoWriter(temp_video, fourcc, 30,
                             (self.screenshot_width, self.screenshot_height))
        
        for frame in output_frames:
            out.write(frame)
        
        out.release()
        
        # Step 5: Add audio
        print("Adding audio...")
        video_clip = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(audio_path)
        
        # Trim audio to video length
        audio_clip = audio_clip.subclip(0, min(audio_clip.duration, video_clip.duration))
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        os.remove(temp_video)
        
        for screenshot in screenshots:
            try:
                os.remove(screenshot)
            except:
                pass
        
        print(f"Voice video generated: {output_path}")
        return output_path


# Initialize generator
voice_generator = VoiceVideoGenerator()


@app.post("/api/generate-voice-video")
async def generate_voice_video(
    audio: UploadFile = File(...),
    website_url: str = Form(...),
    visualization_style: str = Form("waveform"),
    video_size: int = Form(300),
    video_position: str = Form("bottom-right")
):
    """
    Generate personalized video with voice/audio
    
    Parameters:
    - audio: Audio file (MP3, WAV, M4A, etc.)
    - website_url: Target website
    - visualization_style: waveform, spectrum, or circular
    - video_size: Size of visualization (default: 300)
    - video_position: bottom-right, bottom-left, top-right, top-left
    """
    
    job_id = str(uuid.uuid4())
    
    # Save audio
    audio_path = UPLOAD_DIR / f"{job_id}_{audio.filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")
    
    try:
        await voice_generator.generate_voice_video(
            audio_path=str(audio_path),
            website_url=website_url,
            output_path=output_path,
            visualization_style=visualization_style,
            video_size=(video_size, video_size),
            video_position=video_position
        )
        
        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "download_url": f"/api/download/{job_id}",
            "message": "Voice video generated successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Download generated video"""
    video_files = list(OUTPUT_DIR.glob(f"{job_id}*"))
    
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_files[0],
        media_type="video/mp4",
        filename=video_files[0].name
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)