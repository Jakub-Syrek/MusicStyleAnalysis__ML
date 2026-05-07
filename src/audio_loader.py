"""
Download and load audio files from URLs or local paths.

Supports:
- Local files (WAV, MP3, FLAC, etc.)
- Direct HTTP URLs to audio files
- YouTube videos (via yt-dlp)
"""

import os
import glob
import tempfile
import shutil
from pathlib import Path
from typing import Tuple
import requests
import librosa
import numpy as np


class AudioLoader:
    """Load audio from various sources."""

    def __init__(self, temp_dir: str = None):
        """Initialize audio loader.

        Args:
            temp_dir: Directory for temporary files. Uses /tmp if None.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.cleanup_paths = []

    def load(self, source: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load audio from file, URL, or video.

        Args:
            source: File path, HTTP URL, or YouTube URL
            sr: Sample rate (default 16000)

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            ValueError: If source is invalid or loading fails
        """
        # Try local file first
        if os.path.isfile(source):
            return self._load_file(source, sr)

        # Check if URL
        if source.startswith(('http://', 'https://')):
            return self._load_url(source, sr)

        raise ValueError(f"Invalid source: {source}")

    def _load_file(self, file_path: str, sr: int) -> Tuple[np.ndarray, int]:
        """Load audio from local file."""
        try:
            y, sr_loaded = librosa.load(file_path, sr=sr)
            print(f"[OK] Loaded: {file_path}")
            return y, sr_loaded
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {str(e)}")

    def _load_url(self, url: str, sr: int) -> Tuple[np.ndarray, int]:
        """Load audio from URL."""
        # YouTube URL
        if 'youtube.com' in url or 'youtu.be' in url:
            return self._load_youtube(url, sr)

        # Direct audio file URL
        return self._load_direct_url(url, sr)

    def _load_youtube(self, url: str, sr: int) -> Tuple[np.ndarray, int]:
        """Download and load audio from YouTube."""
        try:
            import yt_dlp
        except ImportError:
            raise ValueError(
                "yt-dlp not installed. Run: pip install yt-dlp"
            )

        temp_base = os.path.join(self.temp_dir, "youtube_audio")

        try:
            has_ffmpeg = self._check_ffmpeg()

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_base,
                'quiet': True,
                'no_warnings': True,
            }

            if has_ffmpeg:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }]

            print(f"[DOWNLOAD] YouTube: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

            temp_file = None
            for ext in ['wav', 'webm', 'm4a', 'mp4', 'mkv', '']:
                candidate = f"{temp_base}.{ext}" if ext else temp_base
                if os.path.exists(candidate):
                    temp_file = candidate
                    break

            if not temp_file:
                candidates = glob.glob(f"{temp_base}*")
                if candidates:
                    temp_file = sorted(candidates)[-1]

            if not temp_file or not os.path.exists(temp_file):
                error_msg = (
                    "Failed to locate downloaded audio file. "
                    "YouTube support requires FFmpeg for audio conversion.\n\n"
                    "Install FFmpeg:\n"
                    "  Windows: choco install ffmpeg\n"
                    "  macOS:   brew install ffmpeg\n"
                    "  Linux:   sudo apt-get install ffmpeg\n"
                    "  Or download from: https://ffmpeg.org/download.html"
                )
                raise ValueError(error_msg)

            self.cleanup_paths.append(temp_file)
            try:
                return self._load_file(temp_file, sr)
            except ValueError as e:
                if "Failed to load" in str(e) and not has_ffmpeg:
                    error_msg = (
                        f"Cannot load downloaded audio format. "
                        f"YouTube support requires FFmpeg for audio conversion.\n\n"
                        f"Install FFmpeg:\n"
                        f"  Windows: choco install ffmpeg\n"
                        f"  macOS:   brew install ffmpeg\n"
                        f"  Linux:   sudo apt-get install ffmpeg\n"
                        f"  Or download from: https://ffmpeg.org/download.html"
                    )
                    raise ValueError(error_msg)
                raise

        except Exception as e:
            raise ValueError(f"Failed to load YouTube: {str(e)}")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available on the system."""
        import subprocess
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=2,
                check=False
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _load_direct_url(self, url: str, sr: int) -> Tuple[np.ndarray, int]:
        """Download and load audio from direct HTTP URL."""
        temp_file = os.path.join(
            self.temp_dir,
            f"audio_{hash(url) % 10000}.wav"
        )
        self.cleanup_paths.append(temp_file)

        try:
            print(f"[DOWNLOAD] URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(temp_file, 'wb') as f:
                f.write(response.content)

            return self._load_file(temp_file, sr)

        except Exception as e:
            raise ValueError(f"Failed to download {url}: {str(e)}")

    def cleanup(self):
        """Remove temporary files."""
        for path in self.cleanup_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        self.cleanup_paths.clear()

    def __del__(self):
        """Cleanup on object deletion."""
        self.cleanup()
