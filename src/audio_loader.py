"""
Download and load audio files from URLs or local paths.

Supports:
- Local files (WAV, MP3, FLAC, etc.)
- Direct HTTP URLs to audio files
- YouTube videos (via yt-dlp with bundled FFmpeg)
"""

import os
import glob
import shutil
import tempfile
from typing import Tuple
import requests
import librosa
import numpy as np


def _ensure_ffmpeg() -> bool:
    """Ensure FFmpeg is available, using bundled binaries if needed.

    Returns:
        True if FFmpeg is available (system or bundled), False otherwise.
    """
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        return True

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        return shutil.which('ffmpeg') is not None
    except ImportError:
        return False
    except Exception:
        return False


class AudioLoader:
    """Load audio from various sources."""

    def __init__(self, temp_dir: str = None):
        """Initialize audio loader.

        Args:
            temp_dir: Directory for temporary files. Uses system temp if None.
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
        if os.path.isfile(source):
            return self._load_file(source, sr)

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
        if 'youtube.com' in url or 'youtu.be' in url:
            return self._load_youtube(url, sr)
        return self._load_direct_url(url, sr)

    def _load_youtube(self, url: str, sr: int) -> Tuple[np.ndarray, int]:
        """Download and load audio from YouTube."""
        try:
            import yt_dlp
        except ImportError:
            raise ValueError(
                "yt-dlp not installed. Run: pip install yt-dlp"
            )

        if not _ensure_ffmpeg():
            raise ValueError(
                "FFmpeg not available. Install bundled FFmpeg with:\n"
                "  pip install static-ffmpeg\n"
                "Or install system-wide:\n"
                "  Windows: choco install ffmpeg\n"
                "  macOS:   brew install ffmpeg\n"
                "  Linux:   sudo apt-get install ffmpeg"
            )

        temp_base = os.path.join(self.temp_dir, "youtube_audio")
        self._cleanup_temp_files(temp_base)

        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_base + '.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
            }

            print(f"[DOWNLOAD] YouTube: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

            wav_file = f"{temp_base}.wav"
            if not os.path.exists(wav_file):
                candidates = glob.glob(f"{temp_base}*")
                if not candidates:
                    raise ValueError("Downloaded audio file not found")
                wav_file = sorted(candidates)[-1]

            self.cleanup_paths.append(wav_file)
            return self._load_file(wav_file, sr)

        except Exception as e:
            raise ValueError(f"Failed to load YouTube: {str(e)}")

    def _cleanup_temp_files(self, base_path: str) -> None:
        """Remove leftover temporary files matching base path."""
        for f in glob.glob(f"{base_path}*"):
            try:
                os.remove(f)
            except Exception:
                pass

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
