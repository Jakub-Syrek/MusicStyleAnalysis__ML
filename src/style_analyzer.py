"""
Extract musical style features from audio files.
"""

import librosa
import numpy as np
from typing import Dict, Any


class StyleAnalyzer:
    """
    Analyzes musical characteristics from audio files.

    Extracts tempo, key, instrumentation hints, and emotional characteristics
    for use in music generation.
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize style analyzer.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Analyze musical style from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing style features
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return self.analyze_from_audio(y, sr)
        except Exception as error:
            raise ValueError(f"Failed to analyze audio file {audio_path}: {str(error)}")

    def analyze_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze musical style from audio data.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary containing style features
        """
        features = {
            "tempo": self._extract_tempo(y, sr),
            "loudness": self._extract_loudness(y),
            "spectral_centroid": self._extract_spectral_features(y, sr),
            "zero_crossing_rate": self._extract_zcr(y),
            "mfcc": self._extract_mfcc(y, sr),
        }
        return features

    def _extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo from audio.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Estimated tempo in BPM
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        result = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = result[0] if isinstance(result, tuple) else result
        return float(tempo)

    def _extract_loudness(self, y: np.ndarray) -> float:
        """Extract overall loudness.

        Args:
            y: Audio time series

        Returns:
            RMS loudness value
        """
        rms = librosa.feature.rms(y=y)[0]
        return float(np.mean(rms))

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral centroid (brightness indicator).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Mean spectral centroid
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        return float(np.mean(spectral_centroids))

    def _extract_zcr(self, y: np.ndarray) -> float:
        """Extract zero crossing rate (noisiness indicator).

        Args:
            y: Audio time series

        Returns:
            Mean zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        return float(np.mean(zcr))

    def _extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> list:
        """Extract Mel-frequency cepstral coefficients.

        Args:
            y: Audio time series
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients (default: 13)

        Returns:
            Mean MFCC values
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return [float(np.mean(m)) for m in mfcc]
