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
        """
        Initialize style analyzer.

        @param {number} sample_rate - Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze musical style from audio file.

        @param {string} audio_path - Path to audio file
        @returns {object} Dictionary containing style features
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            features = {
                "tempo": self._extract_tempo(y, sr),
                "loudness": self._extract_loudness(y),
                "spectral_centroid": self._extract_spectral_features(y, sr),
                "zero_crossing_rate": self._extract_zcr(y),
                "mfcc": self._extract_mfcc(y, sr),
            }

            return features
        except Exception as error:
            raise ValueError(f"Failed to analyze audio file {audio_path}: {str(error)}")

    def _extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """
        Extract tempo from audio.

        @param {array} y - Audio time series
        @param {number} sr - Sample rate
        @returns {number} Estimated tempo in BPM
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        result = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = result[0] if isinstance(result, tuple) else result
        return float(tempo)

    def _extract_loudness(self, y: np.ndarray) -> float:
        """
        Extract overall loudness.

        @param {array} y - Audio time series
        @returns {number} RMS loudness value
        """
        rms = librosa.feature.rms(y=y)[0]
        return float(np.mean(rms))

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> float:
        """
        Extract spectral centroid (brightness indicator).

        @param {array} y - Audio time series
        @param {number} sr - Sample rate
        @returns {number} Mean spectral centroid
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        return float(np.mean(spectral_centroids))

    def _extract_zcr(self, y: np.ndarray) -> float:
        """
        Extract zero crossing rate (noisiness indicator).

        @param {array} y - Audio time series
        @returns {number} Mean zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        return float(np.mean(zcr))

    def _extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> list:
        """
        Extract Mel-frequency cepstral coefficients.

        @param {array} y - Audio time series
        @param {number} sr - Sample rate
        @param {number} n_mfcc - Number of MFCC coefficients
        @returns {array} Mean MFCC values
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return [float(np.mean(m)) for m in mfcc]
