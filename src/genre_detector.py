"""
Detect music genre and rhythm characteristics from audio features.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import librosa


class GenreDetector:
  """
  Classify music genre based on audio features.

  Uses tempo, loudness, spectral characteristics, and rhythm patterns
  to classify into basic genres.
  """

  def __init__(self):
    """Initialize genre detector."""
    self.genres = {
      "electronic": {"tempo_range": (110, 140), "loudness_min": 0.4},
      "dance": {"tempo_range": (120, 135), "loudness_min": 0.5},
      "pop": {"tempo_range": (90, 130), "loudness_min": 0.3},
      "rock": {"tempo_range": (95, 140), "loudness_min": 0.4},
      "jazz": {"tempo_range": (70, 110), "loudness_min": 0.2},
      "blues": {"tempo_range": (60, 100), "loudness_min": 0.25},
      "hip_hop": {"tempo_range": (85, 115), "loudness_min": 0.3},
      "classical": {"tempo_range": (60, 120), "loudness_min": 0.15},
      "ambient": {"tempo_range": (40, 80), "loudness_min": 0.1},
    }

  def classify(self, features: Dict[str, Any]) -> Tuple[str, float]:
    """
    Classify genre based on audio features.

    @param {object} features - Audio features from StyleAnalyzer
    @returns {tuple} (genre_name, confidence_score)
    """
    tempo = features.get("tempo", 100)
    loudness = features.get("loudness", 0.3)
    spectral = features.get("spectral_centroid", 2000)

    matches = []

    for genre, criteria in self.genres.items():
      tempo_min, tempo_max = criteria["tempo_range"]
      loudness_min = criteria["loudness_min"]

      tempo_match = tempo_min <= tempo <= tempo_max
      loudness_match = loudness >= loudness_min

      if tempo_match and loudness_match:
        # Calculate confidence
        tempo_distance = min(
          abs(tempo - tempo_min),
          abs(tempo - tempo_max)
        )
        tempo_confidence = 1.0 - (tempo_distance / 50.0)
        loudness_confidence = min(loudness / 0.8, 1.0)

        confidence = (tempo_confidence + loudness_confidence) / 2.0
        matches.append((genre, max(0, confidence)))

    if not matches:
      return "unknown", 0.0

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0]

  def get_genre_description(self, genre: str) -> str:
    """
    Get human-readable genre description.

    @param {string} genre - Genre name
    @returns {string} Description
    """
    descriptions = {
      "electronic": "Electronic/Synth music",
      "dance": "Dance/EDM",
      "pop": "Pop music",
      "rock": "Rock music",
      "jazz": "Jazz",
      "blues": "Blues",
      "hip_hop": "Hip-Hop/Rap",
      "classical": "Classical/Orchestral",
      "ambient": "Ambient/Chill",
      "unknown": "Unknown genre",
    }
    return descriptions.get(genre, genre)


class RhythmAnalyzer:
  """
  Analyze rhythm, beat patterns, and structural breaks in audio.
  """

  def __init__(self, sample_rate: int = 16000):
    """
    Initialize rhythm analyzer.

    @param {number} sample_rate - Audio sample rate in Hz
    """
    self.sample_rate = sample_rate

  def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Analyze rhythm and beat characteristics.

    @param {array} y - Audio time series
    @param {number} sr - Sample rate
    @returns {object} Rhythm analysis results
    """
    try:
      # Detect beats and downbeats
      onset_env = librosa.onset.onset_strength(y=y, sr=sr)
      tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

      # Calculate beat interval (regularity)
      if len(beats) > 1:
        beat_intervals = np.diff(beats)
        beat_regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
      else:
        beat_regularity = 0.0

      # Detect breaks (silence/low energy sections)
      breaks = self._detect_breaks(onset_env)

      # Onset density (how syncopated)
      onset_density = len(librosa.onset.onset_detect(
        onset_envelope=onset_env,
        units='frames'
      )) / len(onset_env)

      return {
        "tempo": float(tempo),
        "beat_regularity": max(0, min(1.0, beat_regularity)),
        "onset_density": float(onset_density),
        "num_breaks": len(breaks),
        "has_strong_rhythm": beat_regularity > 0.7,
      }
    except Exception as error:
      raise ValueError(f"Rhythm analysis failed: {str(error)}")

  def _detect_breaks(self, onset_env: np.ndarray, threshold: float = 0.1) -> List[int]:
    """
    Detect breaks (silent or low-energy sections).

    @param {array} onset_env - Onset strength envelope
    @param {number} threshold - Energy threshold for break detection
    @returns {array} Indices of detected breaks
    """
    # Normalize
    onset_env_norm = onset_env / (np.max(onset_env) + 1e-6)

    # Find low-energy sections
    breaks = np.where(onset_env_norm < threshold)[0]

    # Group consecutive breaks
    if len(breaks) == 0:
      return []

    break_groups = []
    current_group = [breaks[0]]

    for b in breaks[1:]:
      if b - current_group[-1] <= 1:
        current_group.append(b)
      else:
        if len(current_group) > 5:
          break_groups.append(len(current_group))
        current_group = [b]

    return break_groups
