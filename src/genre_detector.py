"""
Detect music genre and rhythm characteristics from audio features.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import librosa


class GenreDetector:
  """
  Classify music genre based on comprehensive audio features.

  Uses tempo, loudness, spectral characteristics, harmonic content,
  and rhythm patterns to classify into genres.
  """

  def __init__(self):
    """Initialize genre detector with feature thresholds."""
    self.genres = {
      "classical": {
        "tempo_range": (60, 120),
        "loudness_range": (0.1, 0.5),
        "spectral_centroid_range": (1500, 3500),
        "zcr_range": (0.01, 0.08),
      },
      "ambient": {
        "tempo_range": (40, 90),
        "loudness_range": (0.05, 0.3),
        "spectral_centroid_range": (1000, 2500),
        "zcr_range": (0.01, 0.05),
      },
      "jazz": {
        "tempo_range": (70, 130),
        "loudness_range": (0.15, 0.4),
        "spectral_centroid_range": (2000, 3500),
        "zcr_range": (0.03, 0.1),
      },
      "blues": {
        "tempo_range": (60, 110),
        "loudness_range": (0.2, 0.45),
        "spectral_centroid_range": (1800, 3200),
        "zcr_range": (0.04, 0.12),
      },
      "rock": {
        "tempo_range": (90, 150),
        "loudness_range": (0.3, 0.6),
        "spectral_centroid_range": (2500, 4500),
        "zcr_range": (0.05, 0.15),
      },
      "pop": {
        "tempo_range": (85, 135),
        "loudness_range": (0.25, 0.55),
        "spectral_centroid_range": (2000, 3800),
        "zcr_range": (0.04, 0.12),
      },
      "hip_hop": {
        "tempo_range": (80, 120),
        "loudness_range": (0.25, 0.5),
        "spectral_centroid_range": (2000, 3500),
        "zcr_range": (0.04, 0.1),
      },
      "electronic": {
        "tempo_range": (100, 160),
        "loudness_range": (0.3, 0.65),
        "spectral_centroid_range": (2000, 5000),
        "zcr_range": (0.05, 0.16),
      },
      "dance": {
        "tempo_range": (115, 150),
        "loudness_range": (0.35, 0.7),
        "spectral_centroid_range": (2500, 5000),
        "zcr_range": (0.05, 0.15),
      },
      "breakcore": {
        "tempo_range": (160, 200),
        "loudness_range": (0.2, 0.7),
        "spectral_centroid_range": (1500, 4000),
        "zcr_range": (0.08, 0.18),
      },
    }

  def classify(self, features: Dict[str, Any]) -> Tuple[str, float]:
    """
    Classify genre based on audio features.

    Args:
      features: Audio features from StyleAnalyzer

    Returns:
      Tuple of (genre_name, confidence_score) between 0-1
    """
    tempo = features.get("tempo", 100)
    loudness = features.get("loudness", 0.3)
    spectral = features.get("spectral_centroid", 2000)
    zcr = features.get("zero_crossing_rate", 0.05)

    matches = []

    for genre, criteria in self.genres.items():
      score = self._calculate_genre_score(
        tempo, loudness, spectral, zcr, criteria
      )
      if score > 0:
        matches.append((genre, score))

    if not matches:
      return "unknown", 0.0

    matches.sort(key=lambda x: x[1], reverse=True)
    best_genre, best_score = matches[0]
    normalized_score = min(best_score / 1.0, 1.0)

    return best_genre, normalized_score

  def _calculate_genre_score(
    self,
    tempo: float,
    loudness: float,
    spectral: float,
    zcr: float,
    criteria: Dict[str, Tuple[float, float]]
  ) -> float:
    """Calculate how well features match genre criteria."""
    score = 0.0
    weights = {"tempo_range": 0.3, "loudness_range": 0.2,
               "spectral_centroid_range": 0.3, "zcr_range": 0.2}

    tempo_min, tempo_max = criteria["tempo_range"]
    if tempo_min <= tempo <= tempo_max:
      tempo_score = 1.0 - abs(tempo - (tempo_min + tempo_max) / 2) / (
        (tempo_max - tempo_min) / 2 + 1e-6
      )
      score += weights["tempo_range"] * max(0, tempo_score)

    loud_min, loud_max = criteria["loudness_range"]
    if loud_min <= loudness <= loud_max:
      loud_score = 1.0 - abs(loudness - (loud_min + loud_max) / 2) / (
        (loud_max - loud_min) / 2 + 1e-6
      )
      score += weights["loudness_range"] * max(0, loud_score)

    spec_min, spec_max = criteria["spectral_centroid_range"]
    if spec_min <= spectral <= spec_max:
      spec_score = 1.0 - abs(spectral - (spec_min + spec_max) / 2) / (
        (spec_max - spec_min) / 2 + 1e-6
      )
      score += weights["spectral_centroid_range"] * max(0, spec_score)

    zcr_min, zcr_max = criteria["zcr_range"]
    if zcr_min <= zcr <= zcr_max:
      zcr_score = 1.0 - abs(zcr - (zcr_min + zcr_max) / 2) / (
        (zcr_max - zcr_min) / 2 + 1e-6
      )
      score += weights["zcr_range"] * max(0, zcr_score)

    return score

  def get_genre_description(self, genre: str) -> str:
    """Get human-readable genre description."""
    descriptions = {
      "classical": "Classical/Orchestral",
      "ambient": "Ambient/Atmospheric",
      "jazz": "Jazz",
      "blues": "Blues",
      "rock": "Rock",
      "pop": "Pop",
      "hip_hop": "Hip-Hop/Rap",
      "electronic": "Electronic/Synth",
      "dance": "Dance/EDM",
      "breakcore": "Breakcore/Jungle",
      "unknown": "Unknown genre",
    }
    return descriptions.get(genre, genre)


class RhythmAnalyzer:
  """Analyze rhythm, beat patterns, and structural breaks in audio."""

  def __init__(self, sample_rate: int = 16000):
    """Initialize rhythm analyzer."""
    self.sample_rate = sample_rate

  def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze rhythm and beat characteristics.

    Args:
      y: Audio time series
      sr: Sample rate

    Returns:
      Dictionary with rhythm analysis results
    """
    try:
      onset_env = librosa.onset.onset_strength(y=y, sr=sr)
      tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

      if len(beats) > 1:
        beat_intervals = np.diff(beats)
        beat_regularity = 1.0 - (np.std(beat_intervals) / (
          np.mean(beat_intervals) + 1e-6
        ))
      else:
        beat_regularity = 0.0

      breaks = self._detect_breaks(onset_env)

      onset_density = len(librosa.onset.onset_detect(
        onset_envelope=onset_env,
        units='frames'
      )) / max(len(onset_env), 1)

      return {
        "tempo": float(tempo),
        "beat_regularity": max(0, min(1.0, beat_regularity)),
        "onset_density": float(onset_density),
        "num_breaks": len(breaks),
        "has_strong_rhythm": beat_regularity > 0.7,
      }
    except Exception as error:
      raise ValueError(f"Rhythm analysis failed: {str(error)}")

  def _detect_breaks(
    self,
    onset_env: np.ndarray,
    threshold: float = 0.1
  ) -> List[int]:
    """Detect breaks (silent or low-energy sections)."""
    onset_env_norm = onset_env / (np.max(onset_env) + 1e-6)
    breaks = np.where(onset_env_norm < threshold)[0]

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
